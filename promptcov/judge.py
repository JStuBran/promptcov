"""LLM-judge confirmation stage.

The lexical metric over-weights harmless rephrasings and can miss pure
semantic shifts, so borderline verdicts get a second opinion: a small,
cheap model rates baseline-vs-variant reply pairs on a coarse 0-3
rubric. The judge is deliberately NOT a metric — it never supplies the
noise floor and it only touches a deterministic borderline band, so its
influence is bounded, recorded per segment, and auditable.

The band is symmetric around the decision boundary — both downgrades of
near-threshold LOAD_BEARING verdicts and upgrades of near-misses are
reachable:

  * alpha/2 < min(p_dense, p_sparse) < 2*alpha        (either direction)
  * dense-significant with effect below 2*min_effect  (downgrade side)
  * sparse tail_hits in {1, 2}                        (either direction)

Judging practice per the literature: coarse scale (LLMs cluster on round
numbers), justification before score, temperature 0, pair order
randomized per item (seeded — the question is symmetric, so
randomization suffices for position bias). The judge never silently
overwrites: the segment note carries both signals.
"""

from __future__ import annotations

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import stats as st
from .runner import Runner

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
MAX_PAIRS = 6                  # judged pairs per borderline leaf

_SCORE_RE = re.compile(r"SCORE:\s*([0-3])")

# rubric thresholds on the pair-mean: below DOWN the judge saw at most
# stylistic drift; at or above UP it saw substantive change
_DOWNGRADE_BELOW = 1.5
_UPGRADE_AT = 2.5


def _pair_prompt(user_msg: str, a: str, b: str) -> str:
    return (
        "You are auditing an AI assistant for behavioral drift. Two "
        "replies to the same user message were produced under two "
        "versions of a system prompt.\n\n"
        f"USER MESSAGE:\n{user_msg}\n\n"
        f"REPLY ONE:\n{a}\n\n"
        f"REPLY TWO:\n{b}\n\n"
        "In one sentence, state the main difference between the replies, "
        "if any. Then, on the final line, write exactly 'SCORE: N' "
        "where N is:\n"
        "0 = identical or near-identical\n"
        "1 = stylistic or wording differences only\n"
        "2 = minor substantive difference\n"
        "3 = materially different content or behavior")


def borderline(d: st.TestResult, cfg) -> bool:
    """Deterministic membership test for the judged band."""
    p_min = min(d.p_value, d.p_tail)
    if cfg.alpha / 2 < p_min < 2 * cfg.alpha:
        return True
    if d.significant and d.mode == "dense" and d.effect < 2 * cfg.min_effect:
        return True
    if d.tail_hits in (1, 2):
        return True
    return False


@dataclass
class JudgeResult:
    model: str
    mean_score: float | None
    n_pairs: int
    verdict_effect: str        # downgraded | upgraded | confirmed | unavailable

    def summary(self) -> dict:
        return {"model": self.model,
                "mean_score": (round(self.mean_score, 3)
                               if self.mean_score is not None else None),
                "n_pairs": self.n_pairs,
                "verdict_effect": self.verdict_effect}


class Judge:
    """Wraps a completion provider (temperature-0, small max_tokens) with
    the Runner cache. The provider's `name` embeds its model and sampling
    config, so swapping --judge-model on a warm cache triggers fresh
    judge calls instead of silently reusing another model's scores."""

    def __init__(self, provider, cache_dir: str, max_pairs: int = MAX_PAIRS):
        self.model = getattr(provider, "model", provider.name)
        self.runner = Runner(provider, cache_dir=cache_dir, concurrency=4)
        self.max_pairs = max_pairs

    def borderline(self, d: st.TestResult, cfg) -> bool:
        return borderline(d, cfg)

    def score_pairs(self, leaf_id: str, inputs: list,
                    variant_outs: list[str], baseline: list[list[str]],
                    scores: list[float],
                    significant: bool = False) -> JudgeResult:
        """Score the top-K most-divergent pairs and return the FINAL
        verdict_effect — the rubric thresholds live here, next to the
        rubric, not in the engine."""
        idx = sorted(range(len(inputs)), key=lambda i: -scores[i])
        idx = idx[:min(self.max_pairs, len(idx))]
        # swap decisions are drawn sequentially (seeded, deterministic)
        # BEFORE the parallel fan-out
        rng = random.Random(f"judge|{leaf_id}")
        swaps = {i: rng.random() < 0.5 for i in idx}

        errors: list[Exception] = []

        def one(i: int) -> int | None:
            a, b = baseline[0][i], variant_outs[i]
            if swaps[i]:
                a, b = b, a
            u = inputs[i]
            user_msg = u if isinstance(u, str) else u.final_user
            try:
                out = self.runner.one("", _pair_prompt(user_msg, a, b),
                                      f"judge-{leaf_id}-{i}")
            except Exception as e:
                errors.append(e)
                return None
            m = _SCORE_RE.findall(out)
            return int(m[-1]) if m else None

        with ThreadPoolExecutor(max_workers=4) as ex:
            vals = [v for v in ex.map(one, idx) if v is not None]
        if not vals:
            # a systemically broken judge (bad model id, auth) must not
            # degrade silently — surface the first error to the operator
            if errors:
                print(f"● judge unavailable for {leaf_id}: {errors[0]}",
                      file=sys.stderr)
            return JudgeResult(self.model, None, 0, "unavailable")
        mean = sum(vals) / len(vals)
        if significant and mean < _DOWNGRADE_BELOW:
            effect = "downgraded"
        elif not significant and mean >= _UPGRADE_AT:
            effect = "upgraded"
        else:
            effect = "confirmed"
        return JudgeResult(self.model, mean, len(vals), effect)
