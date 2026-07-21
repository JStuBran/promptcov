"""The integrity core.

Everything hinges on one question: did the output *change*, beyond what
the unchanged prompt already drifts on its own? We answer it against a
measured null distribution (the noise floor), never against zero.

No scipy. A permutation test on the difference of means is exact enough,
transparent, and dependency-free.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

ALPHA = 0.05
MIN_EFFECT = 0.02          # median divergence must exceed noise median by this
N_PERMUTATIONS = 3000


@dataclass
class TestResult:
    scores: list[float]                # per-input divergence vs baseline
    noise: list[float]                 # noise-floor distribution
    p_value: float = 1.0
    effect: float = 0.0                # median(scores) - median(noise)
    exceed_p95: float = 0.0            # fraction of inputs above noise p95
    tail_hits: int = 0                 # inputs beyond noise p99 (sparse path)
    p_tail: float = 1.0                # binomial tail p-value
    q: float | None = None             # multiplicity-adjusted deciding p
    significant: bool = False
    mode: str = ""                     # "dense" | "sparse" | ""
    top_input_idx: int = -1            # most-divergent input (for examples)

    def summary(self) -> dict:
        return {
            "p": round(self.p_value, 4),
            "effect": round(self.effect, 4),
            "exceed_p95": round(self.exceed_p95, 3),
            "tail_hits": self.tail_hits,
            "p_tail": round(self.p_tail, 4),
            "q": round(self.q, 4) if self.q is not None else None,
            "n": len(self.scores),
            "significant": self.significant,
            "mode": self.mode,
        }


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def permutation_test(a: list[float], b: list[float],
                     n_perm: int = N_PERMUTATIONS, seed: int = 7) -> float:
    """One-sided: is mean(a) > mean(b) beyond chance?"""
    if not a or not b:
        return 1.0
    observed = statistics.fmean(a) - statistics.fmean(b)
    if observed <= 0:
        return 1.0
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        diff = statistics.fmean(pool[:na]) - statistics.fmean(pool[na:])
        if diff >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _binom_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


def evaluate(variant_scores: list[float], noise: list[float],
             alpha: float = ALPHA, min_effect: float = MIN_EFFECT,
             sparse_margin: float = 0.01) -> TestResult:
    r = TestResult(scores=variant_scores, noise=noise)
    if not variant_scores:
        return r
    n = len(variant_scores)
    r.p_value = permutation_test(variant_scores, noise)
    med_n = statistics.median(noise) if noise else 0.0
    r.effect = statistics.median(variant_scores) - med_n
    p95 = percentile(noise, 0.95)
    r.exceed_p95 = sum(1 for s in variant_scores if s > p95) / n

    # sparse path: a rule that fires hard on a few inputs barely moves the
    # median, but the count of scores beyond the noise p99 is itself a test —
    # under the null each input exceeds p99 with prob ~0.01.
    # sparse_margin is absolute, not noise-floor-relative — metrics with a
    # tighter scale (embedding cosine) declare their own floor
    margin = max(min_effect, sparse_margin)
    p99 = percentile(noise, 0.99)
    r.tail_hits = sum(1 for s in variant_scores if s > p99 + margin)
    r.p_tail = _binom_tail(n, r.tail_hits, 0.01)

    dense = (r.p_value < alpha) and (r.effect > min_effect)
    # an empty null (e.g. a single replicate) makes p99 = 0 and every input
    # a "tail hit" — the sparse path is only meaningful against measured noise
    sparse = bool(noise) and (r.p_tail < alpha) and (r.tail_hits >= 2)
    r.significant = dense or sparse
    r.mode = "dense" if dense else ("sparse" if sparse else "")
    r.top_input_idx = max(range(n), key=lambda i: variant_scores[i])
    return r


# --------------------- multiple-comparisons layer ------------------------
#
# Each deletion verdict can become significant via two different tests —
# the permutation p (dense) or the binomial tail p (sparse) — so the
# corrected quantity is the per-leaf DECIDING p: the channel minimum with
# a factor-2 Bonferroni for having tried two families. Correcting only
# the dense channel would leave the sparse "fires on a thin slice" path
# uncorrected. Correction applies to the leaf deletion family only; the
# section-recursion gate is a search heuristic (raw p) and the negation/
# probe cascade is a follow-up family on already-insignificant deletions,
# both deliberately uncorrected.

def deciding_p(r: TestResult) -> float:
    """min(p_dense, p_sparse), doubled per leaf (two-channel Bonferroni),
    capped at 1.0."""
    return min(1.0, 2.0 * min(r.p_value, r.p_tail))


def adjust_pvalues(ps: list[float], method: str = "bh") -> list[float]:
    """Multiplicity adjustment, original order preserved.

    bh          — Benjamini-Hochberg step-up q-values (FDR)
    bonferroni  — min(1, m*p) (FWER)
    none        — pass-through
    """
    m = len(ps)
    if m == 0 or method == "none":
        return list(ps)
    if method == "bonferroni":
        return [min(1.0, p * m) for p in ps]
    if method != "bh":
        raise ValueError(f"unknown correction method {method!r}")
    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    running = 1.0
    for pos in range(m - 1, -1, -1):
        i = order[pos]
        running = min(running, ps[i] * m / (pos + 1))
        q[i] = min(1.0, running)
    return q


# ------------------------------- verdicts --------------------------------

LOAD_BEARING = "LOAD_BEARING"
NO_OBSERVED_EFFECT = "NO_OBSERVED_EFFECT"
REDUNDANT = "REDUNDANT"                  # deletion inert, negation fires
UNEXERCISED = "UNEXERCISED"              # traffic inert, targeted probes fire
INHERITED = "NO_OBSERVED_EFFECT_VIA_SECTION"
NOT_TESTED = "NOT_TESTED"

VERDICT_LABEL = {
    LOAD_BEARING: "Load-bearing",
    NO_OBSERVED_EFFECT: "No observed effect",
    REDUNDANT: "Redundant (covered elsewhere)",
    UNEXERCISED: "Unexercised, but exercisable",
    INHERITED: "No observed effect (section-level)",
    NOT_TESTED: "Not tested",
}


@dataclass
class SegmentVerdict:
    segment_id: str
    verdict: str
    deletion: TestResult | None = None
    negation: TestResult | None = None
    probe: TestResult | None = None
    example: dict = field(default_factory=dict)   # {input, baseline, variant}
    note: str = ""
