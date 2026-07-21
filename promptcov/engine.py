"""Orchestration: the causal-inference loop.

  1. noise floor    — run the UNCHANGED prompt R times; measure self-drift
  2. bisection      — ablate whole sections; recurse only where effect
  3. leaf verdicts  — delete → (negate) → (targeted probes)
  4. prune & verify — assemble the minimal prompt, regression-test it,
                      greedily rescue if anything breaks

A segment is only ever "no observed effect ON THIS DISTRIBUTION".
The engine never says "useless" and neither should you.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

from . import stats as st
from .ablation import negate
from .metrics import LexicalMetric
from .runner import Runner
from .segmenter import Doc, Segment, parse


@dataclass
class Config:
    replicates: int = 3
    do_negate: bool = False
    do_probes: bool = False
    probe_n: int = 4
    exhaustive: bool = False
    alpha: float = st.ALPHA
    min_effect: float = st.MIN_EFFECT
    sparse_margin: float = 0.01
    correction: str = "bh"             # bh | bonferroni | none
    q_level: float = 0.10              # FDR level for bh
    probe_replicates: int = 3
    do_judge: bool = False
    rescue: bool = True
    verbose: bool = True
    concurrency: int = 8


def estimate_calls(doc: Doc, n_inputs: int, cfg: Config) -> tuple[int, int]:
    """(floor, ceiling) of model calls before cache hits.

    Floor: hierarchical run where no section recurses. Ceiling: every leaf
    gets deletion + negation + probes — including the one generation call
    each of those makes (LLM providers write the inversion / the probe
    inputs). Rescue adds more only if verification fails. Cached calls are
    free on reruns.
    """
    s, l = len(doc.sections), len(doc.leaves())
    base = cfg.replicates * n_inputs
    sections = s * n_inputs
    verify = n_inputs
    leaf_worst = l * n_inputs
    if cfg.do_negate:
        leaf_worst += l * n_inputs + l   # inversion runs + generation calls
    if cfg.do_probes:
        # per leaf: probe_replicates baseline sweeps + 1 variant sweep,
        # plus the one generation call
        leaf_worst += l * ((cfg.probe_replicates + 1) * cfg.probe_n) + l
    if cfg.do_judge:
        leaf_worst += l * min(6, n_inputs)   # judged pairs per borderline leaf
    floor = base + sections + verify
    ceil = base + sections + leaf_worst + verify
    if cfg.exhaustive:
        floor = ceil
    return floor, ceil


def estimate_batch_split(doc: Doc, n_inputs: int,
                         cfg: Config) -> tuple[int, int]:
    """(batchable, sync_only) worst-case calls under --batch. Corpus sweeps
    (baseline, section/leaf deletions, first verification) batch at 50%
    price; negation/probe/judge generation and rescue stay synchronous."""
    s, l = len(doc.sections), len(doc.leaves())
    batchable = (cfg.replicates * n_inputs + s * n_inputs + l * n_inputs
                 + n_inputs)
    _, ceil = estimate_calls(doc, n_inputs, cfg)
    return batchable, max(0, ceil - batchable)


@dataclass
class Results:
    doc: Doc | None = None
    n_tests: int = 0                   # size of the corrected leaf family
    inputs: list[str] = field(default_factory=list)
    baseline: list[list[str]] = field(default_factory=list)   # [rep][input]
    noise: list[float] = field(default_factory=list)
    section_tests: dict[str, st.TestResult] = field(default_factory=dict)
    verdicts: dict[str, st.SegmentVerdict] = field(default_factory=dict)
    pruned_prompt: str = ""
    pruned_ids: list[str] = field(default_factory=list)
    kept_redundant: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


SCHEMA_VERSION = 2


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_sha256(inputs: list) -> str:
    """Content hash of the corpus, canonicalized so key order and
    whitespace in the source file can never change the identity."""
    canon = json.dumps(inputs, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
    return _sha256_text(canon)


def _log(cfg: Config, msg: str):
    if cfg.verbose:
        print(msg, file=sys.stderr)


def _variant_scores(outs: list[str], baseline: list[list[str]],
                    metric) -> list[float]:
    """Per-input mean divergence of variant output vs every baseline rep."""
    scores = []
    for i, out in enumerate(outs):
        ds = [metric.score(out, rep[i]) for rep in baseline]
        scores.append(statistics.fmean(ds))
    return scores


def _example(tr: st.TestResult, inputs: list[str], outs: list[str],
             baseline: list[list[str]]) -> dict:
    i = tr.top_input_idx
    if i < 0:
        return {}
    return {"input": inputs[i], "baseline": baseline[0][i],
            "variant": outs[i], "divergence": round(tr.scores[i], 3)}


class Engine:
    def __init__(self, provider, cfg: Config, cache_dir: str, metric=None,
                 judge=None):
        self.p = provider
        self.cfg = cfg
        self.metric = metric or LexicalMetric()
        self.judge = judge
        self.runner = Runner(provider, cache_dir=cache_dir,
                             concurrency=cfg.concurrency,
                             verbose=cfg.verbose)

    def _warm(self, texts: list[str]):
        if hasattr(self.metric, "warm"):
            self.metric.warm(texts)

    # ------------------------------------------------------------------
    def run(self, prompt_text: str, inputs: list[str]) -> Results:
        cfg = self.cfg
        t0 = time.time()
        doc = parse(prompt_text)
        # batch-resume identity: a manifest entry only resumes when it was
        # submitted for this exact prompt + corpus
        self.runner.fingerprint = {"prompt_sha256": _sha256_text(prompt_text),
                                   "corpus_sha256": corpus_sha256(inputs)}
        res = Results(doc=doc, inputs=inputs)
        leaves = doc.leaves()
        _log(cfg, f"● segmented: {len(doc.sections)} sections, "
                  f"{len(leaves)} leaf segments")
        lo, hi = estimate_calls(doc, len(inputs), cfg)
        _log(cfg, f"● plan: {lo:,}–{hi:,} model calls before cache hits "
                  f"(+ rescue only if verification fails)")

        # 1 — baseline replicates + noise floor -------------------------
        _log(cfg, f"● baseline: {cfg.replicates} replicates × "
                  f"{len(inputs)} inputs")
        res.baseline = self.runner.batch_group(
            [(prompt_text, inputs, f"base-r{r}")
             for r in range(cfg.replicates)])
        self._warm([o for rep in res.baseline for o in rep])
        for i in range(len(inputs)):
            reps = [b[i] for b in res.baseline]
            for a in range(len(reps)):
                for b in range(a + 1, len(reps)):
                    res.noise.append(self.metric.score(reps[a], reps[b]))
        med = statistics.median(res.noise) if res.noise else 0.0
        p95 = st.percentile(res.noise, 0.95)
        _log(cfg, f"● noise floor: median {med:.3f}, p95 {p95:.3f} "
                  f"(anything below this is weather, not signal)")

        # 2 — section-level bisection: collect the leaf frontier --------
        # The recursion gate stays on RAW p — it is a search heuristic, not
        # a reported claim, and a section holding contradictory rules can
        # cancel to near-zero mean effect while its children are
        # individually load-bearing.
        sec_outs = self.runner.batch_group(
            [(doc.rebuild(removed={sec.id}), inputs, f"var-del-{sec.id}")
             for sec in doc.sections])
        pending_leaves: list[Segment] = []
        for sec, outs in zip(doc.sections, sec_outs):
            tr = self._eval_variant(outs, res)
            res.section_tests[sec.id] = tr
            signal = tr.significant or tr.p_value < cfg.alpha
            recurse = signal or cfg.exhaustive
            verdict_word = ("EFFECT → recursing" if signal else
                            "exhaustive → testing leaves anyway" if recurse
                            else "no observed effect → children inherit")
            _log(cfg, f"  ▸ {sec.id} “{sec.title}”  "
                      f"eff {tr.effect:+.3f}  p={tr.p_value:.3f}  "
                      f"— {verdict_word}")
            if recurse:
                pending_leaves.extend(sec.children)
            else:
                for leaf in sec.children:
                    res.verdicts[leaf.id] = st.SegmentVerdict(
                        leaf.id, st.INHERITED, deletion=None,
                        note="Whole section deleted with no observed "
                             "effect; leaf not tested individually.")

        leaf_outs = self.runner.batch_group(
            [(doc.rebuild(removed={leaf.id}), inputs, f"var-del-{leaf.id}")
             for leaf in pending_leaves])
        frontier = [(leaf, self._eval_variant(outs, res))
                    for leaf, outs in zip(pending_leaves, leaf_outs)]

        # 2b — correct the deletion family, then decide + cascade -------
        res.n_tests = len(frontier)
        qs = st.adjust_pvalues([st.deciding_p(d) for _, d in frontier],
                               cfg.correction)
        for (leaf, d), qv in zip(frontier, qs):
            d.q = qv
            if cfg.correction != "none":
                self._apply_corrected_decision(d, qv)
            self._leaf_cascade(doc, leaf, inputs, res, d)

        # 3 — prune + verify --------------------------------------------
        self._prune_and_verify(doc, inputs, res)

        res.meta = {
            # comparability contract: everything `check`/`compare` need to
            # validate that two reports are talking about the same run shape
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": _sha256_text(prompt_text),
            "corpus_sha256": corpus_sha256(inputs),
            "metric": self.metric.name,
            "provider": self.p.name,
            "model": getattr(self.p, "model", "mock/aria-sim"),
            "inputs": len(inputs),
            "replicates": cfg.replicates,
            "negate": cfg.do_negate, "probes": cfg.do_probes,
            "probe_n": cfg.probe_n,
            "alpha": cfg.alpha, "min_effect": cfg.min_effect,
            "correction": cfg.correction, "q": cfg.q_level,
            "n_tests": res.n_tests,
            "judge": self.judge is not None,
            "judge_model": self.judge.model if self.judge else None,
            "probe_replicates": cfg.probe_replicates,
            "exhaustive": cfg.exhaustive,
            "temperature": getattr(self.p, "temperature", None),
            "max_tokens": getattr(self.p, "max_tokens", None),
            # run-cost fields — never part of any comparability check
            "provider_calls": self.runner.calls,
            "seconds": round(time.time() - t0, 1),
        }
        return res

    # ------------------------------------------------------------------
    def _eval_variant(self, outs: list[str], res: Results) -> st.TestResult:
        self._warm(outs)
        tr = st.evaluate(_variant_scores(outs, res.baseline, self.metric),
                         res.noise, self.cfg.alpha, self.cfg.min_effect,
                         self.cfg.sparse_margin)
        tr._outs = outs  # stash for example extraction
        return tr

    def _apply_corrected_decision(self, d: st.TestResult, qv: float):
        """Re-decide a leaf deletion under the multiplicity-adjusted
        deciding p. The channel gates mirror evaluate(): dense needs the
        minimum-effect bar, sparse needs measured noise and >= 2 hits."""
        cfg = self.cfg
        thresh = cfg.q_level if cfg.correction == "bh" else cfg.alpha
        dense_ok = d.effect > cfg.min_effect
        sparse_ok = bool(d.noise) and d.tail_hits >= 2
        sig = qv < thresh and (dense_ok or sparse_ok)
        d.significant = sig
        if not sig:
            d.mode = ""
        elif dense_ok and (d.p_value <= d.p_tail or not sparse_ok):
            d.mode = "dense"
        else:
            d.mode = "sparse"

    def _leaf_cascade(self, doc: Doc, leaf: Segment, inputs: list[str],
                      res: Results, d: st.TestResult):
        jr = None
        if self.judge is not None and self.judge.borderline(d, self.cfg):
            from .judge import _DOWNGRADE_BELOW, _UPGRADE_AT
            jr = self.judge.score_pairs(leaf.id, inputs, d._outs,
                                        res.baseline, d.scores)
            if jr.mean_score is not None:
                if d.significant and jr.mean_score < _DOWNGRADE_BELOW:
                    # judge saw at most stylistic drift: the borderline
                    # metric signal stands recorded, the verdict does not
                    jr.verdict_effect = "downgraded"
                    d.significant, d.mode = False, ""
                elif not d.significant and jr.mean_score >= _UPGRADE_AT:
                    jr.verdict_effect = "upgraded"
                    d.significant = True
                    d.mode = "dense" if d.p_value <= d.p_tail else "sparse"
        self._leaf_verdict(doc, leaf, inputs, res, d)
        if jr is not None:
            sv = res.verdicts[leaf.id]
            sv.judge = jr.summary()
            if jr.verdict_effect == "unavailable":
                sentence = (f"Judge ({jr.model}) unavailable — metric "
                            f"verdict stands.")
            else:
                reading = {"downgraded": "rated the drift stylistic",
                           "upgraded": "rated the drift substantive",
                           "confirmed": "concurred with the metric"
                           }[jr.verdict_effect]
                outcome = {"downgraded": " — borderline metric signal "
                                         "recorded, verdict downgraded",
                           "upgraded": " — near-miss metric signal upgraded",
                           "confirmed": ""}[jr.verdict_effect]
                sentence = (f"Judge ({jr.model}) {reading} (mean "
                            f"{jr.mean_score:.2f}/3 over {jr.n_pairs} "
                            f"pairs){outcome}.")
            sv.note = f"{sv.note} {sentence}".strip()

    def _leaf_verdict(self, doc: Doc, leaf: Segment, inputs: list[str],
                      res: Results, d: st.TestResult):
        cfg = self.cfg
        sv = st.SegmentVerdict(leaf.id, st.NOT_TESTED, deletion=d)
        if d.significant:
            sv.verdict = st.LOAD_BEARING
            sv.example = _example(d, inputs, d._outs, res.baseline)
            if d.mode == "sparse":
                sv.note = (f"Fires hard on a thin slice of traffic: "
                           f"{d.tail_hits}/{len(inputs)} inputs moved beyond "
                           f"the noise ceiling (binomial p={d.p_tail:.4f}). "
                           f"Median-based metrics would have called this "
                           f"dead.")
                detail = f"sparse: {d.tail_hits}/{len(inputs)} inputs, " \
                         f"p_tail={d.p_tail:.4f}"
            else:
                detail = f"eff {d.effect:+.3f} p={d.p_value:.3f}"
            _log(cfg, f"      · {leaf.id} LOAD-BEARING  {detail}  "
                      f"“{leaf.display}”")
            res.verdicts[leaf.id] = sv
            return

        # deletion inert → try inversion
        if cfg.do_negate:
            inv = negate(leaf.text, self.p)
            if inv:
                variant = doc.rebuild(replaced={leaf.id: inv})
                outs = self.runner.batch(variant, inputs,
                                         f"var-neg-{leaf.id}")
                self._warm(outs)
                n = st.evaluate(_variant_scores(outs, res.baseline,
                                                self.metric),
                                res.noise, cfg.alpha, cfg.min_effect,
                                cfg.sparse_margin)
                sv.negation = n
                if n.significant:
                    sv.verdict = st.REDUNDANT
                    sv.example = _example(n, inputs, outs, res.baseline)
                    sv.note = ("Deleting it changes nothing; inverting it "
                               "does. The rule is duplicated elsewhere, or "
                               "constrains a behavior the model doesn't "
                               "attempt on this traffic.")
                    _log(cfg, f"      · {leaf.id} REDUNDANT  (delete inert, "
                              f"negation fires eff {n.effect:+.3f})  "
                              f"“{leaf.display}”")
                    res.verdicts[leaf.id] = sv
                    return
            else:
                sv.note = "No safe negation form for this segment."

        # still inert → targeted probes (is it even exercisable?)
        if cfg.do_probes:
            probes = self.p.generate_probes(leaf.text, n=cfg.probe_n) \
                if hasattr(self.p, "generate_probes") else []
            if probes:
                base_p = [self.runner.batch(doc.original, probes,
                                            f"probe-base-r{r}")
                          for r in range(cfg.probe_replicates)]
                self._warm([o for rep in base_p for o in rep])
                pnoise = [self.metric.score(base_p[a][i], base_p[b][i])
                          for i in range(len(probes))
                          for a in range(len(base_p))
                          for b in range(a + 1, len(base_p))]
                variant = doc.rebuild(removed={leaf.id})
                outs = self.runner.batch(variant, probes,
                                         f"probe-del-{leaf.id}")
                self._warm(outs)
                pt = st.evaluate(_variant_scores(outs, base_p, self.metric),
                                 pnoise, self.cfg.alpha, self.cfg.min_effect,
                                 self.cfg.sparse_margin)
                sv.probe = pt
                if pt.significant:
                    sv.verdict = st.UNEXERCISED
                    sv.example = _example(pt, probes, outs, base_p)
                    sv.note = ("Zero effect on real traffic, but targeted "
                               "probes activate it. The rule is live — your "
                               "users just never go there. Probe-based "
                               f"verdict on {len(probes)} probes × "
                               f"{cfg.probe_replicates} noise replicates: "
                               "treat as 'probably live'; raise --probe-n "
                               "before treating it as settled.")
                    _log(cfg, f"      · {leaf.id} UNEXERCISED-but-exercisable "
                              f"(probes fire eff {pt.effect:+.3f})  "
                              f"“{leaf.display}”")
                    res.verdicts[leaf.id] = sv
                    return

        sv.verdict = st.NO_OBSERVED_EFFECT
        _log(cfg, f"      · {leaf.id} no observed effect  "
                  f"(p={d.p_value:.2f})  “{leaf.display}”")
        res.verdicts[leaf.id] = sv

    # ------------------------------------------------------------------
    def _prune_and_verify(self, doc: Doc, inputs: list[str], res: Results):
        cfg = self.cfg
        prunable = {st.NO_OBSERVED_EFFECT, st.INHERITED}
        pruned = {sid for sid, v in res.verdicts.items()
                  if v.verdict in prunable}
        res.kept_redundant = [sid for sid, v in res.verdicts.items()
                              if v.verdict == st.REDUNDANT]

        # drop a section header too when every child is pruned
        for sec in doc.sections:
            if sec.children and all(c.id in pruned for c in sec.children):
                pruned.add(sec.id)

        margin = max(cfg.min_effect, cfg.sparse_margin)
        p99 = st.percentile(res.noise, 0.99)

        def verify(prompt: str,
                   grouped: bool = False) -> tuple[st.TestResult, list[int]]:
            # the first verification is batchable; rescue trials are serial
            # by nature (each depends on the last) and stay synchronous
            outs = (self.runner.batch_group([(prompt, inputs, "verify")])[0]
                    if grouped else
                    self.runner.batch(prompt, inputs, "verify"))
            self._warm(outs)
            scores = _variant_scores(outs, res.baseline, self.metric)
            tr = st.evaluate(scores, res.noise, cfg.alpha, cfg.min_effect,
                             cfg.sparse_margin)
            bad = [i for i, s in enumerate(scores) if s > p99 + margin]
            return tr, bad

        candidate = doc.rebuild(removed=pruned)
        vtr, bad = verify(candidate, grouped=True)
        rescue_log = []

        if vtr.significant and cfg.rescue:
            _log(cfg, f"● verification: pruned prompt is distinguishable "
                      f"from baseline (mode={vtr.mode}, "
                      f"{len(bad)} tail inputs) — starting greedy rescue")
            sec_of = {b.id: b.section_id for b in doc.blocks
                      if b.kind == "leaf"}
            order = sorted(
                (sid for sid in pruned if sid in sec_of),
                key=lambda s: -(res.verdicts[s].negation.effect
                                if res.verdicts.get(s) and
                                res.verdicts[s].negation else 0.0))
            for sid in order:
                if not vtr.significant:
                    break
                # reviving a leaf must also revive its section header,
                # or the rebuild keeps skipping it silently
                revive = {sid, sec_of[sid]}
                trial = doc.rebuild(removed=pruned - revive)
                ttr, tbad = verify(trial)
                improved = (len(tbad) < len(bad)) or (
                    not ttr.significant and vtr.significant)
                if improved:
                    pruned -= revive
                    rescue_log.append(
                        {"segment": sid, "fixed": len(bad) - len(tbad)})
                    vtr, bad = ttr, tbad
            candidate = doc.rebuild(removed=pruned)
            vtr, bad = verify(candidate)

        passed = not vtr.significant
        res.pruned_prompt = candidate
        res.pruned_ids = sorted(pruned)
        res.verification = {
            "passed": passed,
            "stat": vtr.summary(),
            "regressed_inputs": [] if passed else [inputs[i] for i in bad],
            "rescued": rescue_log,
            "original_chars": len(doc.original),
            "pruned_chars": len(candidate),
            "reduction_pct": round(
                100 * (1 - len(candidate) / max(1, len(doc.original))), 1),
        }
        status = ("PASS — statistically indistinguishable from baseline"
                  if passed else
                  f"FAIL ({vtr.mode}, {len(bad)} tail inputs)")
        _log(cfg, f"● pruned prompt: {len(doc.original)} → {len(candidate)} "
                  f"chars (−{res.verification['reduction_pct']}%), "
                  f"verification {status}")
