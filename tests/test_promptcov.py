import json
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promptcov import stats as st
from promptcov.ablation import negate_heuristic
from promptcov.engine import Config, Engine
from promptcov.providers import MockProvider
from promptcov.segmenter import parse

EXAMPLES = os.path.join(os.path.dirname(__file__),
                        "..", "promptcov", "examples")
ARIA = open(os.path.join(EXAMPLES, "aria_prompt.md")).read()


def test_segmenter_roundtrip():
    doc = parse(ARIA)
    assert doc.rebuild() == ARIA
    assert len(doc.sections) == 6
    tricky = "no header preamble\n\n1. rule one\n2. rule two\n## H\npara\n"
    assert parse(tricky).rebuild() == tricky


def test_rebuild_removal_and_replacement():
    doc = parse(ARIA)
    leaf = doc.leaves()[3]
    out = doc.rebuild(removed={leaf.id})
    assert leaf.text not in out and len(out) < len(ARIA)
    out2 = doc.rebuild(replaced={leaf.id: "REPLACED\n"})
    assert "REPLACED" in out2


def test_stats_dense_and_null():
    rng = random.Random(0)
    noise = [abs(rng.gauss(0.05, 0.02)) for _ in range(80)]
    same = [abs(rng.gauss(0.05, 0.02)) for _ in range(30)]
    shifted = [abs(rng.gauss(0.15, 0.02)) for _ in range(30)]
    assert not st.evaluate(same, noise).significant
    r = st.evaluate(shifted, noise)
    assert r.significant and r.mode == "dense"


def test_stats_sparse():
    rng = random.Random(1)
    noise = [abs(rng.gauss(0.05, 0.02)) for _ in range(80)]
    scores = [abs(rng.gauss(0.05, 0.02)) for _ in range(26)] + [0.5, 0.55]
    r = st.evaluate(scores, noise)
    assert r.significant and r.mode == "sparse" and r.tail_hits >= 2


def test_bh_and_bonferroni_textbook_vectors():
    ps = [0.005, 0.011, 0.02, 0.04, 0.13]
    bh = st.adjust_pvalues(ps, "bh")
    expected = [0.025, 0.0275, 1 / 30, 0.05, 0.13]
    assert all(abs(a - b) < 1e-9 for a, b in zip(bh, expected))
    bon = st.adjust_pvalues(ps, "bonferroni")
    assert all(abs(a - b) < 1e-9
               for a, b in zip(bon, [0.025, 0.055, 0.1, 0.2, 0.65]))
    assert st.adjust_pvalues(ps, "none") == ps
    # order preservation: shuffled input maps back correctly
    shuffled = [0.13, 0.005, 0.04, 0.011, 0.02]
    bh2 = st.adjust_pvalues(shuffled, "bh")
    assert abs(bh2[1] - 0.025) < 1e-9 and abs(bh2[0] - 0.13) < 1e-9


def test_deciding_p_combines_channels():
    r = st.TestResult(scores=[], noise=[], p_value=0.03, p_tail=0.5)
    assert abs(st.deciding_p(r) - 0.06) < 1e-12
    r2 = st.TestResult(scores=[], noise=[], p_value=0.9, p_tail=0.001)
    assert abs(st.deciding_p(r2) - 0.002) < 1e-12
    r3 = st.TestResult(scores=[], noise=[], p_value=0.9, p_tail=0.8)
    assert st.deciding_p(r3) == 1.0  # capped


def test_bh_reduces_false_positives_preserves_strong_effect():
    rng = random.Random(42)
    null_ps = [rng.uniform(0.0, 1.0) for _ in range(40)]
    ps = null_ps + [0.0001]
    raw_hits = sum(1 for p in ps if p < 0.05)
    adjusted = st.adjust_pvalues(ps, "bh")
    bh_hits = sum(1 for a in adjusted if a < 0.10)
    assert adjusted[-1] < 0.10          # the real effect survives
    assert bh_hits < raw_hits           # the null flags are filtered
    assert bh_hits <= 2


def test_demo_verdicts_stable_across_correction_modes():
    inputs = [ln.split('"input": "')[1].rsplit('"', 1)[0]
              for ln in open(os.path.join(EXAMPLES, "traffic.jsonl"))
              if ln.strip()]
    pinned = {"S2.L2": st.LOAD_BEARING, "S5.L3": st.LOAD_BEARING,
              "S2.L6": st.NO_OBSERVED_EFFECT, "S6.L2": st.UNEXERCISED}
    with tempfile.TemporaryDirectory() as td:  # shared cache: 2nd run ~free
        for mode in ("none", "bh"):
            eng = Engine(MockProvider(),
                         Config(do_negate=True, do_probes=True,
                                exhaustive=True, verbose=False,
                                correction=mode), cache_dir=td)
            res = eng.run(ARIA, inputs)
            for sid, want in pinned.items():
                assert res.verdicts[sid].verdict == want, (mode, sid)
    # corrected run surfaces q and the family size
    assert res.meta["correction"] == "bh" and res.meta["n_tests"] > 0
    d = res.verdicts["S2.L2"].deletion
    assert d.q is not None and d.summary()["q"] is not None


def test_sparse_channel_does_not_escape_correction():
    # a sparse-only leaf (dense p ~1, binomial tail p tiny) must live or die
    # by its corrected deciding p, exactly like a dense leaf
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(), Config(verbose=False), cache_dir=td)
    sparse = st.TestResult(scores=[0.5] * 5, noise=[0.05] * 20,
                           p_value=0.98, p_tail=0.001, tail_hits=3,
                           effect=0.0, significant=True, mode="sparse")
    eng._apply_corrected_decision(sparse, qv=0.04)   # survives at q<0.10
    assert sparse.significant and sparse.mode == "sparse"
    sparse2 = st.TestResult(scores=[0.5] * 5, noise=[0.05] * 20,
                            p_value=0.98, p_tail=0.001, tail_hits=3,
                            effect=0.0, significant=True, mode="sparse")
    eng._apply_corrected_decision(sparse2, qv=0.4)   # corrected away
    assert not sparse2.significant and sparse2.mode == ""


def _mk_deletion(p_value, p_tail=1.0, effect=0.0, tail_hits=0,
                 significant=False, mode="", scores=None, outs=None):
    d = st.TestResult(scores=scores or [0.1, 0.2], noise=[0.02] * 20,
                      p_value=p_value, p_tail=p_tail, effect=effect,
                      tail_hits=tail_hits, significant=significant,
                      mode=mode)
    d._outs = outs or ["v-out-0", "v-out-1"]
    return d


def test_judge_borderline_band():
    from promptcov import judge as jm
    cfg = Config()  # alpha 0.05, min_effect 0.02
    # clearly significant, big effect, no sparse near-miss: never judged
    assert not jm.borderline(_mk_deletion(0.001, effect=0.3,
                                          significant=True, mode="dense"),
                             cfg)
    # clearly insignificant: never judged
    assert not jm.borderline(_mk_deletion(0.8), cfg)
    # p in the symmetric band
    assert jm.borderline(_mk_deletion(0.07), cfg)
    assert jm.borderline(_mk_deletion(0.03, effect=0.3, significant=True,
                                      mode="dense"), cfg)
    # dense-significant but effect below 2*min_effect
    assert jm.borderline(_mk_deletion(0.001, effect=0.03, significant=True,
                                      mode="dense"), cfg)
    # sparse near-misses
    assert jm.borderline(_mk_deletion(0.9, p_tail=0.6, tail_hits=1), cfg)
    assert jm.borderline(_mk_deletion(0.9, p_tail=0.01, tail_hits=2,
                                      significant=True, mode="sparse"), cfg)


class _FakeJudge:
    """Mirrors Judge's decision contract with a scripted mean score."""

    def __init__(self, mean):
        self.mean = mean
        self.model = "fake-judge"

    def borderline(self, d, cfg):
        return True

    def score_pairs(self, leaf_id, inputs, outs, baseline, scores,
                    significant=False):
        from promptcov.judge import (_DOWNGRADE_BELOW, _UPGRADE_AT,
                                     JudgeResult)
        if self.mean is None:
            return JudgeResult(self.model, None, 0, "unavailable")
        if significant and self.mean < _DOWNGRADE_BELOW:
            effect = "downgraded"
        elif not significant and self.mean >= _UPGRADE_AT:
            effect = "upgraded"
        else:
            effect = "confirmed"
        return JudgeResult(self.model, self.mean, 3, effect)


def _run_cascade(judge, d):
    doc = parse("## S\n\nRule alpha.\n")
    leaf = doc.leaves()[0]
    from promptcov.engine import Results
    res = Results(doc=doc, inputs=["q0", "q1"],
                  baseline=[["b-out-0", "b-out-1"]], noise=[0.02] * 20)
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(), Config(verbose=False), cache_dir=td,
                     judge=judge)
        eng._leaf_cascade(doc, leaf, res.inputs, res, d)
    return res.verdicts[leaf.id]


def test_judge_downgrades_borderline_load_bearing():
    d = _mk_deletion(0.03, effect=0.03, significant=True, mode="dense")
    sv = _run_cascade(_FakeJudge(mean=0.7), d)
    assert sv.verdict == st.NO_OBSERVED_EFFECT
    assert sv.judge["verdict_effect"] == "downgraded"
    assert "downgraded" in sv.note
    # the metric signal stays recorded on the deletion result
    assert sv.deletion.p_value == 0.03


def test_judge_upgrades_near_miss():
    d = _mk_deletion(0.07, effect=0.15)
    sv = _run_cascade(_FakeJudge(mean=2.8), d)
    assert sv.verdict == st.LOAD_BEARING
    assert sv.judge["verdict_effect"] == "upgraded"
    assert "upgraded" in sv.note


def test_judge_unavailable_metric_verdict_stands():
    d = _mk_deletion(0.03, effect=0.3, significant=True, mode="dense")
    sv = _run_cascade(_FakeJudge(mean=None), d)
    assert sv.verdict == st.LOAD_BEARING
    assert sv.judge["verdict_effect"] == "unavailable"
    assert "unavailable" in sv.note


def test_judge_cache_namespaced_by_model():
    from promptcov.judge import Judge

    class _JudgeProvider:
        def __init__(self, name):
            self.name = name
            self.model = name
            self.calls = 0

        def complete(self, system, user, run_tag=""):
            self.calls += 1
            return "Differs a bit.\nSCORE: 1"

    with tempfile.TemporaryDirectory() as td:
        p1 = _JudgeProvider("judge-model-a:t0")
        j1 = Judge(p1, cache_dir=td)
        j1.score_pairs("S1.L1", ["q"], ["v"], [["b"]], [0.5])
        assert p1.calls == 1
        # same model, warm cache: zero new calls
        p1b = _JudgeProvider("judge-model-a:t0")
        Judge(p1b, cache_dir=td).score_pairs("S1.L1", ["q"], ["v"],
                                             [["b"]], [0.5])
        assert p1b.calls == 0
        # different model over the same cache dir: fresh calls
        p2 = _JudgeProvider("judge-model-b:t0")
        Judge(p2, cache_dir=td).score_pairs("S1.L1", ["q"], ["v"],
                                            [["b"]], [0.5])
        assert p2.calls == 1


def test_dry_run_estimate_includes_judge_term():
    from promptcov.engine import estimate_calls
    doc = parse(ARIA)
    _, hi = estimate_calls(doc, 28, Config())
    _, hi_j = estimate_calls(doc, 28, Config(do_judge=True))
    assert hi_j > hi


def test_probe_replicates_scale_estimate_and_noise():
    from promptcov.engine import estimate_calls
    doc = parse(ARIA)
    lo3, hi3 = estimate_calls(doc, 28, Config(do_probes=True,
                                              probe_replicates=3))
    lo4, hi4 = estimate_calls(doc, 28, Config(do_probes=True,
                                              probe_replicates=4))
    assert hi4 > hi3


def test_negation_heuristics():
    assert "NEVER" in negate_heuristic("You MUST ALWAYS do X\n").upper()
    assert negate_heuristic("If the customer asks, comply.\n") is None
    assert negate_heuristic("Keep responses short.\n").startswith("Do not keep")


def test_estimate_and_cli_preflight(capsys):
    from promptcov.engine import estimate_calls
    doc = parse(ARIA)
    cfg = Config(do_negate=True, do_probes=True)
    lo, hi = estimate_calls(doc, 28, cfg)
    assert 0 < lo < hi
    cfg_x = Config(do_negate=True, do_probes=True, exhaustive=True)
    lo_x, hi_x = estimate_calls(doc, 28, cfg_x)
    assert lo_x == hi_x  # exhaustive has no best case

    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    assert main(["segments", "--prompt", prompt]) == 0
    assert "S2.L2" in capsys.readouterr().out
    assert main(["run", "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--dry-run"]) == 0
    assert "model calls" in capsys.readouterr().out


def test_provider_cache_namespace():
    from promptcov.runner import _key
    a = _key("anthropic:m1:t1.0:m1024", "sys", "u", "r0")
    b = _key("anthropic:m1:t0.3:m1024", "sys", "u", "r0")
    assert a != b  # different temperature must never share cache entries


def test_end_to_end_mock_verdicts():
    inputs = [ln.split('"input": "')[1].rsplit('"', 1)[0]
              for ln in open(os.path.join(EXAMPLES, "traffic.jsonl"))
              if ln.strip()]
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(),
                     Config(do_negate=True, do_probes=True, exhaustive=True,
                            verbose=False), cache_dir=td)
        res = eng.run(ARIA, inputs)
    v = {k: sv.verdict for k, sv in res.verdicts.items()}
    assert v["S2.L2"] == st.LOAD_BEARING        # NEVER greet by name
    assert v["S2.L3"] == st.LOAD_BEARING        # concise rule
    assert v["S2.L1"] == st.NO_OBSERVED_EFFECT  # greet-by-name (loses conflict)
    assert v["S2.L6"] == st.NO_OBSERVED_EFFECT  # priya's DO NOT DELETE line
    assert v["S5.L3"] == st.LOAD_BEARING        # cheese
    assert v["S6.L2"] == st.UNEXERCISED         # banana fossil, probe-live
    assert v["S4.L3"] == st.REDUNDANT           # caps refund rule
    assert v["S3.L1"] == st.NO_OBSERVED_EFFECT  # tone
    assert res.verification["passed"]
    assert "banana" in res.pruned_prompt
    assert "DO NOT DELETE" not in res.pruned_prompt


def test_corpus_loader():
    from promptcov.cli import _load_corpus
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                     delete=False) as fh:
        fh.write('{"input": "where is my order"}\n')
        fh.write("plain text line, not JSON\n")
        fh.write('"a bare json string"\n')
        path = fh.name
    try:
        assert _load_corpus(path, None) == [
            "where is my order", "plain text line, not JSON",
            "a bare json string"]
        assert _load_corpus(path, 2) == [
            "where is my order", "plain text line, not JSON"]
        with open(path, "a") as fh:
            fh.write('{"text": "wrong key"}\n')
        with pytest.raises(SystemExit,
                           match=r":4: .*no \"input\" or \"messages\" key"):
            _load_corpus(path, None)
    finally:
        os.unlink(path)


class _StubMetric:
    """Constant-scale stub: proves the engine routes every divergence
    computation — noise floor included — through the configured metric."""
    name = "stub"
    from promptcov.metrics import SPECS as _S
    spec = _S["lexical"]

    def __init__(self):
        self.calls = 0

    def score(self, a, b):
        self.calls += 1
        return 0.0 if a == b else 0.111111


def test_metric_threads_through_scores_and_noise():
    inputs = [f"question number {i}" for i in range(8)]
    stub = _StubMetric()
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(), Config(replicates=2, verbose=False),
                     cache_dir=td, metric=stub)
        res = eng.run(ARIA, inputs)
    assert stub.calls > 0
    assert res.meta["metric"] == "stub"
    # every noise value came from the stub — no path fell back to lexical
    assert set(res.noise) <= {0.0, 0.111111}
    # variant scores are per-input means over the 2 replicates, so each is
    # a mean of stub outputs: 0.0, 0.111111, or their midpoint
    allowed = (0.0, 0.111111 / 2, 0.111111)
    for tr in res.section_tests.values():
        assert all(min(abs(s - v) for v in allowed) < 1e-9
                   for s in tr.scores)


def test_embedding_cache_reuses_vectors():
    from promptcov.metrics import _EmbeddingMetricBase

    class _CountingEmbedder(_EmbeddingMetricBase):
        def __init__(self, cache_dir):
            super().__init__("test-model", cache_dir)
            self.embeds = 0

        def _embed_batch(self, texts):
            self.embeds += len(texts)
            return [[float(len(t)), 1.0] for t in texts]

    with tempfile.TemporaryDirectory() as td:
        m = _CountingEmbedder(td)
        m.score("aaa", "bbbb")
        m.score("aaa", "bbbb")   # same pair again: fully cached
        m.score("bbbb", "aaa")   # reversed: still cached
        assert m.embeds == 2      # one embed per unique text, ever
        # a fresh instance over the same cache dir reloads from disk
        m2 = _CountingEmbedder(td)
        m2.score("aaa", "bbbb")
        assert m2.embeds == 0


def test_local_embedding_metric_missing_extra():
    from promptcov.metrics import LocalEmbeddingMetric
    with pytest.raises(SystemExit, match=r"promptcov\[embeddings\]"):
        LocalEmbeddingMetric()


def test_api_embedding_metric_missing_key(monkeypatch):
    from promptcov.metrics import ApiEmbeddingMetric
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        ApiEmbeddingMetric(api="voyage")


def test_min_effect_sentinel_resolves_to_metric_default(tmp_path):
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")

    def run(extra):
        out = str(tmp_path / "r.html")
        main(["run", "--prompt", prompt, "--corpus", corpus,
              "--provider", "mock", "--quiet", "--max-inputs", "8",
              "--out", out, "--pruned-out", str(tmp_path / "p.md"),
              "--cache-dir", str(tmp_path / "cache")] + extra)
        return json.load(open(str(tmp_path / "r.json")))["meta"]

    assert run([])["min_effect"] == 0.02              # lexical default
    assert run(["--min-effect", "0.07"])["min_effect"] == 0.07


def test_payload_contract_v2():
    from promptcov.report import payload
    inputs = [f"question number {i}" for i in range(8)]

    def run_once():
        with tempfile.TemporaryDirectory() as td:
            eng = Engine(MockProvider(), Config(replicates=2, verbose=False),
                         cache_dir=td)
            return payload(eng.run(ARIA, inputs))

    p1, p2 = run_once(), run_once()
    m = p1["meta"]
    assert m["schema_version"] == 2
    assert len(m["prompt_sha256"]) == 64 and len(m["corpus_sha256"]) == 64
    for key in ("metric", "replicates", "negate", "probes", "probe_n",
                "alpha", "min_effect", "exhaustive", "temperature",
                "max_tokens", "provider", "model"):
        assert key in m, key
    assert m["metric"] == "lexical"
    # content hashes are deterministic across runs
    assert p2["meta"]["prompt_sha256"] == m["prompt_sha256"]
    assert p2["meta"]["corpus_sha256"] == m["corpus_sha256"]


def test_corpus_loader_rejects_truncated_json():
    from promptcov.cli import _load_corpus
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                     delete=False) as fh:
        fh.write('{"input": "x"\n')  # truncated: missing closing brace
        path = fh.name
    try:
        with pytest.raises(SystemExit, match=r":1: .*not\s+valid JSON"):
            _load_corpus(path, None)
    finally:
        os.unlink(path)


def test_replicates_guard():
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    with pytest.raises(SystemExit):
        main(["run", "--prompt", prompt, "--corpus", corpus,
              "--provider", "mock", "--replicates", "1", "--dry-run"])


def test_empty_noise_never_significant():
    # a single replicate yields no noise pairs; nothing may reach
    # significance against an unmeasured null
    r = st.evaluate([0.5] * 10, [])
    assert not r.significant and r.mode == ""


class _CountingProvider:
    name = "counter-mock"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, run_tag=""):
        self.calls += 1
        return f"echo:{user}:{run_tag}"


def test_runner_cache_hits_and_persistence():
    from promptcov.runner import Runner
    with tempfile.TemporaryDirectory() as td:
        p = _CountingProvider()
        r = Runner(p, cache_dir=td)
        assert r.one("sys", "hello", "r0") == "echo:hello:r0"
        assert r.one("sys", "hello", "r0") == "echo:hello:r0"
        assert p.calls == 1 and r.calls == 1
        # a fresh Runner over the same dir must reload the JSONL cache
        p2 = _CountingProvider()
        r2 = Runner(p2, cache_dir=td)
        assert r2.one("sys", "hello", "r0") == "echo:hello:r0"
        assert p2.calls == 0 and r2.calls == 0


class _FakeBatchProvider:
    """Delegates outputs to MockProvider but serves them through the
    Message Batches surface: submit → poll (in_progress once) → results
    in deliberately scrambled order. Shares MockProvider's cache
    namespace so batch and sync runs reuse each other's cache."""
    name = "mock"
    model = "mock/aria-sim"
    batch_enabled = True

    def __init__(self, store=None):
        self._mock = MockProvider()
        self.store = store if store is not None else {}
        self.submits = 0
        self.polls = {}

    def complete(self, system, user, run_tag=""):
        return self._mock.complete(system, user, run_tag)

    def generate_probes(self, rule, n=4):
        return self._mock.generate_probes(rule, n)

    def submit_batch(self, reqs):
        assert len({r["custom_id"] for r in reqs}) == len(reqs), \
            "duplicate custom_ids in one batch submission"
        self.submits += 1
        bid = f"batch_{len(self.store)}"
        self.store[bid] = list(reqs)
        return bid

    def poll_batch(self, bid):
        n = self.polls.get(bid, 0)
        self.polls[bid] = n + 1
        return {"processing_status": "in_progress" if n == 0 else "ended",
                "request_counts": {"processing": 0}}

    def results_for(self, req):
        return ("succeeded",
                self._mock.complete(req["system"], req["user"],
                                    req["run_tag"]))

    def batch_results(self, bid):
        reqs = list(self.store[bid])
        reqs.reverse()          # results arrive in arbitrary order
        for r in reqs:
            kind, payload = self.results_for(r)
            yield r["custom_id"], kind, payload


def _traffic_inputs():
    return [ln.split('"input": "')[1].rsplit('"', 1)[0]
            for ln in open(os.path.join(EXAMPLES, "traffic.jsonl"))
            if ln.strip()]


def test_batch_run_matches_sync_run_exactly(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    inputs = _traffic_inputs()
    cfg = lambda: Config(do_negate=True, do_probes=True, exhaustive=True,  # noqa: E731
                         verbose=False)
    with tempfile.TemporaryDirectory() as td_sync, \
            tempfile.TemporaryDirectory() as td_batch:
        sync_res = Engine(MockProvider(), cfg(),
                          cache_dir=td_sync).run(ARIA, inputs)
        fake = _FakeBatchProvider()
        batch_res = Engine(fake, cfg(),
                           cache_dir=td_batch).run(ARIA, inputs)
        assert fake.submits == 4  # baseline, sections, leaves, verify
        v_sync = {k: sv.verdict for k, sv in sync_res.verdicts.items()}
        v_batch = {k: sv.verdict for k, sv in batch_res.verdicts.items()}
        assert v_sync == v_batch
        assert sync_res.pruned_prompt == batch_res.pruned_prompt
        # cache namespace is shared: a sync run over the batch run's cache
        # dir makes zero fresh provider calls for the corpus sweeps
        eng2 = Engine(MockProvider(), cfg(), cache_dir=td_batch)
        eng2.run(ARIA, inputs)
        assert eng2.runner.calls == 0


def test_batch_dedupes_duplicate_corpus_rows(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import Runner
    with tempfile.TemporaryDirectory() as td:
        fake = _FakeBatchProvider()
        r = Runner(fake, cache_dir=td)
        outs = r.batch_group([("sys", ["dup", "dup", "other"], "t0")])
        # the assert inside submit_batch proves the dedupe; results fan
        # back to every position
        assert outs[0][0] == outs[0][1]
        assert len(outs[0]) == 3


def test_batch_resume_from_manifest(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import Runner

    class _KilledMidPoll(_FakeBatchProvider):
        def poll_batch(self, bid):
            raise KeyboardInterrupt

    from promptcov.trace import Trace
    trace = Trace.from_messages([{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}])
    with tempfile.TemporaryDirectory() as td:
        store = {}
        killed = _KilledMidPoll(store)
        r1 = Runner(killed, cache_dir=td)
        r1.fingerprint = {"prompt_sha256": "aaa", "corpus_sha256": "bbb"}
        with pytest.raises(KeyboardInterrupt):
            r1.batch_group([("sys", ["u1", "u2", trace], "t0")])
        assert killed.submits == 1
        # rerun: same fingerprint, same store — resumes, zero resubmission;
        # the Trace row round-trips through the manifest faithfully
        fake2 = _FakeBatchProvider(store)
        r2 = Runner(fake2, cache_dir=td)
        r2.fingerprint = {"prompt_sha256": "aaa", "corpus_sha256": "bbb"}
        outs = r2.batch_group([("sys", ["u1", "u2", trace], "t0")])
        assert fake2.submits == 0 and len(outs[0]) == 3
        assert all(o for o in outs[0])


def _manifest_entry(runner, bid, fingerprint, created):
    runner._manifest_append({
        "batch_id": bid, "tag": "t0", "fingerprint": fingerprint,
        "created": created,
        **runner._pack_requests([{"custom_id": "x", "system": "s",
                                  "user": "u", "run_tag": "t0"}])})


def test_batch_discards_expired_manifest_by_age_alone(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import Runner
    fp = {"prompt_sha256": "SAME", "corpus_sha256": "SAME"}
    with tempfile.TemporaryDirectory() as td:
        store = {}
        r0 = Runner(_FakeBatchProvider(store), cache_dir=td)
        _manifest_entry(r0, "batch_old", fp, created=1.0)  # ancient
        fake = _FakeBatchProvider(store)
        r = Runner(fake, cache_dir=td)
        r.fingerprint = fp           # SAME fingerprint: age alone discards
        r.batch_group([("sys", ["u1"], "t0")])
        assert fake.submits == 1
        assert "batch_old" not in r._manifest_open()


def test_batch_skips_foreign_fingerprint_without_tombstoning(monkeypatch):
    # another run shape's live batch (fresh, different fingerprint) must
    # be left alone — skipping, not discarding, so the other run can
    # still resume it
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import Runner
    with tempfile.TemporaryDirectory() as td:
        store = {}
        r0 = Runner(_FakeBatchProvider(store), cache_dir=td)
        _manifest_entry(r0, "batch_foreign",
                        {"prompt_sha256": "OTHER", "corpus_sha256": "X"},
                        created=runner_mod.time.time())
        fake = _FakeBatchProvider(store)
        r = Runner(fake, cache_dir=td)
        r.fingerprint = {"prompt_sha256": "MINE", "corpus_sha256": "Y"}
        r.batch_group([("sys", ["u1"], "t0")])
        assert fake.submits == 1                       # own work proceeds
        assert "batch_foreign" in r._manifest_open()   # still open


def test_batch_no_wait_never_blocks_on_resume(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import BatchPending, Runner

    class _StuckInProgress(_FakeBatchProvider):
        def poll_batch(self, bid):
            return {"processing_status": "in_progress"}

    fp = {"prompt_sha256": "A", "corpus_sha256": "B"}
    with tempfile.TemporaryDirectory() as td:
        store = {}
        p1 = _FakeBatchProvider(store)
        r1 = Runner(p1, cache_dir=td)
        r1.fingerprint = fp
        r1.no_wait = True
        with pytest.raises(BatchPending):
            r1.batch_group([("sys", ["u1"], "t0")])
        # rerun with --no-wait while the batch is still in flight: exits
        # via BatchPending instead of polling for up to 24h
        p2 = _StuckInProgress(store)
        r2 = Runner(p2, cache_dir=td)
        r2.fingerprint = fp
        r2.no_wait = True
        with pytest.raises(BatchPending):
            r2.batch_group([("sys", ["u1"], "t0")])
        assert p2.submits == 0


def test_batch_per_item_errors(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import Runner

    class _ExpireOnce(_FakeBatchProvider):
        def results_for(self, req):
            if req["user"] == "flaky" and self.submits == 1:
                return ("retryable", "expired")
            return super().results_for(req)

    class _InvalidItem(_FakeBatchProvider):
        def results_for(self, req):
            if req["user"] == "bad":
                return ("errored_invalid", "invalid_request_error")
            return super().results_for(req)

    class _AlwaysExpires(_FakeBatchProvider):
        def results_for(self, req):
            return ("retryable", "expired")

    with tempfile.TemporaryDirectory() as td:
        p = _ExpireOnce()
        r = Runner(p, cache_dir=td)
        outs = r.batch_group([("sys", ["ok", "flaky"], "t0")])
        assert p.submits == 2 and len(outs[0]) == 2   # one bounded resubmit
    with tempfile.TemporaryDirectory() as td:
        r = Runner(_InvalidItem(), cache_dir=td)
        with pytest.raises(RuntimeError, match="invalid_request"):
            r.batch_group([("sys", ["ok", "bad"], "t0")])
    with tempfile.TemporaryDirectory() as td:
        r = Runner(_AlwaysExpires(), cache_dir=td)
        with pytest.raises(RuntimeError, match="unresolved"):
            r.batch_group([("sys", ["ok"], "t0")])


def test_batch_no_wait_submits_and_resumes(monkeypatch):
    import promptcov.runner as runner_mod
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    from promptcov.runner import BatchPending, Runner
    with tempfile.TemporaryDirectory() as td:
        store = {}
        p1 = _FakeBatchProvider(store)
        r1 = Runner(p1, cache_dir=td)
        r1.no_wait = True
        with pytest.raises(BatchPending):
            r1.batch_group([("sys", ["u1"], "t0")])
        assert p1.submits == 1
        p2 = _FakeBatchProvider(store)
        r2 = Runner(p2, cache_dir=td)
        outs = r2.batch_group([("sys", ["u1"], "t0")])
        assert p2.submits == 0 and outs[0][0]


class _Resp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text or (json.dumps(json_data) if json_data else "")

    def json(self):
        return self._json


class _ScriptedHTTP:
    """Stub httpx-like client: scripted responses (or exceptions), calls
    recorded as (method, url, body)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _bare_anthropic(client):
    from promptcov.providers import AnthropicProvider
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model, p.max_tokens, p.temperature = "test-model", 64, 1.0
    p.batch_enabled = True
    p._client = client
    return p


def test_retry_request_retries_transport_and_status(monkeypatch):
    import promptcov.providers as prov
    monkeypatch.setattr(prov.time, "sleep", lambda s: None)

    class _Boom(Exception):
        pass

    ok = _Resp(200, {"fine": True})
    # transport errors retried via the injectable exception set
    c = _ScriptedHTTP([_Boom(), _Boom(), ok])
    r = prov.retry_request(c, "GET", "/x", retryable_exceptions=(_Boom,))
    assert r.json() == {"fine": True} and len(c.calls) == 3
    # transient statuses retried
    c2 = _ScriptedHTTP([_Resp(429), _Resp(529), ok])
    assert prov.retry_request(c2, "GET", "/x").json() == {"fine": True}
    # hard 4xx fails fast
    c3 = _ScriptedHTTP([_Resp(400, text="bad request body")])
    with pytest.raises(RuntimeError, match="API 400"):
        prov.retry_request(c3, "GET", "/x")
    # exhaustion names the last transport error
    c4 = _ScriptedHTTP([_Boom("net down")] * 6)
    with pytest.raises(RuntimeError, match="net down"):
        prov.retry_request(c4, "GET", "/x", retryable_exceptions=(_Boom,))


def test_anthropic_submit_batch_body_shape():
    from promptcov.trace import Trace
    client = _ScriptedHTTP([_Resp(200, {"id": "batch_wire"})])
    p = _bare_anthropic(client)
    trace = Trace.from_messages([{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}])
    bid = p.submit_batch([
        {"custom_id": "k1", "system": "SYS", "user": "hello",
         "run_tag": "t"},
        {"custom_id": "k2", "system": "SYS", "user": trace,
         "run_tag": "t"},
    ])
    assert bid == "batch_wire"
    method, url, body = client.calls[0]
    assert (method, url) == ("POST", "/v1/messages/batches")
    r1, r2 = body["requests"]
    assert r1["custom_id"] == "k1"
    sysblock = r1["params"]["system"][0]
    assert sysblock["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert r1["params"]["messages"] == [{"role": "user",
                                         "content": "hello"}]
    assert r2["params"]["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}]


def test_anthropic_batch_results_classification():
    results_jsonl = "\n".join([
        json.dumps({"custom_id": "ok", "result": {
            "type": "succeeded", "message": {"content": [
                {"type": "text", "text": "hello "},
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "world"}]}}}),
        json.dumps({"custom_id": "bad", "result": {
            "type": "errored",
            "error": {"error": {"type": "invalid_request_error"}}}}),
        json.dumps({"custom_id": "flaky", "result": {
            "type": "errored", "error": {"error": {"type": "api_error"}}}}),
        json.dumps({"custom_id": "late", "result": {"type": "expired"}}),
    ])
    client = _ScriptedHTTP([
        _Resp(200, {"processing_status": "ended",
                    "results_url": "https://api/results/x"}),
        _Resp(200, None, text=results_jsonl),
    ])
    p = _bare_anthropic(client)
    triples = list(p.batch_results("batch_wire"))
    assert ("ok", "succeeded", "hello world") in triples
    kinds = {cid: kind for cid, kind, _ in triples}
    assert kinds == {"ok": "succeeded", "bad": "errored_invalid",
                     "flaky": "retryable", "late": "retryable"}


def test_anthropic_complete_omits_empty_system():
    client = _ScriptedHTTP([
        _Resp(200, {"content": [{"type": "text", "text": "judged"}]})])
    p = _bare_anthropic(client)
    assert p.complete("", "judge prompt") == "judged"
    _, _, body = client.calls[0]
    assert "system" not in body   # the API rejects empty text blocks


def test_trace_key_never_collides_with_equal_plain_string():
    from promptcov.engine import corpus_sha256
    from promptcov.runner import _key
    from promptcov.trace import Trace
    t = Trace.from_messages([{"role": "user", "content": "q"}])
    impostor = t.canonical                     # plain string, same bytes
    assert _key("mock", "s", t, "r0") != _key("mock", "s", impostor, "r0")
    assert corpus_sha256([t]) != corpus_sha256([impostor])


def test_corrected_decision_is_monotone():
    # a leaf evaluate() rejected must never be flipped significant by the
    # correction stage, even when its deciding q clears the threshold
    # (mixed-channel scenario: sparse near-miss p + dense effect bar)
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(), Config(verbose=False), cache_dir=td)
    mixed = st.TestResult(scores=[0.1] * 10, noise=[0.05] * 20,
                          p_value=0.2, p_tail=0.02, tail_hits=1,
                          effect=0.05, significant=False, mode="")
    eng._apply_corrected_decision(mixed, qv=0.04)
    assert not mixed.significant and mixed.mode == ""
    # and a significant leaf keeps its evaluate()-vetted mode on survival
    dense = st.TestResult(scores=[0.3] * 10, noise=[0.05] * 20,
                          p_value=0.001, p_tail=0.9, tail_hits=0,
                          effect=0.25, significant=True, mode="dense")
    eng._apply_corrected_decision(dense, qv=0.02)
    assert dense.significant and dense.mode == "dense"


def test_estimate_batch_split_arithmetic():
    from promptcov.engine import (Config, estimate_batch_split,
                                  estimate_calls)
    doc = parse(ARIA)
    cfg = Config(do_negate=True, do_probes=True, do_judge=True)
    batchable, sync_only = estimate_batch_split(doc, 28, cfg)
    s, l = len(doc.sections), len(doc.leaves())
    assert batchable == cfg.replicates * 28 + s * 28 + l * 28 + 28
    _, ceil = estimate_calls(doc, 28, cfg)
    assert batchable + sync_only == ceil


def test_dry_run_batch_split(capsys):
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    assert main(["run", "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--dry-run", "--batch"]) == 0
    out = capsys.readouterr().out
    assert "50% price" in out and "synchronous" in out


# ------------------------------ multi-turn ------------------------------

def _write_corpus(tmp_path, lines):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_multiturn_corpus_grammar(tmp_path):
    from promptcov.cli import _load_corpus
    ok = _write_corpus(tmp_path, [
        '{"input": "plain single turn"}',
        '{"messages": [{"role": "user", "content": "q1"}, '
        '{"role": "assistant", "content": "a1"}, '
        '{"role": "user", "content": "q2"}]}',
    ])
    rows = _load_corpus(ok, None)
    assert isinstance(rows[0], str)          # legacy cache keys survive
    from promptcov.trace import Trace
    assert isinstance(rows[1], Trace) and rows[1].final_user == "q2"

    cases = [
        ('{"messages": [{"role": "system", "content": "x"}, '
         '{"role": "user", "content": "q"}]}', "system"),
        ('{"messages": [{"role": "user", "content": "q"}, '
         '{"role": "assistant", "content": "a"}]}', "assistant turn"),
        ('{"messages": [{"role": "assistant", "content": "a"}]}',
         "alternate"),
        ('{"messages": [{"role": "user", "content": 42}]}', "string"),
        ('{"messages": []}', "non-empty"),
    ]
    for row, needle in cases:
        bad = _write_corpus(tmp_path, [row])
        with pytest.raises(SystemExit, match=needle):
            _load_corpus(bad, None)


def test_trace_canonicalization_and_keys():
    from promptcov.runner import _key
    from promptcov.trace import Trace
    a = Trace.from_messages([{"role": "user", "content": "q1"},
                             {"role": "assistant", "content": "a1"},
                             {"role": "user", "content": "q2"}])
    b = Trace.from_messages([{"content": "q1", "role": "user"},
                             {"content": "a1", "role": "assistant"},
                             {"content": "q2", "role": "user"}])
    assert a.canonical == b.canonical
    assert _key("mock", "sys", str(a), "r0") == _key("mock", "sys",
                                                     str(b), "r0")
    # a truncated prefix of the same conversation is a different key
    prefix = Trace.from_messages([{"role": "user", "content": "q1"}])
    assert _key("mock", "sys", str(a), "r0") != _key("mock", "sys",
                                                     str(prefix), "r0")


def test_multiturn_end_to_end_offline():
    from promptcov.cli import _load_corpus
    from promptcov.report import payload
    corpus = os.path.join(EXAMPLES, "traffic_multiturn.jsonl")
    rows = _load_corpus(corpus, None)
    assert len(rows) == 4
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(MockProvider(), Config(replicates=2, verbose=False),
                     cache_dir=td)
        res = eng.run(ARIA, rows)
    assert res.meta["inputs"] == 4
    # one divergence score per row, traces included
    assert all(len(tr.scores) == 4 for tr in res.section_tests.values())
    json.dumps(payload(res))   # the whole payload stays JSON-serializable
    # determinism: the corpus hash covers trace content canonically
    from promptcov.engine import corpus_sha256
    assert corpus_sha256(rows) == corpus_sha256(_load_corpus(corpus, None))


# ------------------------------ check mode ------------------------------

def _report_for(prompt_text, td):
    from promptcov.report import payload
    eng = Engine(MockProvider(),
                 Config(do_negate=True, do_probes=True, exhaustive=True,
                        verbose=False), cache_dir=td)
    return payload(eng.run(prompt_text, _traffic_inputs()))


LB_LINE = "2. NEVER greet the customer by name (PII incident, see JIRA-4471)"
DEAD_LINE = ("Think step by step. Take a deep breath. "
             "This is very important to my career.")


def _check_fixtures(tmp_path):
    """(baseline, del_lb, del_dead, new_dead) payloads over edited ARIAs."""
    fixtures = {}
    variants = {
        "baseline": ARIA,
        "del_lb": ARIA.replace(LB_LINE + "\n\n", ""),
        "del_dead": ARIA.replace(DEAD_LINE + "\n\n", ""),
        "new_dead": ARIA + "\nAlways remember that CloudNest is great.\n",
    }
    for name, text in variants.items():
        assert name == "baseline" or text != ARIA
        d = tmp_path / name
        d.mkdir()
        fixtures[name] = _report_for(text, str(d))
    return fixtures


def test_check_policy_outcomes(tmp_path):
    from promptcov.check import (EXIT_PASS, EXIT_POLICY, compare_reports)
    fx = _check_fixtures(tmp_path)
    # deleting a dead rule passes
    code, out = compare_reports(fx["baseline"], fx["del_dead"])
    assert code == EXIT_PASS and out["result"] == "pass"
    # deleting a LOAD_BEARING rule fails closed, naming the rule
    code, out = compare_reports(fx["baseline"], fx["del_lb"])
    assert code == EXIT_POLICY
    assert any(LB_LINE in r["text"] for r in out["regressions"])
    # a new dead rule warns by default, fails under --strict
    code, out = compare_reports(fx["baseline"], fx["new_dead"])
    assert code == EXIT_PASS and out["new_dead"]
    code, out = compare_reports(fx["baseline"], fx["new_dead"], strict=True)
    assert code == EXIT_POLICY


def test_check_preconditions_and_freshness(tmp_path):
    import copy
    from promptcov.check import CheckError, compare_reports
    fx = _check_fixtures(tmp_path)
    base, cand = fx["baseline"], copy.deepcopy(fx["del_dead"])
    # corpus mismatch
    bad = copy.deepcopy(cand)
    bad["meta"]["corpus_sha256"] = "0" * 64
    with pytest.raises(CheckError, match="corpus mismatch"):
        compare_reports(base, bad)
    # verdict-affecting config mismatch (correction)
    bad = copy.deepcopy(cand)
    bad["meta"]["correction"] = "none"
    with pytest.raises(CheckError, match="config mismatch on 'correction'"):
        compare_reports(base, bad)
    # freshness gate: --prompt file that doesn't hash to the candidate
    other = tmp_path / "edited_prompt.md"
    other.write_text(ARIA + "\nedited\n")
    with pytest.raises(CheckError, match="stale candidate"):
        compare_reports(base, cand, prompt_path=str(other))
    # legacy payload
    from promptcov.check import load_report
    legacy = tmp_path / "legacy.json"
    old = copy.deepcopy(base)
    del old["meta"]["schema_version"]
    legacy.write_text(json.dumps(old))
    with pytest.raises(CheckError, match="legacy report"):
        load_report(str(legacy))


def test_check_unknown_and_duplicates():
    from promptcov.check import EXIT_PASS, compare_reports

    def stub(segs):
        return {"meta": {"schema_version": 2, "prompt_sha256": "p" * 64,
                         "corpus_sha256": "c" * 64, "metric": "lexical",
                         "model": "m", "temperature": 1.0,
                         "max_tokens": 64, "replicates": 2,
                         "negate": False, "probes": False, "probe_n": 4,
                         "probe_replicates": 3, "alpha": 0.05,
                         "min_effect": 0.02, "correction": "bh", "q": 0.1,
                         "exhaustive": False, "judge": False,
                         "judge_model": None},
                "segments": segs}

    leaf = lambda i, text, v: {"id": f"S1.L{i}", "kind": "leaf",  # noqa: E731
                               "text": text, "verdict": v}
    # baseline INHERITED leaf becomes LOAD_BEARING: resolution change, pass
    base = stub([leaf(1, "rule A\n", st.INHERITED)])
    cand = stub([leaf(1, "rule A\n", st.LOAD_BEARING)])
    code, out = compare_reports(base, cand)
    assert code == EXIT_PASS and out["resolution_changes"]
    # duplicate texts: one of two identical LOAD_BEARING leaves deleted,
    # the survivor matches — not a regression
    base = stub([leaf(1, "dup rule\n", st.LOAD_BEARING),
                 leaf(2, "dup rule\n", st.LOAD_BEARING)])
    cand = stub([leaf(1, "dup rule\n", st.LOAD_BEARING)])
    code, out = compare_reports(base, cand)
    assert code == EXIT_PASS and not out["regressions"]
    # unmatched new text with a live verdict is unknown, not gated
    base = stub([leaf(1, "rule A\n", st.LOAD_BEARING)])
    cand = stub([leaf(1, "rule A\n", st.LOAD_BEARING),
                 leaf(2, "brand new rule\n", st.UNEXERCISED)])
    code, out = compare_reports(base, cand)
    assert code == EXIT_PASS and out["new_unmatched"]


def test_check_neutralized_and_deleted_kept_buckets():
    from promptcov.check import EXIT_PASS, EXIT_POLICY, compare_reports

    def stub(segs):
        return {"meta": {"schema_version": 2, "prompt_sha256": "p" * 64,
                         "corpus_sha256": "c" * 64, "metric": "lexical",
                         "model": "m", "temperature": 1.0,
                         "max_tokens": 64, "replicates": 2,
                         "negate": True, "probes": True, "probe_n": 4,
                         "probe_replicates": 3, "alpha": 0.05,
                         "min_effect": 0.02, "correction": "bh", "q": 0.1,
                         "exhaustive": False, "judge": False,
                         "judge_model": None},
                "segments": segs}

    leaf = lambda i, text, v: {"id": f"S1.L{i}", "kind": "leaf",  # noqa: E731
                               "text": text, "verdict": v}
    # in-place neutralization: same text, LOAD_BEARING -> dead
    base = stub([leaf(1, "rule A\n", st.LOAD_BEARING)])
    cand = stub([leaf(1, "rule A\n", st.NO_OBSERVED_EFFECT)])
    code, out = compare_reports(base, cand)
    assert code == EXIT_PASS and out["neutralized"]
    code, out = compare_reports(base, cand, strict=True)
    assert code == EXIT_POLICY
    # deleting a kept REDUNDANT/UNEXERCISED rule is surfaced, never gated
    base = stub([leaf(1, "spare rule\n", st.REDUNDANT),
                 leaf(2, "dormant rule\n", st.UNEXERCISED)])
    cand = stub([])
    code, out = compare_reports(base, cand, strict=True)
    assert code == EXIT_PASS
    assert {d["verdict"] for d in out["deleted_kept"]} == {
        st.REDUNDANT, st.UNEXERCISED}


def test_check_run_maps_child_failure_to_exit_3(tmp_path, monkeypatch):
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    out = str(tmp_path / "b.html")
    assert main(["run", "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--quiet", "--out", out,
                 "--pruned-out", str(tmp_path / "p.md"),
                 "--cache-dir", str(tmp_path / "cache")]) == 0
    # child analysis raises SystemExit (truncated corpus row): infra, 3
    bad_corpus = tmp_path / "bad.jsonl"
    bad_corpus.write_text('{"input": "x"\n')
    rc = main(["check", str(tmp_path / "b.json"), "--run",
               "--prompt", prompt, "--corpus", str(bad_corpus),
               "--cache-dir", str(tmp_path / "cache")])
    assert rc == 3


def test_check_run_cleans_tempdir_and_replays_max_inputs(tmp_path,
                                                        monkeypatch):
    import promptcov.cli as cli_mod
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    cache = str(tmp_path / "cache")
    out = str(tmp_path / "trunc.html")
    # baseline over a TRUNCATED corpus
    assert main(["run", "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--quiet", "--max-inputs", "8",
                 "--out", out, "--pruned-out", str(tmp_path / "p.md"),
                 "--cache-dir", cache]) == 0
    meta = json.load(open(str(tmp_path / "trunc.json")))["meta"]
    assert meta["max_inputs"] == 8
    made = []
    import tempfile as _tf
    real_mkdtemp = _tf.mkdtemp
    monkeypatch.setattr(_tf, "mkdtemp",
                        lambda **kw: made.append(real_mkdtemp(**kw))
                        or made[-1])
    # --run replays --max-inputs so the corpus hash matches, and cleans up
    assert main(["check", str(tmp_path / "trunc.json"), "--run",
                 "--prompt", prompt, "--corpus", corpus,
                 "--cache-dir", cache]) == 0
    assert made and not os.path.exists(made[0])


def test_check_cli_run_parity(tmp_path, capsys):
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    cache = str(tmp_path / "cache")
    out = str(tmp_path / "base.html")
    assert main(["run", "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--quiet", "--negate", "--probes",
                 "--exhaustive", "--out", out,
                 "--pruned-out", str(tmp_path / "p.md"),
                 "--cache-dir", cache]) == 0
    base_json = str(tmp_path / "base.json")
    # two-file mode: identical prompt → pass, exit 0
    assert main(["check", base_json, base_json, "--prompt", prompt]) == 0
    # --run mode regenerates with the baseline's config and agrees
    assert main(["check", base_json, "--run", "--prompt", prompt,
                 "--corpus", corpus, "--cache-dir", cache, "--json"]) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["result"] == "pass"
    # missing candidate without --run is a usage error (argparse exit 2)
    with pytest.raises(SystemExit):
        main(["check", base_json])


# ----------------------------- compare mode -----------------------------

def _cmp_stub(model, verdicts):
    return {"meta": {"schema_version": 2, "prompt_sha256": "p" * 64,
                     "corpus_sha256": "c" * 64, "metric": "lexical",
                     "model": model, "temperature": 1.0, "max_tokens": 64,
                     "replicates": 2, "negate": False, "probes": False,
                     "probe_n": 4, "probe_replicates": 3, "alpha": 0.05,
                     "min_effect": 0.02, "correction": "bh", "q": 0.1,
                     "exhaustive": False, "judge": False,
                     "judge_model": None},
            "segments": [{"id": f"S1.L{i + 1}", "kind": "leaf",
                          "text": f"rule {i + 1}\n", "verdict": v}
                         for i, v in enumerate(verdicts)]}


def test_compare_matrix_and_preconditions():
    from promptcov.check import CheckError
    from promptcov.compare import compare_payloads
    a = _cmp_stub("model-a", [st.LOAD_BEARING, st.NO_OBSERVED_EFFECT,
                              st.INHERITED])
    b = _cmp_stub("model-b", [st.NO_OBSERVED_EFFECT, st.NO_OBSERVED_EFFECT,
                              st.LOAD_BEARING])
    out = compare_payloads(a, b)
    assert out["n_disagreements"] == 1
    top = out["rows"][0]
    assert top["disagreement"] and top["a"] == st.LOAD_BEARING
    # INHERITED-vs-verdict is unknown, never a disagreement
    unk = next(r for r in out["rows"] if r["id"] == "S1.L3")
    assert unk["unknown"] and not unk["disagreement"]
    assert "NOT comparable across models" in out["note"]
    # same model on both sides is rejected
    with pytest.raises(CheckError, match="two different models"):
        compare_payloads(a, _cmp_stub("model-a", [st.LOAD_BEARING] * 3))
    # prompt mismatch is rejected
    import copy
    b2 = copy.deepcopy(b)
    b2["meta"]["prompt_sha256"] = "x" * 64
    with pytest.raises(CheckError, match="prompt mismatch"):
        compare_payloads(a, b2)
    # config mismatch is rejected
    b3 = copy.deepcopy(b)
    b3["meta"]["replicates"] = 5
    with pytest.raises(CheckError, match="config mismatch"):
        compare_payloads(a, b3)


def test_compare_cli_orchestration_offline(tmp_path):
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    out = str(tmp_path / "cmp.html")
    assert main(["compare", "--models", "mock/sim-a,mock/sim-b",
                 "--prompt", prompt, "--corpus", corpus,
                 "--provider", "mock", "--out", out, "--quiet",
                 "--cache-dir", str(tmp_path / "cache")]) == 0
    assert os.path.exists(out)
    assert os.path.exists(str(tmp_path / "cmp-mock_sim-a.json"))
    assert os.path.exists(str(tmp_path / "cmp-mock_sim-b.json"))
    # identical simulated behavior → zero disagreements, but a full matrix
    html = open(out).read()
    assert "NOT" in html and "noise floor" in html


@pytest.mark.parametrize("exc", [RuntimeError("provider exploded"),
                                 SystemExit("corpus rejected")])
def test_compare_cli_partial_failure_keeps_first_report(tmp_path,
                                                        monkeypatch, exc):
    import promptcov.cli as cli_mod
    from promptcov.cli import main
    prompt = os.path.join(EXAMPLES, "aria_prompt.md")
    corpus = os.path.join(EXAMPLES, "traffic.jsonl")
    real_provider = cli_mod._provider

    def boom(name, model, *a, **kw):
        if model == "mock/boom":
            raise exc
        return real_provider(name, model, *a, **kw)

    monkeypatch.setattr(cli_mod, "_provider", boom)
    out = str(tmp_path / "cmp.html")
    rc = main(["compare", "--models", "mock/sim-a,mock/boom",
               "--prompt", prompt, "--corpus", corpus,
               "--provider", "mock", "--out", out, "--quiet",
               "--cache-dir", str(tmp_path / "cache")])
    assert rc == 3
    assert os.path.exists(str(tmp_path / "cmp-mock_sim-a.json"))
    assert not os.path.exists(out)


# ------------------------------ real judge -------------------------------

class _ScriptedJudgeProvider:
    def __init__(self, outputs, name="scripted-judge:t0"):
        self.name = name
        self.model = "scripted-judge"
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, system, user, run_tag=""):
        self.prompts.append(user)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _real_judge(outputs, td):
    from promptcov.judge import Judge
    return Judge(_ScriptedJudgeProvider(outputs), cache_dir=td)


def test_real_judge_decisions_and_parse_paths(capsys):
    with tempfile.TemporaryDirectory() as td:
        # downgrade: significant + stylistic mean
        j = _real_judge(["It differs slightly.\nSCORE: 1"], td)
        r = j.score_pairs("S1.L1", ["q"], ["v"], [["b"]], [0.5],
                          significant=True)
        assert r.verdict_effect == "downgraded" and r.mean_score == 1.0
    with tempfile.TemporaryDirectory() as td:
        # upgrade: near-miss + substantive mean
        j = _real_judge(["Materially different.\nSCORE: 3"], td)
        r = j.score_pairs("S1.L1", ["q"], ["v"], [["b"]], [0.5],
                          significant=False)
        assert r.verdict_effect == "upgraded"
    with tempfile.TemporaryDirectory() as td:
        # SCORE line missing: pair contributes nothing -> unavailable
        j = _real_judge(["no score anywhere in this reply"], td)
        r = j.score_pairs("S1.L1", ["q"], ["v"], [["b"]], [0.5],
                          significant=True)
        assert r.verdict_effect == "unavailable" and r.n_pairs == 0
    with tempfile.TemporaryDirectory() as td:
        # systemic failure: unavailable AND an operator-visible warning
        j = _real_judge([RuntimeError("API 404: model not found")], td)
        r = j.score_pairs("S1.L1", ["q"], ["v"], [["b"]], [0.5],
                          significant=True)
        assert r.verdict_effect == "unavailable"
        assert "model not found" in capsys.readouterr().err


def test_real_judge_sees_trace_final_turn():
    from promptcov.trace import Trace
    trace = Trace.from_messages([{"role": "user", "content": "earlier"},
                                 {"role": "assistant", "content": "mid"},
                                 {"role": "user", "content": "FINAL ASK"}])
    with tempfile.TemporaryDirectory() as td:
        j = _real_judge(["fine\nSCORE: 0"], td)
        j.score_pairs("S1.L1", [trace], ["v"], [["b"]], [0.5])
        prompt = j.runner.provider.prompts[0]
        assert "FINAL ASK" in prompt
        assert '"role"' not in prompt   # canonical JSON never leaks in


class _EitherOrProvider:
    """Behavior depends only on whether at least one of two rules survives:
    deleting either alone is inert, deleting both is catastrophic. This is
    exactly the case the greedy rescue loop exists for."""
    name = "either-mock"

    def complete(self, system, user, run_tag=""):
        if "alpha" in system or "beta" in system:
            return f"OK: {user}"
        return (f"DEGRADED {user} — completely different behavior, "
                f"much longer and much worse than before")


def test_prune_verification_rescue():
    prompt = ("## RULES\n\n"
              "Remember alpha at all times.\n\n"
              "Remember beta at all times.\n")
    inputs = [f"question number {i}" for i in range(10)]
    with tempfile.TemporaryDirectory() as td:
        eng = Engine(_EitherOrProvider(),
                     Config(replicates=2, verbose=False), cache_dir=td)
        res = eng.run(prompt, inputs)
    # each rule alone shows no effect, so both get pruned; verification
    # then fails and the rescue loop must bring one back
    assert res.verification["passed"]
    assert res.verification["rescued"]
    survivors = [w for w in ("alpha", "beta") if w in res.pruned_prompt]
    assert len(survivors) == 1
