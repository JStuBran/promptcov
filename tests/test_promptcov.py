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
        with pytest.raises(SystemExit, match=r":4: .*no \"input\" key"):
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
