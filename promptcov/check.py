"""CI gate: `promptcov check BASELINE CANDIDATE`.

Diffs two stored report payloads — no API calls, no keys in CI, no
statistical flakiness from fresh runs. Regressions are judged by
direction against the BASELINE's verdicts, matched by leaf text (the
structural S{n}.L{m} ids drift on every edit; text is the stable
anchor).

Fail-closed: a baseline LOAD_BEARING leaf whose text is absent from the
candidate always fails. Deletion and rewording are indistinguishable by
text, and a reworded load-bearing rule needs re-verification anyway —
the failure message says regenerating the baseline clears a rewording.

Exit codes (argparse owns 2, so policy avoids it):
  0  pass (warnings allowed)
  1  policy failure — LOAD_BEARING text lost, or --strict + new dead rule
  3  infra/usage failure — unreadable/legacy reports, comparability
     precondition mismatch, stale candidate under --prompt
"""

from __future__ import annotations

import hashlib
import json
import sys

from . import stats as st

EXIT_PASS, EXIT_POLICY, EXIT_INFRA = 0, 1, 3

# every verdict-affecting config field recorded by the payload contract —
# a mismatch makes the two reports incomparable (run-cost fields like
# seconds/provider_calls/inputs are deliberately absent)
PARITY_FIELDS = ("metric", "model", "temperature", "max_tokens",
                 "replicates", "negate", "probes", "probe_n",
                 "probe_replicates", "alpha", "min_effect", "correction",
                 "q", "exhaustive", "judge", "judge_model")

_UNKNOWN = {st.INHERITED, st.NOT_TESTED, None}


class CheckError(Exception):
    """Infra/usage failure — maps to exit 3."""


def load_report(path: str) -> dict:
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise CheckError(f"report not found: {path}")
    except json.JSONDecodeError as e:
        raise CheckError(f"{path}: not valid JSON ({e})")
    meta = payload.get("meta")
    if not isinstance(meta, dict) or "segments" not in payload:
        raise CheckError(f"{path}: not a promptcov report payload")
    sv = meta.get("schema_version")
    if not isinstance(sv, int) or sv < 2:
        raise CheckError(
            f"{path}: legacy report (schema_version={sv!r}) — regenerate "
            f"it with promptcov >= 0.2, which records the comparability "
            f"contract check depends on")
    return payload


def _leaves(payload: dict) -> list[dict]:
    return [s for s in payload["segments"] if s.get("kind") == "leaf"]


def _by_text(leaves: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for leaf in leaves:
        out.setdefault(leaf["text"], []).append(leaf)
    return out


def check_preconditions(baseline: dict, candidate: dict,
                        prompt_path: str | None = None):
    bm, cm = baseline["meta"], candidate["meta"]
    if bm["corpus_sha256"] != cm["corpus_sha256"]:
        raise CheckError(
            f"corpus mismatch: baseline analyzed corpus "
            f"{bm['corpus_sha256'][:12]}…, candidate "
            f"{cm['corpus_sha256'][:12]}… — verdicts are only comparable "
            f"over the same traffic")
    for field in PARITY_FIELDS:
        if bm.get(field) != cm.get(field):
            raise CheckError(
                f"config mismatch on {field!r}: baseline "
                f"{bm.get(field)!r} vs candidate {cm.get(field)!r} — "
                f"this field changes verdicts, so the comparison would "
                f"reflect settings, not the prompt edit")
    if prompt_path is not None:
        try:
            with open(prompt_path) as fh:
                actual = hashlib.sha256(
                    fh.read().encode("utf-8")).hexdigest()
        except FileNotFoundError:
            raise CheckError(f"--prompt file not found: {prompt_path}")
        if actual != cm["prompt_sha256"]:
            raise CheckError(
                f"stale candidate: {prompt_path} hashes to "
                f"{actual[:12]}… but the candidate report was generated "
                f"from {cm['prompt_sha256'][:12]}… — regenerate the "
                f"candidate report before gating on it")


def compare_reports(baseline: dict, candidate: dict, strict: bool = False,
                    prompt_path: str | None = None) -> tuple[int, dict]:
    """Returns (exit_code, machine-readable outcome)."""
    check_preconditions(baseline, candidate, prompt_path)

    base_by_text = _by_text(_leaves(baseline))
    cand_by_text = _by_text(_leaves(candidate))

    regressions, unknown, resolution_changes = [], [], []
    for text, occurrences in base_by_text.items():
        verdicts = [o.get("verdict") for o in occurrences]
        if text not in cand_by_text:
            if st.LOAD_BEARING in verdicts:
                o = next(o for o in occurrences
                         if o.get("verdict") == st.LOAD_BEARING)
                eff = ((o.get("deletion") or {}).get("effect"))
                regressions.append({"id": o["id"], "text": text,
                                    "baseline_effect": eff})
            elif all(v in _UNKNOWN for v in verdicts):
                unknown.append({"id": occurrences[0]["id"], "text": text,
                                "side": "baseline",
                                "verdict": verdicts[0]})
        else:
            cand_vs = [o.get("verdict") for o in cand_by_text[text]]
            b, c = verdicts[0], cand_vs[0]
            if (b in _UNKNOWN) != (c in _UNKNOWN):
                resolution_changes.append(
                    {"text": text, "baseline": b, "candidate": c})

    new_dead, new_unmatched = [], []
    for text, occurrences in cand_by_text.items():
        if text in base_by_text:
            continue
        for o in occurrences:
            v = o.get("verdict")
            if v == st.NO_OBSERVED_EFFECT:
                new_dead.append({"id": o["id"], "text": text})
            else:
                new_unmatched.append({"id": o["id"], "text": text,
                                      "verdict": v})

    if regressions:
        code = EXIT_POLICY
    elif new_dead and strict:
        code = EXIT_POLICY
    else:
        code = EXIT_PASS
    return code, {
        "result": "fail" if code else "pass",
        "regressions": regressions,
        "new_dead": new_dead,
        "new_unmatched": new_unmatched,
        "resolution_changes": resolution_changes,
        "unknown": unknown,
        "strict": strict,
    }


def render_outcome(outcome: dict, out=sys.stderr):
    p = lambda msg: print(msg, file=out)  # noqa: E731
    if outcome["regressions"]:
        p(f"● REGRESSION: {len(outcome['regressions'])} baseline "
          f"LOAD_BEARING rule(s) missing from the candidate (fail-closed: "
          f"a rewording also lands here — regenerate the baseline to "
          f"clear it)")
        for r in outcome["regressions"]:
            eff = (f"  eff {r['baseline_effect']:+.3f}"
                   if r.get("baseline_effect") is not None else "")
            p(f"  ▸ {r['id']}{eff}  “{r['text'].strip()[:80]}”")
    for nd in outcome["new_dead"]:
        tag = "FAIL (--strict)" if outcome["strict"] else "warning"
        p(f"● {tag}: new rule lands NO_OBSERVED_EFFECT  "
          f"“{nd['text'].strip()[:80]}”")
    for nu in outcome["new_unmatched"]:
        p(f"  · unmatched new text ({nu['verdict']}): "
          f"“{nu['text'].strip()[:80]}” — unknown, not gated")
    for rc in outcome["resolution_changes"]:
        p(f"  · resolution change (not a regression): "
          f"{rc['baseline']} → {rc['candidate']}  "
          f"“{rc['text'].strip()[:60]}”")
    if outcome["result"] == "pass":
        p("● check: PASS")
    else:
        p("● check: FAIL")
