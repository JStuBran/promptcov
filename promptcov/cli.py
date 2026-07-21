"""promptcov — coverage analysis for system prompts.

  promptcov run  --prompt sys.md --corpus traffic.jsonl --provider anthropic
  promptcov demo                       # offline, no API key needed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .engine import Config, Engine
from .report import render


def _load_corpus(path: str, max_inputs: int | None) -> list:
    """Rows: {"input": str} or a plain line (single-turn, stays a plain
    str so v0.1 cache keys survive), or {"messages": [...]} (multi-turn
    Trace, final-turn replay). Mixed corpora are valid."""
    from .trace import Trace, validate_messages
    inputs: list = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if line.startswith("{"):
                    raise SystemExit(
                        f"{path}:{lineno}: line starts with '{{' but is not "
                        f"valid JSON — a truncated or malformed row would "
                        f"otherwise be silently replayed as literal text. "
                        f"Fix the row or quote it as a JSON string.")
                inputs.append(line)  # plain-text line: use it verbatim
                continue
            if isinstance(obj, dict):
                if "messages" in obj:
                    err = validate_messages(obj["messages"],
                                            f"{path}:{lineno}")
                    if err:
                        raise SystemExit(f"{path}:{lineno}: {err}")
                    inputs.append(Trace.from_messages(obj["messages"]))
                elif "input" in obj:
                    inputs.append(str(obj["input"]))
                else:
                    raise SystemExit(
                        f'{path}:{lineno}: JSON object has no "input" or '
                        f'"messages" key (keys: {sorted(obj)}). Expected '
                        'one object per line like {"input": "user '
                        'message"} or {"messages": [{"role": "user", '
                        '"content": "..."}]}.')
            else:
                inputs.append(str(obj))
    return inputs[:max_inputs] if max_inputs else inputs


def _provider(name: str, model: str, temperature: float, max_tokens: int,
              batch: bool = False):
    if name == "mock":
        from .providers import MockProvider
        # the run-parser default model is an anthropic id; only an
        # explicitly mock-flavored model re-namespaces the mock
        return MockProvider(None if model == "claude-sonnet-4-6" else model)
    from .providers import AnthropicProvider
    return AnthropicProvider(model=model, temperature=temperature,
                             max_tokens=max_tokens, batch=batch)


def _check(ap, args) -> int:
    from .check import (EXIT_INFRA, CheckError, compare_reports,
                        load_report, render_outcome)
    try:
        baseline = load_report(args.baseline)
        if args.run:
            if not (args.prompt and args.corpus):
                ap.error("check --run needs --prompt and --corpus")
            if args.candidate:
                ap.error("check --run generates the candidate itself — "
                         "drop the positional CANDIDATE argument")
            candidate_path = _check_run_candidate(baseline, args)
        else:
            if not args.candidate:
                ap.error("check needs a CANDIDATE report (or --run)")
            candidate_path = args.candidate
        candidate = load_report(candidate_path)
        code, outcome = compare_reports(baseline, candidate,
                                        strict=args.strict,
                                        prompt_path=args.prompt)
    except CheckError as e:
        print(f"● check: not comparable — {e}", file=sys.stderr)
        return EXIT_INFRA
    render_outcome(outcome)
    if args.json:
        print(json.dumps(outcome, indent=1))
    return code


def _check_run_candidate(baseline: dict, args) -> str:
    """--run mode: produce the candidate report with the baseline's own
    recorded config, so the comparability preconditions hold by
    construction."""
    import tempfile
    m = baseline["meta"]
    out = os.path.join(tempfile.mkdtemp(prefix="promptcov-check-"),
                       "candidate.html")
    provider = "mock" if str(m.get("provider", "")).startswith("mock") \
        else "anthropic"
    argv = ["run", "--prompt", args.prompt, "--corpus", args.corpus,
            "--provider", provider, "--out", out,
            "--pruned-out", os.path.join(os.path.dirname(out), "pruned.md"),
            "--cache-dir", args.cache_dir, "--quiet",
            "--model", str(m["model"]),
            "--replicates", str(m["replicates"]),
            "--probe-n", str(m["probe_n"]),
            "--probe-replicates", str(m["probe_replicates"]),
            "--alpha", str(m["alpha"]),
            "--min-effect", str(m["min_effect"]),
            "--correction", str(m["correction"]),
            "--q", str(m["q"])]
    parts = str(m["metric"]).split(":")
    argv += ["--metric", parts[0]]
    if parts[0] == "embedding" and len(parts) > 1:
        if parts[1] in ("voyage", "openai"):
            argv += ["--embedding-api", parts[1]]
            if len(parts) > 2:
                argv += ["--embedding-model", ":".join(parts[2:])]
        else:
            argv += ["--embedding-model", ":".join(parts[1:])]
    if m.get("temperature") is not None:
        argv += ["--temperature", str(m["temperature"])]
    if m.get("max_tokens") is not None:
        argv += ["--max-tokens", str(m["max_tokens"])]
    for flag, key in (("--negate", "negate"), ("--probes", "probes"),
                      ("--exhaustive", "exhaustive"), ("--judge", "judge")):
        if m.get(key):
            argv.append(flag)
    if m.get("judge"):
        argv += ["--judge-model", str(m["judge_model"])]
    rc = main(argv)
    if rc != 0:
        from .check import CheckError
        raise CheckError(f"--run analysis failed with exit code {rc}")
    return os.path.splitext(out)[0] + ".json"


def _compare(ap, args) -> int:
    from .check import EXIT_INFRA, CheckError, load_report
    from .compare import compare_payloads, render_compare
    try:
        if args.models:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            if len(models) != 2:
                ap.error("--models wants exactly two, comma-separated")
            if args.reports:
                ap.error("compare --models runs the analyses itself — "
                         "drop the positional report arguments")
            if not (args.prompt and args.corpus):
                ap.error("compare --models needs --prompt and --corpus")
            base = os.path.splitext(args.out)[0]
            paths = []
            for m in models:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", m)
                out = f"{base}-{safe}.html"
                argv = ["run", "--prompt", args.prompt,
                        "--corpus", args.corpus,
                        "--provider", args.provider, "--model", m,
                        "--out", out, "--cache-dir", args.cache_dir,
                        "--replicates", str(args.replicates),
                        "--pruned-out",
                        f"{base}-{safe}.pruned.md"]
                for flag in ("negate", "probes", "exhaustive", "quiet"):
                    if getattr(args, flag):
                        argv.append(f"--{flag}")
                try:
                    rc = main(argv)
                except Exception as e:
                    rc, err = 1, e
                else:
                    err = None
                if rc != 0:
                    done = ", ".join(paths) or "none"
                    print(f"● compare: analysis for model {m!r} failed"
                          f"{f' ({err})' if err else ''} — completed "
                          f"reports kept: {done}", file=sys.stderr)
                    return EXIT_INFRA
                paths.append(os.path.splitext(out)[0] + ".json")
            path_a, path_b = paths
        else:
            if len(args.reports) != 2:
                ap.error("compare wants two report JSONs (or --models)")
            path_a, path_b = args.reports
        outcome = compare_payloads(load_report(path_a),
                                   load_report(path_b))
    except CheckError as e:
        print(f"● compare: not comparable — {e}", file=sys.stderr)
        return EXIT_INFRA
    with open(args.out, "w") as fh:
        fh.write(render_compare(outcome))
    print(f"● {outcome['model_a']} vs {outcome['model_b']}: "
          f"{outcome['n_disagreements']} disagreement(s) — "
          f"{outcome['note']}", file=sys.stderr)
    for r in outcome["rows"]:
        if r["disagreement"]:
            print(f"  ▸ {r['id']}  {r['a']} vs {r['b']}  "
                  f"“{r['text'].strip()[:70]}”", file=sys.stderr)
    print(f"● comparison report: {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(outcome, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promptcov", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="analyze a prompt against a corpus")
    run.add_argument("--prompt", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--provider", choices=["anthropic", "mock"],
                     default="anthropic")
    run.add_argument("--model", default="claude-sonnet-4-6")
    run.add_argument("--out", default="promptcov_report.html")
    run.add_argument("--pruned-out", default=None,
                     help="where to write the verified pruned prompt")
    run.add_argument("--replicates", type=int, default=3)
    run.add_argument("--negate", action="store_true",
                     help="test rule inversion to separate REDUNDANT from inert")
    run.add_argument("--probes", action="store_true",
                     help="generate targeted probes for inert rules")
    run.add_argument("--probe-n", type=int, default=4,
                     help="probe inputs per rule (more = stronger "
                          "UNEXERCISED verdicts, more calls)")
    run.add_argument("--exhaustive", action="store_true",
                     help="test every leaf even in inert sections")
    run.add_argument("--metric", choices=["lexical", "embedding"],
                     default="lexical",
                     help="divergence metric — supplies both the variant "
                          "scores and the noise floor, on one scale")
    run.add_argument("--embedding-model", default=None,
                     help="embedding model id (default: potion-base-32M "
                          "locally, voyage-4-lite / text-embedding-3-small "
                          "for APIs)")
    run.add_argument("--embedding-api",
                     choices=["local", "voyage", "openai"], default="local",
                     help="local = model2vec via promptcov[embeddings]; "
                          "voyage/openai need the matching *_API_KEY")
    run.add_argument("--alpha", type=float, default=0.05)
    run.add_argument("--correction", choices=["bh", "bonferroni", "none"],
                     default="bh",
                     help="multiple-comparisons correction over the leaf "
                          "deletion family (default: Benjamini-Hochberg)")
    run.add_argument("--q", type=float, default=0.10,
                     help="FDR level for --correction bh")
    run.add_argument("--probe-replicates", type=int, default=3,
                     help="baseline replicates for the probe noise floor "
                          "(more = stronger UNEXERCISED verdicts)")
    run.add_argument("--judge", action="store_true",
                     help="LLM second opinion on borderline verdicts "
                          "(reference-based 0-3 rubric, temperature 0)")
    run.add_argument("--judge-model", default="claude-haiku-4-5",
                     help="model for --judge; judging is high-volume, "
                          "so a cheap model is the right default")
    run.add_argument("--min-effect", type=float, default=None,
                     help="minimum-effect gate; defaults to the selected "
                          "metric's calibrated value (0.02 for lexical)")
    run.add_argument("--max-inputs", type=int, default=None)
    run.add_argument("--temperature", type=float, default=1.0,
                     help="MATCH YOUR PRODUCTION SETTING — the noise floor "
                          "is only valid at the temperature you deploy at")
    run.add_argument("--max-tokens", type=int, default=1024,
                     help="raise if your agent's responses run long; "
                          "truncation reads as fake divergence")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--batch", action="store_true",
                     help="run corpus sweeps through the Anthropic Message "
                          "Batches API (50%% price; up to 24h per phase; "
                          "safe to Ctrl-C and rerun to resume)")
    run.add_argument("--no-wait", action="store_true",
                     help="with --batch: submit the current phase and exit; "
                          "rerun the same command later to resume")
    run.add_argument("--dry-run", action="store_true",
                     help="print the segmentation and call estimate, "
                          "make zero model calls, exit")
    run.add_argument("--no-rescue", action="store_true")
    run.add_argument("--cache-dir", default=".promptcov_cache")
    run.add_argument("--quiet", action="store_true")

    seg = sub.add_parser("segments", help="preview how a prompt file will "
                         "be segmented — no model calls, no key needed")
    seg.add_argument("--prompt", required=True)

    sub.add_parser("demo", help="run the packaged CloudNest/Aria demo offline")

    chk = sub.add_parser("check", help="CI gate: fail when a prompt edit "
                         "deletes a LOAD_BEARING rule (diffs two stored "
                         "report JSONs — no model calls, no keys)")
    chk.add_argument("baseline", help="committed baseline report JSON")
    chk.add_argument("candidate", nargs="?", default=None,
                     help="candidate report JSON (omit with --run)")
    chk.add_argument("--strict", action="store_true",
                     help="also fail when a NEW rule lands "
                          "NO_OBSERVED_EFFECT (default: warn)")
    chk.add_argument("--prompt", default=None,
                     help="freshness gate: exit 3 if this working-tree "
                          "prompt file does not hash to the candidate's "
                          "prompt_sha256 — CI should always pass it")
    chk.add_argument("--json", action="store_true",
                     help="write the machine-readable outcome to stdout")
    chk.add_argument("--run", action="store_true",
                     help="generate the candidate report first by running "
                          "the analysis with the BASELINE's recorded "
                          "config (requires --prompt and --corpus)")
    chk.add_argument("--corpus", default=None,
                     help="corpus for --run")
    chk.add_argument("--cache-dir", default=".promptcov_cache")

    cmp = sub.add_parser("compare", help="cross-model comparison: which "
                         "rules are load-bearing on one model but dead on "
                         "another (verdict-level; effect sizes are never "
                         "compared across models)")
    cmp.add_argument("reports", nargs="*",
                     help="two report JSONs from runs of different models "
                          "over the same prompt+corpus (omit with "
                          "--models)")
    cmp.add_argument("--models", default=None,
                     help="comma-separated pair, e.g. "
                          "'claude-sonnet-4-6,claude-haiku-4-5' — runs "
                          "the analysis once per model, then compares")
    cmp.add_argument("--prompt", default=None)
    cmp.add_argument("--corpus", default=None)
    cmp.add_argument("--provider", choices=["anthropic", "mock"],
                     default="anthropic")
    cmp.add_argument("--negate", action="store_true")
    cmp.add_argument("--probes", action="store_true")
    cmp.add_argument("--exhaustive", action="store_true")
    cmp.add_argument("--replicates", type=int, default=3)
    cmp.add_argument("--out", default="promptcov_compare.html",
                     help="comparison report path; orchestration mode also "
                          "writes <out-base>-<model>.html/json per model")
    cmp.add_argument("--json", action="store_true")
    cmp.add_argument("--quiet", action="store_true")
    cmp.add_argument("--cache-dir", default=".promptcov_cache")

    args = ap.parse_args(argv)

    if args.cmd == "segments":
        from .segmenter import parse
        doc = parse(open(args.prompt).read())
        for sec in doc.sections:
            print(f"\n{sec.id}  “{sec.title}”  — {len(sec.children)} leaves")
            for leaf in sec.children:
                flag = "  ⚠ LARGE" if len(leaf.text) > 600 else ""
                print(f"    {leaf.id:<8} {leaf.display}{flag}")
        n_leaves = len(doc.leaves())
        print(f"\n{len(doc.sections)} sections · {n_leaves} leaf segments")
        if n_leaves < 4:
            print("⚠ very coarse segmentation — add blank lines between "
                  "rules and ## headers between sections so ablation has "
                  "units to work with")
        if any(len(l.text) > 600 for l in doc.leaves()):
            print("⚠ LARGE leaves ablate as one block — split them with "
                  "blank lines for finer verdicts")
        return 0

    if args.cmd == "check":
        return _check(ap, args)

    if args.cmd == "compare":
        return _compare(ap, args)

    if args.cmd == "demo":
        here = os.path.join(os.path.dirname(__file__), "examples")
        return main(["run",
                     "--prompt", os.path.join(here, "aria_prompt.md"),
                     "--corpus", os.path.join(here, "traffic.jsonl"),
                     "--provider", "mock", "--negate", "--probes",
                     "--exhaustive",
                     "--out", "aria_report.html",
                     "--pruned-out", "aria_pruned.md"])

    if args.replicates < 2:
        ap.error("--replicates must be >= 2: the noise floor is built from "
                 "pairwise divergence between baseline replicates, and one "
                 "replicate yields an empty null distribution")

    with open(args.prompt) as fh:
        prompt_text = fh.read()
    inputs = _load_corpus(args.corpus, args.max_inputs)
    if len(inputs) < 8:
        print(f"warning: only {len(inputs)} inputs — statistics will be weak; "
              f"aim for 20+", file=sys.stderr)

    from .metrics import SPECS
    spec = SPECS[args.metric]
    min_effect = (args.min_effect if args.min_effect is not None
                  else spec.min_effect)
    cfg = Config(replicates=args.replicates, do_negate=args.negate,
                 do_probes=args.probes, probe_n=args.probe_n,
                 exhaustive=args.exhaustive,
                 alpha=args.alpha, min_effect=min_effect,
                 sparse_margin=spec.sparse_margin,
                 correction=args.correction, q_level=args.q,
                 probe_replicates=args.probe_replicates,
                 do_judge=args.judge,
                 rescue=not args.no_rescue, verbose=not args.quiet,
                 concurrency=args.concurrency)
    if args.judge and args.provider != "anthropic":
        ap.error("--judge requires --provider anthropic (offline tests "
                 "exercise the judge through injected doubles)")

    if args.dry_run:
        from .engine import estimate_batch_split, estimate_calls
        from .segmenter import parse
        doc = parse(prompt_text)
        lo, hi = estimate_calls(doc, len(inputs), cfg)
        print(f"plan: {len(doc.sections)} sections · {len(doc.leaves())} "
              f"leaves · {len(inputs)} inputs · {cfg.replicates} replicates")
        print(f"model calls: {lo:,} (nothing recurses) → {hi:,} (worst "
              f"case), before cache hits — reruns are ~free")
        if args.batch:
            batchable, sync_only = estimate_batch_split(doc, len(inputs),
                                                        cfg)
            print(f"with --batch: up to {batchable:,} calls run through "
                  f"Message Batches at 50% price; up to {sync_only:,} "
                  f"generation/rescue/judge calls stay synchronous at "
                  f"full price")
        print("no model calls were made. run `promptcov segments` to "
              "inspect the leaf breakdown.")
        return 0

    metric = None
    if args.metric != "lexical":
        # constructed after --dry-run returns: metric setup may download a
        # model or require an API key, and dry runs must stay free
        from .metrics import make_metric
        metric = make_metric(args.metric, cache_dir=args.cache_dir,
                             embedding_model=args.embedding_model,
                             embedding_api=args.embedding_api)

    judge = None
    if args.judge:
        from .judge import Judge
        from .providers import AnthropicProvider
        judge = Judge(AnthropicProvider(model=args.judge_model,
                                        temperature=0.0, max_tokens=400),
                      cache_dir=args.cache_dir)

    engine = Engine(_provider(args.provider, args.model,
                              args.temperature, args.max_tokens,
                              batch=args.batch), cfg,
                    cache_dir=args.cache_dir, metric=metric, judge=judge)
    engine.runner.no_wait = args.no_wait
    from .runner import BatchPending
    try:
        res = engine.run(prompt_text, inputs)
    except BatchPending as bp:
        print(f"● submitted and not waiting: "
              f"{', '.join(bp.batch_ids)}", file=sys.stderr)
        print("● results are retained for 29 days — rerun this exact "
              "command to poll, merge, and continue", file=sys.stderr)
        return 0

    html = render(res, title=os.path.basename(args.prompt))
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"● report: {args.out}", file=sys.stderr)

    from .report import payload
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w") as fh:
        json.dump(payload(res), fh, indent=1)
    print(f"● machine-readable verdicts: {json_path}", file=sys.stderr)

    pruned_path = args.pruned_out or (
        os.path.splitext(args.prompt)[0] + ".pruned.md")
    with open(pruned_path, "w") as fh:
        fh.write(res.pruned_prompt)
    print(f"● verified pruned prompt: {pruned_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
