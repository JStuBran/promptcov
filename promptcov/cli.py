"""promptcov — coverage analysis for system prompts.

  promptcov run  --prompt sys.md --corpus traffic.jsonl --provider anthropic
  promptcov demo                       # offline, no API key needed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import Config, Engine
from .report import render


def _load_corpus(path: str, max_inputs: int | None) -> list[str]:
    inputs: list[str] = []
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
                if "input" not in obj:
                    raise SystemExit(
                        f'{path}:{lineno}: JSON object has no "input" key '
                        f"(keys: {sorted(obj)}). Expected one object per "
                        'line like {"input": "user message"}.')
                inputs.append(str(obj["input"]))
            else:
                inputs.append(str(obj))
    return inputs[:max_inputs] if max_inputs else inputs


def _provider(name: str, model: str, temperature: float, max_tokens: int):
    if name == "mock":
        from .providers import MockProvider
        return MockProvider()
    from .providers import AnthropicProvider
    return AnthropicProvider(model=model, temperature=temperature,
                             max_tokens=max_tokens)


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
                 rescue=not args.no_rescue, verbose=not args.quiet,
                 concurrency=args.concurrency)

    if args.dry_run:
        from .engine import estimate_calls
        from .segmenter import parse
        doc = parse(prompt_text)
        lo, hi = estimate_calls(doc, len(inputs), cfg)
        print(f"plan: {len(doc.sections)} sections · {len(doc.leaves())} "
              f"leaves · {len(inputs)} inputs · {cfg.replicates} replicates")
        print(f"model calls: {lo:,} (nothing recurses) → {hi:,} (worst "
              f"case), before cache hits — reruns are ~free")
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

    engine = Engine(_provider(args.provider, args.model,
                              args.temperature, args.max_tokens), cfg,
                    cache_dir=args.cache_dir, metric=metric)
    res = engine.run(prompt_text, inputs)

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
