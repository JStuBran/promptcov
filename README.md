# promptcov

**Coverage analysis for system prompts.** Find out which rules in your prompt are load-bearing, which are dead weight, and which are landmines nobody has stepped on yet — with statistics, not vibes.

```
$ promptcov demo

● noise floor: median 0.051, p95 0.109 (anything below this is weather, not signal)
  ▸ S2 "RULES"
      · S2.L2 LOAD-BEARING   eff +0.081  "2. NEVER greet the customer by name (PII incident…)"
      · S2.L6 no observed effect (p=0.32) "7. Do not use lookup_order — DO NOT DELETE THIS LINE…"
  ▸ S3 "TONE (final version, do not reopen this discussion — Marcus)"  p=1.000
      · S5.L3 LOAD-BEARING   sparse: 3/28 inputs  "Never discuss cheese (see incident #2291)"
      · S6.L2 UNEXERCISED-but-exercisable (probes fire eff +0.902)  "If the user says 'banana'…"
● pruned prompt: 1787 → 873 chars (−51.1%), verification PASS
```

## Why

Every production system prompt older than three months is a haunted house. There's the rule that contradicts the rule two lines above it. The `DO NOT DELETE` comment from someone who left the company. The tone paragraph nobody is allowed to reopen. The `<!-- temp fix, remove after Tuesday -->` from fourteen months ago. Nobody deletes anything, because nobody can prove anything is safe to delete — so the file only grows, and everyone quietly resolves its contradictions in their head, or worse, the model does.

Code got out of this exact trap decades ago with test coverage. You don't argue about whether a function is dead; you look at the coverage report. **promptcov is that report, for prompts.** It ablates each segment of your prompt, replays your real traffic against the ablated variant, and measures whether the output distribution actually moved — relative to how much the *unchanged* prompt already drifts on its own.

The result isn't an opinion. It's a regression table you can put in a PR.

## Quickstart

```bash
pip install promptcov
promptcov demo          # offline, no API key — runs the packaged haunted-prompt demo
open aria_report.html
```

The demo analyzes a fictional (but painfully familiar) customer-support prompt against 28 traffic samples using a deterministic mock model, and produces the full HTML report plus a verified pruned prompt.

## Real usage

```bash
pip install 'promptcov[anthropic]'
export ANTHROPIC_API_KEY=sk-ant-…

promptcov run \
  --prompt system_prompt.md \
  --corpus traffic.jsonl \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --negate --probes \
  --out report.html \
  --pruned-out system_prompt.pruned.md
```

**Corpus format** — one JSON object per line, real user inputs from your logs:

```json
{"input": "hey where's my order #88231"}
{"input": "I want a refund. — Marcus T."}
```

20+ inputs minimum for the statistics to mean anything; 50–200 representative inputs is the sweet spot. The corpus *is* the distribution your claims are relative to — sample it honestly.

### Flags that matter

| Flag | What it buys you |
|---|---|
| `--negate` | Tests rule *inversion*, separating `REDUNDANT` (content matters, but it's covered elsewhere) from truly inert text |
| `--probes` | Asks the model to generate targeted inputs for rules your traffic never touches, separating `UNEXERCISED` (live rule, dormant traffic) from dead text |
| `--probe-n N` | Probe inputs per rule (default 4). More probes = stronger `UNEXERCISED` verdicts, more calls |
| `--probe-replicates N` | Baseline replicates behind the probe noise floor (default 3) |
| `--metric embedding` | Semantic divergence via static embeddings — local (`pip install 'promptcov[embeddings]'`, offline model2vec) or API (`--embedding-api voyage\|openai`). Lexical catches wording/format drift; embeddings catch paraphrase-vs-semantic change |
| `--judge` | LLM second opinion on borderline verdicts only (0–3 rubric, temperature 0, `--judge-model` defaults to a cheap model). Both signals are always recorded — the judge never silently overwrites |
| `--correction bh\|bonferroni\|none` | Multiple-comparisons correction over the leaf-deletion family (default Benjamini-Hochberg at `--q 0.10`). Reports show raw p and adjusted q side by side |
| `--batch` | Runs the corpus sweeps through the Anthropic Message Batches API at 50% price. Safe to Ctrl-C — rerunning resumes in-flight batches from a manifest. `--no-wait` submits and exits |
| `--exhaustive` | Tests every leaf even inside sections that showed no section-level effect (slower, catches cancellation — see Limitations) |
| `--replicates N` | Baseline replicates for the noise floor (default 3; more = tighter floor) |
| `--max-inputs N` | Cap corpus size for a cheap first pass |
| `--temperature T` | **Match your production setting.** The noise floor is only a valid null at the temperature you actually deploy at |
| `--max-tokens N` | Raise if your agent's replies run long — truncation reads as fake divergence |
| `--dry-run` | Print segmentation + call estimate, make zero model calls |
| `--concurrency N` | Parallel requests (default 8; lower it if you're rate-limited) |

Every run also writes `<report>.json` — machine-readable verdicts carrying a versioned comparability contract (schema version, prompt/corpus content hashes, metric identity, full config) that the two commands below depend on.

### CI mode

```bash
promptcov check baseline.json candidate.json --prompt system_prompt.md
```

Fails the build (exit 1) when a prompt edit deletes a `LOAD_BEARING` rule; warns on new rules that land `NO_OBSERVED_EFFECT` (`--strict` fails). Diffs two committed report JSONs — no model calls, no keys in CI. Rules are matched by whitespace-normalized text, so edits elsewhere in the file (including blank-line drift around a deleted neighbor) don't shift verdicts; the gate is **fail-closed**: a *reworded* load-bearing rule also fails until you regenerate the baseline, because a reworded load-bearing rule needs re-verification anyway. `--prompt` adds a freshness gate (exit 3 when the committed candidate report doesn't match the working-tree prompt), `--run` regenerates the candidate under the baseline's own recorded config, `--json` emits the machine-readable outcome. Exit codes: 0 pass, 1 policy, 3 infra/not-comparable.

### Cross-model comparison

```bash
promptcov compare --models claude-sonnet-4-6,claude-haiku-4-5 \
  --prompt system_prompt.md --corpus traffic.jsonl --negate --probes
```

Which rules are load-bearing on Sonnet but dead on Haiku? Verdict-level only, by design — each verdict is relative to its own model's noise floor, and effect sizes are never compared across models. Also accepts two pre-generated report JSONs (same prompt + corpus + config enforced, different models required).

### Multi-turn traces

```json
{"messages": [{"role": "user", "content": "hi, my package never arrived"},
              {"role": "assistant", "content": "Sorry to hear that — order number?"},
              {"role": "user", "content": "it's 4417, any update?"}]}
```

Corpus rows can be recorded conversations: prior turns replay verbatim (teacher-forced) and only the final assistant turn is regenerated under each ablated prompt — one score per row, statistics unchanged. Mixed single/multi-turn corpora are fine. Traces must alternate roles, start and end on a user turn, and contain no `system` role (the prompt file *is* the system prompt).

### Pilot checklist for your first real prompt

```bash
promptcov segments --prompt agent.md          # 1. does the parse match your mental model? (free)
promptcov run ... --dry-run                   # 2. how many calls is this? (free)
promptcov run ... --max-inputs 20             # 3. cheap pilot — sanity-check the noise floor
promptcov run ... --negate --probes           # 4. the real run (pilot calls are already cached)
```

If step 1 shows one giant leaf, add blank lines between rules and `##` headers between sections — the tool can only ablate the units your formatting gives it.

## How it works

1. **Noise floor first.** The unchanged prompt is run `R` times over the corpus. The divergence between identical-prompt runs is the null distribution. Every subsequent claim is tested against *this*, never against zero — because "the output changed" is meaningless for a stochastic system that changes on its own.

2. **Hierarchical bisection.** Whole sections are ablated first; the engine recurses into individual rules only where there's statistical signal. On a 200-line prompt this is the difference between ~40 variant runs and ~400.

3. **Two significance paths per segment.**
   - *Dense:* permutation test (3,000 permutations) on mean divergence, plus a minimum-effect gate — statistically detectable but trivially small shifts don't count.
   - *Sparse:* a rule that fires hard on 3 of 200 inputs barely moves the median, so median-based metrics call it dead. Instead: count inputs exceeding the noise p99 and test that count against a Binomial(n, 0.01) tail — firing hard on a handful of inputs is a signal, not an outlier.

4. **Negation** (`--negate`). If deleting a rule is inert but *inverting* it fires, the rule is `REDUNDANT`: its content is enforced by something else in the prompt (or by the model's defaults). promptcov keeps redundant rules in the pruned prompt — they're your safety margin, not your dead weight.

5. **Targeted probes** (`--probes`). For rules still inert after deletion and negation, the model generates inputs designed to trigger them. If probes fire, the verdict is `UNEXERCISED`: the rule works, your users just never go there. Also kept in the pruned prompt.

6. **Prune, verify, rescue.** Segments with no observed effect are removed, the pruned prompt is replayed against the full corpus, and the result must be statistically indistinguishable from baseline. If it isn't, a greedy rescue loop re-adds pruned segments (highest negation-effect first) until verification passes. You ship nothing that hasn't survived its own regression test.

### Verdicts

| Verdict | Meaning | In pruned prompt? |
|---|---|---|
| `LOAD_BEARING` | Deleting it measurably changes output on your traffic | kept |
| `REDUNDANT` | Deletion inert, inversion fires — covered elsewhere | kept |
| `UNEXERCISED` | Traffic never triggers it; targeted probes prove it's live | kept |
| `NO_OBSERVED_EFFECT` | Deletion, inversion, and probes all inert *on this distribution* | removed |
| `NO_OBSERVED_EFFECT_VIA_SECTION` | Whole section inert; leaf not individually tested | removed |

## The honest part

Read this before you paste the report into a PR.

- **promptcov never says "useless."** The strongest claim it makes is *no observed effect on this input distribution, at this alpha, with this divergence metric*. That is a real, useful, defensible claim. It is not a proof.
- **Unexercised ≠ useless.** The `banana` rule in the demo has zero effect on real traffic and a +0.90 effect the moment someone says banana. Your compliance rules, your edge-case handlers, your incident-response lines — these will show up as `UNEXERCISED` precisely *because* they work. That's why probes exist and why unexercised rules are never auto-pruned.
- **Your corpus is the contract.** If your traffic sample doesn't contain refund requests, promptcov cannot tell you your refund rules matter. Garbage distribution in, confident garbage out.
- **The pruned prompt is a candidate, not a decree.** It passed regression on your corpus. Ship it behind a flag, watch it, keep the diff.

## Limitations (current)

- **The lexical metric is still the default.** It catches wording, structure, and content shifts; it can miss pure *semantic* changes hiding under similar wording, and it can over-weight harmless rephrasings. `--metric embedding` and `--judge` exist for exactly this — but embedding-scale calibration constants are reasoned defaults, not measured ones; sanity-check borderline verdicts against a recorded corpus before trusting them.
- **Section-level cancellation.** Two contradictory rules in one section can cancel to near-zero *mean* effect when the whole section is ablated. The engine mitigates this by recursing on the permutation p-value alone (not just effect size), but pathological cases exist — `--exhaustive` is the guaranteed-complete mode.
- **Final-turn replay only.** Multi-turn traces regenerate the last assistant turn; per-turn teacher forcing, tool-call trajectories, and agentic loops are not yet replayed.
- **Negation is heuristic** for the mock and pattern-based fallbacks; the Anthropic provider asks the model to write the inversion, which is better but not infallible. A failed negation degrades gracefully to "no safe negation form."
- **The null is approximate.** Variant scores are per-input *means over R baseline replicates*, while noise-floor entries are *single pairwise* divergences that share replicates and inputs — same mean under the null, but not the i.i.d. exchangeability a permutation test formally assumes. Mean-difference permutation is robust to this in practice, but the p-values are honest approximations, not exact.
- **Correction covers the deletion family only.** Each leaf's deciding p (dense/sparse channel minimum, factor-2 adjusted) gets Benjamini-Hochberg by default; the section-recursion gate and the negate/probe cascade deliberately stay at raw alpha (search heuristic and follow-up family respectively). `--correction none` reproduces v0.1's deletion decisions exactly; probe verdicts additionally need `--probe-replicates 2` to match v0.1's hardcoded 2-replicate probe floor.
- **Probes are low-powered.** The probe path judges a rule on `--probe-n` inputs (default 4) against a `--probe-replicates` noise floor. `UNEXERCISED` verdicts carry this caveat in the report: treat them as "probably live," and raise `--probe-n` before treating them as settled.
- **The judge audits only the borderline band.** Confidently significant verdicts are never re-examined, so a systematic metric bias outside the band would go unseen. That's the cost of keeping judge spend bounded.
- **Cost.** Roughly `(replicates + tested_variants) × corpus_size` model calls. Hierarchical mode, response caching (built in — reruns are free), Anthropic prompt caching (built in), and `--batch` (50% off the corpus sweeps, resumable) keep this manageable. A 20-rule prompt × 50 inputs ≈ low thousands of calls on a first run, half price under `--batch`.

## Roadmap

Shipped in 0.2.0 (on [PyPI](https://pypi.org/project/promptcov/)): embedding + LLM-judge divergence, the Batches backend, `promptcov check`, teacher-forced multi-turn replay, cross-model `compare`, and Benjamini-Hochberg correction. Still ahead:

- **Per-turn teacher forcing** and tool-call / agentic trajectory replay
- **Embedding calibration study**: measured (not reasoned) `min_effect` / `sparse_margin` defaults per embedding model
- **Judge spot-audits** of confidently-significant verdicts, to catch systematic metric bias the borderline band can't see

## The refactoring skill

`skills/prompt-refactor/SKILL.md` packages the full workflow — segmentation checklist, corpus-engineering rules, the metric/flags decision tree, the verdict interpretation table, and the memo + CI-gate deliverables — as an agent skill. If you use Claude Code:

```bash
cp -r skills/prompt-refactor ~/.claude/skills/
```

then `/prompt-refactor` turns "here's my prompt and a week of logs" into a verified smaller prompt with a regression gate, with the statistical judgment calls encoded instead of improvised.

## The demo, for the record

`promptcov/examples/aria_prompt.md` is a fictional support-agent prompt containing: two contradictory greeting rules, a triple-stacked refund rule, a tone section nobody may reopen, a `DO NOT DELETE` line from a departed engineer, a temp fix from last May, and one rule about cheese. The mock provider deterministically simulates a model whose behavior is a pure function of which rules survive — so the demo's verdicts are ground-truth-checkable, and the whole pipeline (stats, probes, negation, rescue, report) runs offline in seconds. `tests/` asserts the verdicts.

Run it. Watch the cheese rule get its justice.

---

*promptcov v0.2.0 — MIT licensed, no hard dependencies, Python ≥3.10. Built because every rule deserves a trial.*
