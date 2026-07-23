---
name: prompt-refactor
description: Evidence-based system-prompt refactoring using promptcov. Use when the user wants to shrink, clean up, audit, or "refactor" a system prompt, find out which prompt rules are load-bearing vs dead weight, or set up regression gating for prompt edits. Takes a prompt file plus real or synthesized traffic; delivers verdicts per rule, a verified pruned prompt, a refactoring memo, and a CI gate.
---

# Prompt Refactor

Refactor a system prompt the way you'd refactor code — except nothing is
deleted on opinion. Every rule is ablated, replayed against traffic, and
statistically tested against the prompt's own noise floor; the output is
a verified smaller prompt plus a regression gate for future edits.

You are the judgment layer. `promptcov` is the measurement engine — do
not re-implement any statistics yourself.

## Inputs you need before starting

1. **The prompt file** (markdown-ish text).
2. **Traffic** — real user messages from logs (strongly preferred) or a
   corpus you will synthesize (Phase 2).
3. **An Anthropic API key** in `ANTHROPIC_API_KEY` and a small budget.
   Typical full run on `claude-haiku-4-5` with `--batch`: **$1–5**.
   Mock-provider dry runs and cost estimates are free.
4. promptcov installed: `pip install 'promptcov[anthropic,embeddings]'`
   (or from the repo: `uv run --extra anthropic --extra embeddings
   promptcov ...`).

## Phase 1 — Segmentation

Run `promptcov segments --prompt <file>` and check the parse against
your mental model of the prompt's rules. The tool can only ablate the
units the formatting gives it.

Reformat until every independently-deletable rule is its own leaf:

- One rule per paragraph, **blank line between rules**.
- `## HEADERS` between logical sections (hierarchical bisection tests
  sections first, so grouping related rules cuts cost).
- Split any leaf the tool flags as `LARGE` (>600 chars) unless it is
  genuinely one atomic rule.
- A preamble ("You are X, created by Y") should be its own leaf — it is
  a testable rule like any other, and often a surprising one.

Segmentation edits change bytes, not meaning. Keep rule wording verbatim.

## Phase 2 — Corpus

**The corpus is the contract: every verdict means "on this
distribution." This phase determines whether the results mean anything.**

Real logs win. Format is one JSON object per line:
`{"input": "user message"}` for single turns, or
`{"messages": [{"role": "user", ...}, ...]}` for recorded conversations
(must alternate roles, start and end on a user turn, no system role —
the prompt under test IS the system prompt).

When synthesizing, engineer coverage deliberately:

- **20 messages minimum; 30–50 is the sweet spot.** Below 20 the
  statistics are weak and promptcov will say so.
- **Walk the prompt rule by rule** and ask: "what user message would
  make this rule matter?" Include 1–3 such messages per rule — decline
  bait for refusal-style rules, emoji-laden messages for emoji rules,
  a "what's in this photo?" with no photo for image rules, ambiguous
  one-worders for clarification rules, and so on.
- **Keep the majority mundane.** Most messages should be ordinary
  traffic for the product's domain, or you are testing a distribution
  that doesn't exist. Rare-situation rules are what `--probes` is for —
  don't stuff the corpus to force them.
- Vary length, tone, formality, and topic the way real users do.

## Phase 3 — Configuration decision tree

- **Model & temperature: match production.** The noise floor is only a
  valid null at the deployment settings. Use `claude-haiku-4-5` for a
  cheap pilot even if production is bigger — then confirm survivors on
  the production model.
- **Metric:** start with `--metric embedding` for tone / style /
  persona / formatting prompts, or any model at temperature ≳0.7.
  Case data: on identical cached outputs, a tone-rules prompt had a
  lexical noise floor of **0.752** (blind — nothing can register) vs
  **0.102** with embeddings. Plain `lexical` is fine for low-temperature,
  content-heavy prompts where wording changes ARE the signal.
- **Always pass `--negate --probes`.** Negation separates REDUNDANT
  (guardrail) from dead; probes separate UNEXERCISED (live but dormant)
  from dead. Without them every kept rule is invisible.
- **Two run shapes:**
  - *Minimal-core mode* (default hierarchy): cheapest; finds the
    smallest verified prompt but leaves untested leaves as
    section-inherited verdicts.
  - *Per-rule mode* (`--exhaustive`): every rule gets its own deletion +
    negation + probe trial. Use when the deliverable is a per-rule memo.
- **Always `--batch`** (50% price, resumable; safe to Ctrl-C and rerun).
- **Sequence:** `--dry-run` for the call estimate → pilot with
  `--max-inputs 20` → full run (pilot calls are already cached, so the
  full run only pays the difference).
- **Judge** (`--judge`) only when borderline verdicts matter enough to
  spend a second opinion on.

## Phase 4 — Run and interpret

Sanity-check the noise floor line first. If the median is above ~0.5,
the metric cannot see past the model's own drift — switch to
`--metric embedding` (a rerun re-scores cached outputs nearly free)
before believing any verdict.

| Verdict | Meaning | Refactoring action |
|---|---|---|
| `LOAD_BEARING` | Deleting it measurably changes output | Keep. This is the core |
| `REDUNDANT` | Deletion inert, inversion fires | Keep — it's a guardrail against behavior the model doesn't currently attempt. Deleting it bets the model's defaults never change |
| `UNEXERCISED` | Traffic never triggers it; probes prove it's live | Keep — your users just haven't gone there yet. Often the identity/compliance/edge-case rules |
| `NO_OBSERVED_EFFECT` | Deletion, inversion, and probes all inert | Candidate for deletion — with the honesty clause below |
| `..._VIA_SECTION` | Whole section inert; leaf never individually tested | Unknown, not dead. Rerun `--exhaustive` before deleting selectively |

Interpretation rules that prevent bad refactors:

- **Rules can matter only in combination.** The prune-verify-rescue loop
  exists because deleting N individually-inert rules together can still
  shift behavior; anything the rescue loop restores earned its keep.
- **"No observed effect" ≠ useless.** The full claim is: no observed
  effect *on this distribution, with this model and metric, at this
  alpha*. Say it that way in the memo.
- **A near-miss p-value is information.** Report anything with p under
  ~2× alpha as "suggestive, undersampled" rather than dead.
- **Never hand-delete UNEXERCISED or REDUNDANT rules** to chase a
  smaller number. The pruned prompt promptcov emits already keeps them.

## Phase 5 — Deliverables

Produce all four:

1. **Refactoring memo** (the human-readable artifact): per rule — kept
   or deleted, verdict, one-line evidence (effect / p / what fired), and
   the honesty caveats. Lead with the headline (e.g. "1 of 15 rules
   carries the measurable effect; 2 are live-but-dormant; warmth is a
   guardrail").
2. **The verified pruned prompt** (`--pruned-out`). Ship it behind a
   flag or A/B, not as a hard cutover — it passed regression on the
   corpus, not on the universe.
3. **The CI gate.** Commit the baseline report JSON next to the prompt
   and wire into CI:
   ```bash
   promptcov check baseline.json candidate.json --prompt system_prompt.md
   ```
   Exit 0 pass / 1 policy failure (a LOAD_BEARING rule deleted) /
   3 not-comparable. `--strict` also fails on new dead rules and
   in-place neutralizations. This turns the one-off refactor into a
   permanent practice.
4. **(Optional) cross-model note:** `promptcov compare` haiku-vs-sonnet
   on the same corpus answers "do smaller models need the rules more?"
   Verdict-level only — never compare effect sizes across models.

## Limits — say these out loud

- Single-turn-centric: multi-turn traces replay teacher-forced with only
  the final turn regenerated; tool-calling/agentic prompts will show
  misleading UNEXERCISED verdicts because the replayer never makes tool
  calls.
- Synthetic corpora inherit your imagination's blind spots; prefer logs.
- Verdicts are model-specific. A rule dead on haiku may be load-bearing
  on a model with different defaults — confirm deletions on the
  production model before shipping.
