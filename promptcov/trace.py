"""Multi-turn corpus rows (teacher-forced, final-turn replay).

A Trace is a recorded conversation whose FINAL assistant turn is
regenerated under each ablated prompt — the prior turns are replayed
verbatim (teacher forcing). One regenerated turn per row keeps one
divergence score per row, so the pairing, noise floor, and sparse
statistics are untouched. Per-turn regeneration (T scores per row) is a
deliberate non-feature for now.

str(Trace) is the canonical JSON serialization (sorted keys, fixed
separators): semantically identical rows produce identical cache keys
regardless of key order in the source file, and the whole trace — not
just the final turn — enters the key, so different prefixes of one
conversation never collide. Single-turn rows stay plain `str` all the
way through, which keeps v0.1 cache keys byte-identical for existing
caches.

Grammar (enforced by the corpus loader with location-prefixed errors):
messages alternate roles starting with "user", contain only string
content, contain no "system" role (it would collide with the ablated
prompt under test), and end on a "user" turn (a trailing assistant turn
leaves nothing to regenerate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass


def canonical_json(obj) -> str:
    """The one canonical serialization: cache keys and corpus hashes both
    depend on these exact kwargs staying identical."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


@dataclass(frozen=True)
class Trace:
    messages: tuple[tuple[str, str], ...]    # ((role, content), ...)

    @staticmethod
    def from_messages(msgs: list[dict]) -> "Trace":
        return Trace(tuple((m["role"], m["content"]) for m in msgs))

    @property
    def canonical(self) -> str:
        return canonical_json(
            [{"content": c, "role": r} for r, c in self.messages])

    @property
    def final_user(self) -> str:
        return self.messages[-1][1]

    def __str__(self) -> str:
        return self.canonical


def validate_messages(msgs) -> str | None:
    """Returns None when valid, else the error text (caller prefixes the
    file:line location)."""
    if not isinstance(msgs, list) or not msgs:
        return '"messages" must be a non-empty list of {"role", "content"}'
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return f'messages[{i}] must be an object with "role" and ' \
                   f'"content"'
        if m["role"] == "system":
            return ('a "system" role inside a trace would collide with '
                    'the ablated system prompt under test — remove it '
                    '(the prompt file IS the system prompt)')
        if m["role"] not in ("user", "assistant"):
            return f'messages[{i}] has unknown role {m["role"]!r}'
        if not isinstance(m["content"], str):
            return (f'messages[{i}] content must be a string — tool calls '
                    f'and structured content are not replayable yet')
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return (f'messages must alternate user/assistant starting '
                    f'with "user" (messages[{i}] is {m["role"]!r})')
    if msgs[-1]["role"] != "user":
        return ('the trace ends on an assistant turn — there is nothing '
                'to regenerate. Drop the final assistant message; replay '
                'regenerates it under each ablated prompt')
    return None
