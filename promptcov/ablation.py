"""Build prompt variants: delete a segment, or negate a leaf rule.

Negation distinguishes REDUNDANT from INERT: if deleting a rule changes
nothing but *inverting* it changes behavior, the rule is covered by a
sibling — a different finding than "decorative".
"""

from __future__ import annotations

import re

_SWAPS = [
    (r"\bALWAYS\b", "NEVER"), (r"\bNEVER\b", "ALWAYS"),
    (r"\bAlways\b", "Never"), (r"\bNever\b", "Always"),
    (r"\balways\b", "never"), (r"\bnever\b", "always"),
    (r"\bMUST NOT\b", "MUST"), (r"\bmust not\b", "must"),
    (r"\bMUST\b", "MUST NOT"), (r"\bmust\b", "must not"),
    (r"\bDo not\b", "Do"), (r"\bDO NOT\b", "DO"),
    (r"\bdo not\b", "do"), (r"\bdon't\b", "do"),
]


def negate_heuristic(text: str) -> str | None:
    """Cheap textual inversion. Returns None when no safe inversion exists."""
    for pat, rep in _SWAPS:
        if re.search(pat, text):
            return re.sub(pat, rep, text, count=1)
    # generic imperative → prohibition
    m = re.match(r"^(\s*(?:\d+[.)]\s|[-*]\s)?)([A-Z][a-z]+\b)(.*)$",
                 text, flags=re.S)
    if m and m.group(2) not in ("If", "When", "The", "This", "You"):
        return f"{m.group(1)}Do not {m.group(2).lower()}{m.group(3)}"
    return None


def negate(text: str, provider=None) -> str | None:
    """Prefer an LLM inversion when the provider offers one."""
    if provider is not None and hasattr(provider, "negate_rule"):
        try:
            inv = provider.negate_rule(text)
            if inv and inv.strip() and inv.strip() != text.strip():
                if not inv.endswith("\n") and text.endswith("\n"):
                    inv += "\n"
                return inv
        except Exception:
            pass
    return negate_heuristic(text)
