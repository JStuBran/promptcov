"""Output divergence: 0.0 = identical, 1.0 = unrelated.

Blend of character-level sequence similarity and word-set Jaccard.
Cheap first-pass signal; an LLM judge can be layered on top (--judge)
for behavioral rather than surface deltas.
"""

from __future__ import annotations

import difflib
import re

_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def divergence(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        jac = 1.0
    else:
        jac = len(wa & wb) / max(1, len(wa | wb))
    return round(1.0 - (0.6 * ratio + 0.4 * jac), 6)
