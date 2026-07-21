"""Segment a system prompt into an ablatable tree.

Model: the document is an ordered list of *blocks*. A block is either a
section header line or a leaf (one rule / one prose paragraph). Sections
own contiguous runs of blocks. Reconstruction is exact by construction:
joining all block texts reproduces the original file byte-for-byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


HEADER_RE = re.compile(r"^(#{1,4})\s")
# A line that *starts* a new rule-like leaf even without a blank line above it.
RULE_START_RE = re.compile(
    r"^(\s*)("
    r"\d+[.)]\s"                      # numbered rules
    r"|[-*]\s"                        # bullets
    r"|CRITICAL\b|IMPORTANT\b|NOTE\b|WARNING\b|REMEMBER\b"
    r"|If\b|When\b|Never\b|Always\b|Do not\b|Don't\b|Use\b|Avoid\b"
    r"|You are\b|You must\b|You MUST\b|Respond\b|Keep\b|Think\b|Take\b"
    r"|<!--"
    r")"
)
CONTINUATION_RE = re.compile(r"^\s{2,}\S")  # indented wrap of previous rule


@dataclass
class Segment:
    id: str
    kind: str                 # "section" | "leaf" | "header"
    text: str                 # exact text incl. newlines
    section_id: str           # owning section id ("" for the section itself)
    title: str = ""
    children: list["Segment"] = field(default_factory=list)

    @property
    def display(self) -> str:
        t = " ".join(self.text.split())
        return t[:88] + ("…" if len(t) > 88 else "")


@dataclass
class Doc:
    original: str
    blocks: list[Segment]              # ordered: headers + leaves
    sections: list[Segment]            # section nodes (children = leaves)

    def leaves(self) -> list[Segment]:
        return [b for b in self.blocks if b.kind == "leaf"]

    def rebuild(self, removed: set[str] | None = None,
                replaced: dict[str, str] | None = None) -> str:
        removed = removed or set()
        replaced = replaced or {}
        out: list[str] = []
        for b in self.blocks:
            if b.id in removed or b.section_id in removed:
                continue
            out.append(replaced.get(b.id, b.text))
        return "".join(out)


def _split_lines_keepends(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def parse(text: str) -> Doc:
    lines = _split_lines_keepends(text)
    blocks: list[Segment] = []
    sections: list[Segment] = []

    sec_idx = 0
    cur_section = Segment(id=f"S{sec_idx}", kind="section", text="",
                          section_id="", title="(preamble)")
    sections.append(cur_section)

    leaf_lines: list[str] = []
    leaf_counter = 0

    def flush_leaf():
        nonlocal leaf_lines, leaf_counter
        if not leaf_lines:
            return
        txt = "".join(leaf_lines)
        # pure-whitespace runs attach to the previous leaf so we never
        # ablate "a blank line" as if it were a rule
        if txt.strip() == "":
            if cur_section.children:
                # same object lives in blocks, so this updates both views
                cur_section.children[-1].text += txt
            elif blocks and blocks[-1].kind == "header":
                blocks[-1].text += txt
            else:  # leading whitespace before anything: glue to a stub
                seg = Segment(id=f"{cur_section.id}.pad", kind="leaf",
                              text=txt, section_id=cur_section.id)
                blocks.append(seg)
                cur_section.children.append(seg)
            leaf_lines = []
            return
        leaf_counter += 1
        seg = Segment(id=f"{cur_section.id}.L{leaf_counter}", kind="leaf",
                      text=txt, section_id=cur_section.id)
        blocks.append(seg)
        cur_section.children.append(seg)
        leaf_lines = []

    for line in lines:
        if HEADER_RE.match(line):
            flush_leaf()
            sec_idx += 1
            leaf_counter = 0
            cur_section = Segment(id=f"S{sec_idx}", kind="section", text="",
                                  section_id="", title=line.strip().lstrip("#").strip())
            sections.append(cur_section)
            hdr = Segment(id=f"S{sec_idx}.hdr", kind="header", text=line,
                          section_id=cur_section.id, title=cur_section.title)
            blocks.append(hdr)
            continue

        stripped = line.strip()
        if stripped == "":
            leaf_lines.append(line)
            flush_leaf()
            continue

        starts_new = RULE_START_RE.match(line) and not CONTINUATION_RE.match(line)
        if starts_new and leaf_lines and "".join(leaf_lines).strip() != "":
            flush_leaf()
        leaf_lines.append(line)

    flush_leaf()

    doc = Doc(original=text, blocks=blocks,
              sections=[s for s in sections if s.children])
    assert doc.rebuild() == text, "segmenter failed exact reconstruction"
    return doc
