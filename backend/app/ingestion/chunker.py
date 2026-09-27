"""Split page text into page- and section-aware chunks.

Chunks never cross a page boundary, so every chunk maps to exactly one source
page. Headings are detected with simple heuristics; when none is found the
chunk's section is `None` (it is not invented).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Chunk, PageText

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 150

_NUMBERED_HEADING = re.compile(
    r"^(?:(?:section|clause|part|article)\s+)?\d+(?:\.\d+)*[.)]?\s+[A-Za-z]", re.IGNORECASE
)
_KEYWORD_HEADING = re.compile(
    r"^(?:section|clause|part|article|schedule|annexure)\s+[\dA-Z]+\b", re.IGNORECASE
)


def is_heading(line: str) -> bool:
    """Heuristic heading detector for policy wordings."""
    line = line.strip()
    if not line or len(line) > 80:
        return False
    words = line.split()
    if _KEYWORD_HEADING.match(line) and len(words) <= 10:
        return True
    if _NUMBERED_HEADING.match(line) and len(words) <= 8 and not line.endswith((".", ",", ";")):
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters) and len(words) <= 10


@dataclass
class _Segment:
    section: str | None
    start: int
    end: int


def _segments(text: str, current_section: str | None) -> tuple[list[_Segment], str | None]:
    """Split one page's text at heading lines. Returns segments and the last section seen."""
    segments: list[_Segment] = []
    seg_start = 0
    offset = 0
    section = current_section
    for line in text.splitlines(keepends=True):
        if is_heading(line) and offset > seg_start:
            segments.append(_Segment(section, seg_start, offset))
            seg_start = offset
        if is_heading(line):
            section = line.strip()
        offset += len(line)
    segments.append(_Segment(section, seg_start, len(text)))
    # A segment that started with a heading belongs to that heading's section.
    for seg in segments:
        first_line = text[seg.start : seg.end].split("\n", 1)[0]
        if is_heading(first_line):
            seg.section = first_line.strip()
    return segments, section


def _windows(text: str, start: int, end: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Character windows over text[start:end], breaking on whitespace where possible."""
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        stop = min(pos + max_chars, end)
        if stop < end:
            cut = text.rfind(" ", pos + max_chars // 2, stop)
            cut_nl = text.rfind("\n", pos + max_chars // 2, stop)
            cut = max(cut, cut_nl)
            if cut > pos:
                stop = cut
        spans.append((pos, stop))
        if stop >= end:
            break
        next_pos = max(stop - overlap, pos + 1)
        # Start the next window at a word boundary.
        while next_pos < stop and not text[next_pos - 1].isspace():
            next_pos += 1
        pos = next_pos
    return spans


def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    doc_name: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        if page.empty_text:
            continue
        segments, section = _segments(page.text, section)
        for seg in segments:
            for w_start, w_end in _windows(page.text, seg.start, seg.end, max_chars, overlap):
                raw = page.text[w_start:w_end]
                stripped = raw.strip()
                if not stripped:
                    continue
                lead = len(raw) - len(raw.lstrip())
                c_start = w_start + lead
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:p{page.page}:c{len(chunks)}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page.page,
                        section=seg.section,
                        text=stripped,
                        char_start=c_start,
                        char_end=c_start + len(stripped),
                    )
                )
    return chunks
