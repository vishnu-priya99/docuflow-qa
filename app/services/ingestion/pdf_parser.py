"""PDF parsing: heading-aware unit extraction per page, with table-aware
extraction via pdfplumber.

Heading detection is a deterministic heuristic (short, title/upper-case
line, no trailing sentence punctuation, optionally numbered like "1.0" or
"2.1") - never LLM-generated. It scans every line of every page (not just
the first), since real documents commonly carry multiple sections per
page - and tracks the current section as a running value across the whole
document, so a page with no heading of its own still inherits whatever
section was last seen. When no heading is ever detected, section stays
None for every chunk, per the "do not invent a section" rule.

Tables are detected structurally (via pdfplumber's line/rectangle-based
table detection, not just text position), so cell-to-column mapping is
correct even for complex tables - unlike plain text extraction, which only
reconstructs a table correctly by coincidence. Each detected table becomes
its own ParsedUnit carrying its header row (see units.ParsedUnit.
table_header) - the chunker repeats that header on every chunk if the
table is long enough to be split, so no chunk of table data ever loses
the column labels that give the numbers meaning.

Running headers/footers (e.g. a document-number line repeated at the top
of every page, or a page-number/disclaimer line at the bottom) are
detected and stripped before chunking - otherwise that boilerplate leaks
into body text on every page and pollutes every chunk on that page, not
just one. Detection is generic (works for any document, not just this
project's own samples): the first and last non-empty line of every page
are compared after normalizing away digits (so "Page 1 of 3" and
"Page 2 of 3" are recognized as the same repeating pattern); a pattern
that recurs across more than one page is treated as running header/footer
and skipped everywhere it appears, top or bottom respectively. A pattern
appearing only once is left alone, since that's just an ordinary line
that happens to open or close a page.

A text run sitting immediately above a table can be cut short purely by
the table's bounding box (e.g. a document title with no paragraph text
between it and a title-block table right below), not by any real content
boundary. A fragment shorter than _MIN_STANDALONE_LEAD_IN_CHARS is folded
into the table's own chunk instead of being emitted as its own near-empty
one - too little text for retrieval or reranking to judge on its own
merits, and prone to scoring high on bare keyword overlap alone (e.g. a
title chunk outranking the paragraph that actually answers the
question). A real paragraph-length run above the threshold still flushes
as its own unit, unaffected.
"""
from __future__ import annotations

import io
import re

import pdfplumber

from app.services.ingestion.table_rendering import cell_text
from app.services.ingestion.units import ParsedUnit

# Allows numbered headings ("1.0 COMPLAINT SUMMARY", "2.1 Root Cause") via
# the embedded ".". Deliberately excludes ":" and "|" so running headers/
# footers like "Document No: X | Rev: A" - which are real body text on
# every page, not headings - don't get mistaken for one.
_HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,.&/'\-]{2,79}$")

_DIGITS_RE = re.compile(r"\d+")

# A text run immediately above a table can be cut short purely by the
# table's bounding box (e.g. a document title sitting right above a
# title-block table, with no paragraph text between them) rather than by
# any real content boundary. A fragment that short carries too little
# text for retrieval or reranking to judge on its own merits - it gets
# folded into the table's chunk instead of becoming its own near-empty
# one. A real paragraph-length run above this threshold is unaffected.
_MIN_STANDALONE_LEAD_IN_CHARS = 80


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    if line.isupper() or line.istitle():
        return bool(_HEADING_RE.match(line))
    return False


def _normalize_for_repeat_detection(line: str) -> str:
    """Collapses digit runs so page-number differences ("Page 1 of 3" vs
    "Page 2 of 3") don't prevent an otherwise-identical running header/
    footer line from being recognized as the same repeating pattern."""
    return _DIGITS_RE.sub("#", line.strip())


def _detect_running_lines(pdf) -> tuple[set[str], set[str]]:
    """First pass over the whole document: collects the first and last
    non-empty line of every page, and returns the normalized patterns that
    repeat on more than one page - the running header/footer patterns to
    strip in the real extraction pass."""
    top_lines: list[str] = []
    bottom_lines: list[str] = []
    for page in pdf.pages:
        lines = [line for line in (page.extract_text() or "").splitlines() if line.strip()]
        if lines:
            top_lines.append(_normalize_for_repeat_detection(lines[0]))
            bottom_lines.append(_normalize_for_repeat_detection(lines[-1]))

    def _repeated(values: list[str]) -> set[str]:
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return {v for v, n in counts.items() if n > 1}

    return _repeated(top_lines), _repeated(bottom_lines)


def _render_table(rows: list[list[str | None]]) -> tuple[str, str]:
    """Renders a pdfplumber table (list of rows, each a list of cells) as
    plain pipe-separated text, one row per line. Returns (full_text,
    header_line) - the header is just the first rendered row, in the same
    format as every other row, so prepending it back onto a later chunk
    reads naturally rather than looking like an inserted annotation. See
    table_rendering.cell_text for why an empty cell renders as "(blank)"
    rather than nothing."""
    lines = [" | ".join(cell_text(cell) for cell in row) for row in rows]
    return "\n".join(lines), (lines[0] if lines else "")


def _page_segments(page) -> list[tuple[str, str | list[list[str | None]]]]:
    """Splits one page into an ordered sequence of ("text", str) and
    ("table", rows) segments, in top-to-bottom reading order - so a table
    in the middle of a page's text flow stays in the middle, not lumped at
    the start or end."""
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    if not tables:
        return [("text", page.extract_text() or "")]

    segments: list[tuple[str, str | list[list[str | None]]]] = []
    cursor_top = 0.0
    for table in tables:
        _x0, top, _x1, bottom = table.bbox
        if top > cursor_top:
            above_text = page.crop((0, cursor_top, page.width, top)).extract_text() or ""
            if above_text.strip():
                segments.append(("text", above_text))
        segments.append(("table", table.extract()))
        cursor_top = bottom
    if cursor_top < page.height:
        below_text = page.crop((0, cursor_top, page.width, page.height)).extract_text() or ""
        if below_text.strip():
            segments.append(("text", below_text))
    return segments


def parse_pdf(content: bytes) -> list[ParsedUnit]:
    units: list[ParsedUnit] = []
    current_section: str | None = None

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        header_patterns, footer_patterns = _detect_running_lines(pdf)
        running_patterns = header_patterns | footer_patterns

        for page_index, page in enumerate(pdf.pages, start=1):
            segment_lines: list[str] = []

            def flush() -> None:
                segment_text = "\n".join(segment_lines).strip()
                if segment_text:
                    units.append(
                        ParsedUnit(
                            text=segment_text,
                            section=current_section,
                            page_start=page_index,
                            page_end=page_index,
                        )
                    )
                segment_lines.clear()

            for kind, segment_content in _page_segments(page):
                if kind == "table":
                    leading_text = "\n".join(segment_lines).strip()
                    is_short_lead_in = 0 < len(leading_text) < _MIN_STANDALONE_LEAD_IN_CHARS
                    if is_short_lead_in:
                        segment_lines.clear()
                    else:
                        flush()
                    rendered, header_line = _render_table(segment_content)
                    if rendered.strip():
                        if is_short_lead_in:
                            rendered = f"{leading_text}\n{rendered}"
                        units.append(
                            ParsedUnit(
                                text=rendered,
                                section=current_section,
                                page_start=page_index,
                                page_end=page_index,
                                table_header=header_line,
                            )
                        )
                    continue

                for line in segment_content.splitlines():
                    if _normalize_for_repeat_detection(line) in running_patterns:
                        continue
                    if _looks_like_heading(line):
                        flush()
                        current_section = line.strip()
                        continue
                    segment_lines.append(line)
            flush()

    return units
