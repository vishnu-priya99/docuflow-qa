"""Shared intermediate representation produced by every document parser.

A ``ParsedUnit`` is a contiguous span of source text plus the location
metadata the parser could deterministically establish for it (page, slide,
paragraph, section/heading...). The chunker (chunker.py) may split a single
unit into several chunks; every resulting chunk inherits the unit's
location/section metadata unchanged, per the metadata rule in the spec.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedUnit:
    text: str
    section: str | None = None

    # PDF
    page_start: int | None = None
    page_end: int | None = None

    # DOCX
    paragraph_start: int | None = None
    paragraph_end: int | None = None

    # PPTX
    slide_number: int | None = None
    slide_title: str | None = None

    # TXT - offset of this unit's text within the original file, so the
    # chunker can compute absolute char/line positions.
    base_char_offset: int | None = None

    # PDF tables (see pdf_parser.py): the table's header row, rendered the
    # same way as the data rows. When set, this unit's first line IS this
    # header - the chunker repeats it on every chunk after the first if the
    # table is long enough to be split, so no chunk of table data ever
    # loses the column labels that give the numbers meaning.
    table_header: str | None = None


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    slide_number: int | None = None
    slide_title: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
