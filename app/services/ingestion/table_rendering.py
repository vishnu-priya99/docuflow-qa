"""Shared table-cell rendering for pdf_parser/docx_parser/pptx_parser.

A genuinely empty table cell (an unfilled form field - a blank signature,
date, or sign-off box) must render as an explicit marker, not as nothing.
Rendered as nothing, it's textually indistinguishable from a transcription
gap, so nothing signals to a downstream LLM that the field was actually
checked and found empty, not merely omitted - or, worse, some renderers
drop an empty cell entirely, losing column alignment along with it.
Verified: without this, an unfilled approval-date cell got read as
ambiguous rather than "no value", and a nearby date elsewhere in context
got fabricated in as if it were the answer. This applies to any table with
unfilled fields in any supported document type, not just one case.
"""
from __future__ import annotations

_BLANK_CELL = "(blank)"


def cell_text(cell: str | None) -> str:
    text = (cell or "").strip()
    return text if text else _BLANK_CELL
