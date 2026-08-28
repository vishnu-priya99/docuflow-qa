"""TXT parsing: a single unit carrying the whole file, with base_char_offset
set so the chunker can compute absolute char/line positions per chunk."""
from __future__ import annotations

from app.services.ingestion.units import ParsedUnit


def parse_txt(content: bytes) -> list[ParsedUnit]:
    text = content.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    return [ParsedUnit(text=text, base_char_offset=0)]
