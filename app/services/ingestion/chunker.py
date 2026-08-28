"""Deterministic, programmatic chunker.

Splits each ParsedUnit's text into overlapping chunks via LangChain's
RecursiveCharacterTextSplitter (paragraph -> line -> word -> character,
falling back only when it has to - avoids cutting mid-sentence/mid-word
the way a fixed-offset cut would). Every chunk produced from a given unit
inherits that unit's location/section metadata verbatim, plus its own
chunk_id/chunk_index (assigned by the caller) and, when the unit carries
a base_char_offset (TXT), its own absolute char_start/char_end and
line_start/line_end.

No LLM is involved anywhere in this module.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.ingestion.units import Chunk, ParsedUnit


def _split_text(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Return (chunk_text, start_offset, end_offset) tuples covering ``text``."""
    if not text:
        return []
    if len(text) <= size:
        return [(text, 0, len(text))]

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    docs = splitter.create_documents([text])
    return [(d.page_content, d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs]


def _line_number_at(text: str, offset: int) -> int:
    """1-indexed line number containing ``offset`` within ``text``."""
    return text.count("\n", 0, offset) + 1


def build_chunks(units: list[ParsedUnit]) -> list[Chunk]:
    """Chunk every unit using module-level default sizes.

    Kept as a thin wrapper so callers needn't import settings directly;
    prefer ``build_chunks_with_sizes`` when you have explicit config.
    """
    from app.core.config import get_settings

    settings = get_settings()
    return build_chunks_with_sizes(
        units, chunk_size=settings.chunk_size_chars, overlap=settings.chunk_overlap_chars
    )


def build_chunks_with_sizes(units: list[ParsedUnit], *, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for unit in units:
        text = (unit.text or "").strip()
        if not text:
            continue

        # Preserve absolute offsets for TXT files where base_char_offset is set.
        source_text = unit.text if unit.base_char_offset is not None else text
        spans = _split_text(source_text, chunk_size, overlap)

        for span_index, (chunk_text, local_start, local_end) in enumerate(spans):
            if not chunk_text.strip():
                continue
            # A table long enough to be split: every piece after the first
            # gets its header row prepended, so it stays interpretable on
            # its own (see units.ParsedUnit.table_header).
            if unit.table_header and span_index > 0:
                chunk_text = f"{unit.table_header}\n{chunk_text}"
            char_start = char_end = line_start = line_end = None
            if unit.base_char_offset is not None:
                char_start = unit.base_char_offset + local_start
                char_end = unit.base_char_offset + local_end
                line_start = _line_number_at(source_text, local_start)
                line_end = _line_number_at(source_text, max(local_end - 1, 0))

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    chunk_index=index,
                    section=unit.section,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    paragraph_start=unit.paragraph_start,
                    paragraph_end=unit.paragraph_end,
                    slide_number=unit.slide_number,
                    slide_title=unit.slide_title,
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            index += 1
    return chunks
