"""Orchestrates: file -> parse -> chunk -> embed -> Qdrant.

Chunk content and its location metadata are stored only in Qdrant - the
retrieval path never reads them from anywhere else, so there's no
separate Postgres copy. document_id is set equal to file_id (one
uploaded file = one logical document; no versioning is supported).
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.db.base import new_uuid
from app.models.file import FileRecord
from app.services.embeddings.base import EmbeddingProvider
from app.services.ingestion.chunker import build_chunks
from app.services.ingestion.docx_parser import parse_docx
from app.services.ingestion.pdf_parser import parse_pdf
from app.services.ingestion.pptx_parser import parse_pptx
from app.services.ingestion.txt_parser import parse_txt
from app.services.ingestion.units import ParsedUnit
from app.vector.qdrant_client import ChunkPayload, QdrantService

logger = get_logger(__name__)

_PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "pptx": parse_pptx,
    "txt": parse_txt,
}


class UnsupportedFileTypeError(ValueError):
    pass


def parse_document(file_type: str, content: bytes) -> list[ParsedUnit]:
    parser = _PARSERS.get(file_type)
    if parser is None:
        raise UnsupportedFileTypeError(f"No document parser for file type: {file_type}")
    return parser(content)


async def ingest_document(
    *,
    qdrant: QdrantService,
    embedder: EmbeddingProvider,
    file_record: FileRecord,
    content: bytes,
) -> int:
    """Runs the full pipeline for one PDF/DOCX/PPTX/TXT file.

    Returns the number of chunks created. Raises on unrecoverable errors;
    callers are expected to mark the file record failed on exception.
    """
    units = parse_document(file_record.file_type, content)
    chunks = build_chunks(units)

    if not chunks:
        logger.warning("No extractable text in file %s (%s)", file_record.file_id, file_record.filename)
        return 0

    texts = [c.text for c in chunks]
    vectors = await embedder.embed(texts)

    document_id = file_record.file_id
    points: list[tuple[str, list[float], dict]] = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        point_id = new_uuid()
        payload = ChunkPayload(
            user_id=file_record.user_id,
            session_id=file_record.session_id,
            file_id=file_record.file_id,
            filename=file_record.filename,
            file_type=file_record.file_type,
            document_id=document_id,
            chunk_id=point_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            char_count=len(chunk.text),
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            paragraph_start=chunk.paragraph_start,
            paragraph_end=chunk.paragraph_end,
            slide_number=chunk.slide_number,
            slide_title=chunk.slide_title,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        points.append((point_id, vector, payload.to_payload()))

    await qdrant.upsert(points)

    # Chunk content itself isn't logged here - it's stored in Qdrant and
    # browsable in its web dashboard. [RETRIEVAL]/[RERANK] (qdrant_client.py,
    # cross_encoder_rerank.py/llm_rerank.py) still log full chunk text for
    # whichever ones get pulled into a question's evidence.
    logger.info(
        "[UPLOAD] %s (%s) file_id=%s -> %d chunk(s) created",
        file_record.filename, file_record.file_type, file_record.file_id, len(chunks),
    )
    return len(chunks)
