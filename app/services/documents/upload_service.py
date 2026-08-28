"""End-to-end file upload handling: validate -> store original -> ingest."""
from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.file import SUPPORTED_DOCUMENT_TYPES, SUPPORTED_FILE_TYPES, SUPPORTED_STRUCTURED_TYPES, FileRecord
from app.services.documents.ingestion_service import UnsupportedFileTypeError, ingest_document
from app.services.embeddings.base import EmbeddingProvider
from app.services.excel.excel_service import ingest_excel
from app.services.storage.base import StorageBackend
from app.vector.qdrant_client import QdrantService

logger = get_logger(__name__)


class NoExtractableContentError(RuntimeError):
    """Raised when a file parsed successfully but yielded nothing to index -
    e.g. a scanned/image-only PDF with no text layer, or an empty
    spreadsheet. Treated as an ingestion failure rather than a silent
    "ready" file with zero retrievable content."""


def detect_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    if ext not in SUPPORTED_FILE_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}'. Supported: {sorted(SUPPORTED_FILE_TYPES)}"
        )
    return ext


async def process_upload(
    *,
    db: AsyncSession,
    qdrant: QdrantService,
    embedder: EmbeddingProvider,
    storage: StorageBackend,
    session_id: str,
    user_id: str,
    filename: str,
    content: bytes,
) -> FileRecord:
    file_type = detect_file_type(filename)

    file_record = FileRecord(
        session_id=session_id,
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        storage_path="",
        size_bytes=len(content),
        status="processing",
    )
    db.add(file_record)
    await db.flush()

    storage_path = await storage.save(
        session_id=session_id, file_id=file_record.file_id, filename=filename, content=content
    )
    file_record.storage_path = storage_path

    try:
        if file_type in SUPPORTED_DOCUMENT_TYPES:
            chunk_count = await ingest_document(
                qdrant=qdrant, embedder=embedder, file_record=file_record, content=content
            )
            if chunk_count == 0:
                raise NoExtractableContentError(
                    "No extractable text was found in this file. If it's a scanned or "
                    "image-based document, this pipeline has no OCR step and can't read it."
                )
        elif file_type in SUPPORTED_STRUCTURED_TYPES:
            _workbook, total_rows = await ingest_excel(
                db=db, qdrant=qdrant, embedder=embedder, file_record=file_record, content=content
            )
            if total_rows == 0:
                raise NoExtractableContentError("No data rows were found in this file.")
        file_record.status = "ready"
    except Exception as exc:  # noqa: BLE001 - surface any ingestion failure on the file record
        logger.exception("Ingestion failed for file %s (%s)", file_record.file_id, filename)
        file_record.status = "failed"
        file_record.error_message = str(exc)[:2000]

    await db.flush()
    return file_record
