from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_id, get_db, get_embedder, get_qdrant, get_storage
from app.core.config import get_settings
from app.models.file import FileRecord
from app.schemas.file import FileListOut, FileOut
from app.services import session_service
from app.services.documents.ingestion_service import UnsupportedFileTypeError
from app.services.documents.upload_service import process_upload
from app.services.embeddings.base import EmbeddingProvider
from app.services.storage.base import StorageBackend
from app.vector.qdrant_client import QdrantService

router = APIRouter()


@router.post("/{session_id}/files", response_model=FileOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    session_id: str,
    file: UploadFile,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
    embedder: EmbeddingProvider = Depends(get_embedder),
    storage: StorageBackend = Depends(get_storage),
) -> FileOut:
    session = await session_service.get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    settings = get_settings()
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit.",
        )

    try:
        file_record = await process_upload(
            db=db,
            qdrant=qdrant,
            embedder=embedder,
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            filename=file.filename or "upload",
            content=content,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    await db.commit()
    if file_record.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Ingestion failed: {file_record.error_message}",
        )
    return FileOut.model_validate(file_record)


@router.get("/{session_id}/files", response_model=FileListOut)
async def list_files(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> FileListOut:
    session = await session_service.get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    result = await db.execute(
        select(FileRecord).where(FileRecord.session_id == session_id).order_by(FileRecord.created_at.asc())
    )
    files = list(result.scalars().all())
    return FileListOut(files=[FileOut.model_validate(f) for f in files])
