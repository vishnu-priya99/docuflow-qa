from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_id, get_db, get_qdrant, get_storage
from app.models.message import Message
from app.schemas.chat import MessageListOut, MessageOut
from app.schemas.session import SessionCreate, SessionListOut, SessionOut
from app.services import session_service
from app.services.storage.base import StorageBackend
from app.vector.qdrant_client import QdrantService

router = APIRouter()


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    from app.services.auth_service import get_or_create_user

    await get_or_create_user(db, user_id)
    session = await session_service.create_session(db, user_id=user_id, title=payload.title)
    await db.commit()
    return SessionOut.model_validate(session)


@router.get("", response_model=SessionListOut)
async def list_sessions(
    user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
) -> SessionListOut:
    sessions = await session_service.list_sessions(db, user_id=user_id)
    return SessionListOut(sessions=[SessionOut.model_validate(s) for s in sessions])


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    session = await session_service.get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return SessionOut.model_validate(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
    storage: StorageBackend = Depends(get_storage),
) -> None:
    deleted = await session_service.delete_session(
        db, user_id=user_id, session_id=session_id, qdrant=qdrant, storage=storage
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    await db.commit()


@router.get("/{session_id}/messages", response_model=MessageListOut)
async def list_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> MessageListOut:
    session = await session_service.get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    )
    messages = list(result.scalars().all())
    return MessageListOut(messages=[MessageOut.model_validate(m) for m in messages])
