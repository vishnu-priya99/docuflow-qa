"""Session lifecycle - creation and full cascading deletion.

Deleting a session removes, in order: its dynamically-created structured
data tables, its Qdrant vectors (both document chunks and Excel sheet
summaries), its original files from storage, then the session row itself
(which cascades to messages/files/excel_* rows via FK ON DELETE CASCADE -
see models).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.excel import ExcelSheet
from app.models.session import ChatSession
from app.services.excel.dynamic_tables import drop_session_tables
from app.services.storage.base import StorageBackend
from app.vector.qdrant_client import QdrantService

logger = get_logger(__name__)


async def create_session(db: AsyncSession, *, user_id: str, title: str = "New chat") -> ChatSession:
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.flush()
    return session


async def list_sessions(db: AsyncSession, *, user_id: str) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc())
    )
    return list(result.scalars().all())


async def get_session(db: AsyncSession, *, user_id: str, session_id: str) -> ChatSession | None:
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id, ChatSession.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def delete_session(
    db: AsyncSession,
    *,
    user_id: str,
    session_id: str,
    qdrant: QdrantService,
    storage: StorageBackend,
) -> bool:
    session = await get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        return False

    table_names_result = await db.execute(
        select(ExcelSheet.table_name).where(ExcelSheet.session_id == session_id)
    )
    table_names = [row[0] for row in table_names_result.all()]
    if table_names:
        await drop_session_tables(db, table_names)

    await qdrant.delete_session(user_id=user_id, session_id=session_id)
    await storage.delete_session(session_id)

    await db.delete(session)
    await db.flush()
    logger.info("Deleted session %s (user=%s) and all owned data", session_id, user_id)
    return True
