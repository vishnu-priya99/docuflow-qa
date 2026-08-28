"""Async SQLAlchemy engine/session management.

Two engines are maintained:
  * ``engine`` / ``get_db`` - full read/write access, used by the
    application for all normal CRUD and structured-data ingestion.
  * ``readonly_engine`` / ``get_readonly_db`` - used ONLY to execute
    LLM-generated SQL. In production this should point at a dedicated
    read-only PostgreSQL role (see scripts/init_readonly_role.sql). If a
    separate URL isn't configured it falls back to the primary engine, and
    safety then relies entirely on the SQL validator (see services/sql).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine: AsyncEngine = create_async_engine(settings.database_url, pool_pre_ping=True, future=True)
readonly_engine: AsyncEngine = create_async_engine(
    settings.effective_readonly_database_url, pool_pre_ping=True, future=True
)

SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
ReadonlySessionLocal = async_sessionmaker(bind=readonly_engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def get_readonly_db() -> AsyncGenerator[AsyncSession, None]:
    async with ReadonlySessionLocal() as session:
        yield session
