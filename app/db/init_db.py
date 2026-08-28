from __future__ import annotations

from sqlalchemy import inspect

from app.core.logging import get_logger
from app.db.base import Base
from app.db.session import engine

# Import models so they're registered on Base.metadata before create_all.
from app import models  # noqa: F401

logger = get_logger(__name__)


async def init_models() -> None:
    """Create all tables that don't exist yet.

    A lightweight substitute for a migration tool, appropriate for this
    project's scope (see "Do not introduce unnecessary infrastructure").
    """
    async with engine.begin() as conn:
        existing_before = set(await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names()))
        await conn.run_sync(Base.metadata.create_all)

    expected = set(Base.metadata.tables.keys())
    pre_existing = sorted(expected & existing_before)
    newly_created = sorted(expected - existing_before)
    logger.info(
        "Database schema ready | %d table(s) already existed %s | %d newly created %s",
        len(pre_existing), pre_existing, len(newly_created), newly_created,
    )
