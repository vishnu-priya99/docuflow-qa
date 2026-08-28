"""On-demand snapshot of what's actually stored right now - Postgres tables
(both the app's fixed schema and dynamically-created per-sheet xlsx_*
tables) and the Qdrant collection.

Not part of the user-facing product surface (no auth, not linked from the
frontend) - a verification tool for development/demo use. Kept as an
in-process endpoint (rather than a standalone script, the way
scripts/inspect_db.py is for Postgres) mainly for convenience - checking
via a browser/curl call needs nothing extra running beyond the app itself.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_qdrant
from app.core.logging import get_logger
from app.db.base import Base
from app.db.session import get_db
from app.vector.qdrant_client import QdrantService

# Import models so Base.metadata is populated with the app's own tables.
from app import models  # noqa: F401

logger = get_logger(__name__)
router = APIRouter()


@router.get("/state")
async def debug_state(db: AsyncSession = Depends(get_db), qdrant: QdrantService = Depends(get_qdrant)) -> dict:
    static_tables = sorted(Base.metadata.tables.keys())

    result = await db.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    )
    all_tables = [row[0] for row in result.fetchall()]
    dynamic_tables = [t for t in all_tables if t.startswith("xlsx_")]

    async def _row_count(table_name: str) -> int:
        # table_name always comes from information_schema (real existing
        # identifiers this app created via sanitize.py, [a-z0-9_] only) -
        # never user input - so safe to interpolate into the identifier
        # position, which can't be parameter-bound in SQL anyway.
        r = await db.execute(text(f'SELECT count(*) FROM "{table_name}"'))
        return r.scalar_one()

    postgres_snapshot = {
        "app_schema_tables": {t: await _row_count(t) for t in static_tables if t in all_tables},
        "dynamic_sheet_tables": {t: await _row_count(t) for t in dynamic_tables},
    }
    qdrant_snapshot = await qdrant.get_stats()

    logger.info("=" * 88)
    logger.info("[DEBUG STATE] Postgres app-schema tables: %s", postgres_snapshot["app_schema_tables"])
    logger.info("[DEBUG STATE] Postgres dynamic xlsx_* tables: %s", postgres_snapshot["dynamic_sheet_tables"])
    logger.info("[DEBUG STATE] Qdrant: %s", qdrant_snapshot)
    logger.info("=" * 88)

    return {"postgres": postgres_snapshot, "qdrant": qdrant_snapshot}
