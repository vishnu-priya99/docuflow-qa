"""Shared test fixtures.

Runs the whole app in-process against a real PostgreSQL database (reads
DATABASE_URL/DATABASE_URL_READONLY from .env, falling back to the same
local default as the app itself - see scripts/setup_postgres_windows.ps1
or docker-compose.yml if you don't have Postgres running) and a real
Qdrant server (reads QDRANT_URL from .env too, falling back to
http://localhost:6333 - see .qdrant-portable/qdrant.exe; no embedded/
:memory: mode - see app/vector/qdrant_client.py), plus the deterministic
MockLLMProvider and the offline hashing embedding provider.

IMPORTANT: Qdrant must be running before `pytest` will work.

Every test gets its own user_id/session_id so tests never interfere with
each other even though the backend is shared for the whole session
(mirrors how the real server is a long-lived process). A session-scoped
teardown wipes everything the suite created (by user_id prefix) from
Postgres, and drops the entire test-only Qdrant collection (safe since
QDRANT_COLLECTION is a dedicated test name, never the real one) so
repeated runs don't leave debris.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

# Environment must be set before anything under `app` is imported, since
# app.core.config.Settings is read at several modules' import time.
from dotenv import dotenv_values

_env_file = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
for _key in ("DATABASE_URL", "DATABASE_URL_READONLY", "QDRANT_URL"):
    if _env_file.get(_key):
        os.environ[_key] = _env_file[_key]
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

_TEST_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / f"test_files_{uuid.uuid4().hex}"

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_DIM"] = "384"
# Must override, not just rely on config.py's default - a real .env (e.g.
# RERANK_PROVIDER=cross_encoder for local dev) would otherwise leak into
# every test run and download an ML model over the network mid-suite.
os.environ["RERANK_PROVIDER"] = "llm"
# Dedicated collection name (not the real "document_chunks") - keeps test
# data separate from production data on the same Qdrant server.
os.environ["QDRANT_COLLECTION"] = "test_document_chunks"
os.environ["STORAGE_LOCAL_PATH"] = str(_TEST_STORAGE_PATH)
os.environ["SQL_QUERY_TIMEOUT_SECONDS"] = "5"
os.environ["SQL_MAX_ROWS"] = "500"
os.environ["CHUNK_SIZE_CHARS"] = "600"
os.environ["CHUNK_OVERLAP_CHARS"] = "80"
os.environ["CORS_ORIGINS"] = "http://localhost:8000"

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.vector.qdrant_client import get_qdrant_service  # noqa: E402

# Every test in this suite generates ids via this prefix (see the user_id
# fixture below and its ad-hoc uses in a couple of test files) - the
# teardown below uses it to find and remove everything the suite created.
_TEST_USER_PREFIX = "user_"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client():
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    await _cleanup_test_data()
    await get_qdrant_service().delete_collection()
    shutil.rmtree(_TEST_STORAGE_PATH, ignore_errors=True)


async def _cleanup_test_data() -> None:
    """Remove every session (and, via cascade, everything it owns) created
    by this suite, then drop the dynamically-created structured-data
    tables those sessions owned - deleting a `sessions` row cascades to
    the ORM-tracked excel_sheets/excel_workbooks rows, but not to the raw
    `xlsx_*` table those rows describe (see services/excel/dynamic_tables.py).

    Table names are captured BEFORE deleting anything, scoped strictly to
    this suite's own test sessions (by the user_id prefix). This must
    never degrade into "drop every xlsx_* table in the database" - tests
    run against the same real Postgres instance as normal manual use (see
    README - Postgres is a hard requirement, no SQLite fallback), so a
    blanket sweep would destroy a live session's real data as a side
    effect of simply running `pytest`.
    """
    prefix_pattern = _TEST_USER_PREFIX.replace("_", r"\_") + "%"
    async with SessionLocal() as db:
        result = await db.execute(
            text(
                "SELECT es.table_name FROM excel_sheets es "
                "JOIN sessions s ON s.session_id = es.session_id "
                "WHERE s.user_id LIKE :prefix ESCAPE '\\'"
            ),
            {"prefix": prefix_pattern},
        )
        table_names = [row[0] for row in result.all()]

        await db.execute(
            text("DELETE FROM sessions WHERE user_id LIKE :prefix ESCAPE '\\'"), {"prefix": prefix_pattern}
        )
        await db.execute(
            text("DELETE FROM users WHERE user_id LIKE :prefix ESCAPE '\\'"), {"prefix": prefix_pattern}
        )
        for table_name in table_names:
            await db.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
        await db.commit()


@pytest.fixture
def user_id() -> str:
    return f"{_TEST_USER_PREFIX}{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(loop_scope="session")
async def auth_headers(client: httpx.AsyncClient, user_id: str) -> dict[str, str]:
    response = await client.post("/api/auth/login", json={"user_id": user_id})
    assert response.status_code == 200
    return {"X-User-Id": user_id}


@pytest_asyncio.fixture(loop_scope="session")
async def session_id(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> str:
    response = await client.post("/api/sessions", json={"title": "Test session"}, headers=auth_headers)
    assert response.status_code == 201
    return response.json()["session_id"]


async def upload_file(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    headers: dict[str, str],
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
) -> httpx.Response:
    files = {"file": (filename, content, content_type)}
    return await client.post(f"/api/sessions/{session_id}/files", files=files, headers=headers)


async def ask(client: httpx.AsyncClient, *, session_id: str, headers: dict[str, str], question: str) -> dict:
    response = await client.post(
        f"/api/sessions/{session_id}/chat", json={"question": question}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()
