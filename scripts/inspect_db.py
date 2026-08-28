"""Standalone viewer for the app's live PostgreSQL contents.

For Qdrant instead, use its own official web dashboard (Qdrant is a real
server here too now, not embedded/:memory: - see app/vector/qdrant_client.py) -
run .qdrant-portable/qdrant.exe, then open http://localhost:6333/dashboard.

Reads DATABASE_URL from .env via the app's own Settings, so it always
points at whatever Postgres the app itself is using - nothing to
duplicate/keep in sync by hand.

Usage (run from the project root, or anywhere - it locates the project
root itself):
    python scripts/inspect_db.py                     # overview: every table, columns, row counts
    python scripts/inspect_db.py --sessions           # one row per chat session + what it owns
    python scripts/inspect_db.py --table files        # dump rows from one table
    python scripts/inspect_db.py --table files --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# So this runs as `python scripts/inspect_db.py` from anywhere, without
# needing PYTHONPATH set or the venv activated in a particular cwd first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg  # noqa: E402

from app.core.config import get_settings  # noqa: E402

SEP = "*" * 50


def _dsn(database_url: str) -> str:
    # asyncpg wants a bare postgresql:// DSN - the app's URL carries the
    # SQLAlchemy "+asyncpg" driver suffix, which asyncpg itself rejects.
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _table_names(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )
    return [r["table_name"] for r in rows]


async def _columns(conn: asyncpg.Connection, table_name: str) -> list[tuple[str, str]]:
    rows = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position",
        table_name,
    )
    return [(r["column_name"], r["data_type"]) for r in rows]


async def _row_count(conn: asyncpg.Connection, table_name: str) -> int:
    return await conn.fetchval(f'SELECT count(*) FROM "{table_name}"')


async def print_overview(conn: asyncpg.Connection) -> None:
    tables = await _table_names(conn)
    static_tables = [t for t in tables if not t.startswith("xlsx_")]
    dynamic_tables = [t for t in tables if t.startswith("xlsx_")]

    print("=" * 88)
    print(f"POSTGRES OVERVIEW - {len(static_tables)} app table(s), {len(dynamic_tables)} dynamic sheet table(s)")
    print("=" * 88)

    for group_label, group in (("APP SCHEMA", static_tables), ("DYNAMIC XLSX/CSV SHEET TABLES", dynamic_tables)):
        if not group:
            continue
        print(f"\n--- {group_label} ---")
        for table_name in group:
            cols = await _columns(conn, table_name)
            count = await _row_count(conn, table_name)
            print()
            print(SEP)
            print()
            print(f"[TABLE] {table_name}  ({count} row(s))")
            for col_name, data_type in cols:
                print(f"    {col_name} ({data_type})")
    print()
    print(SEP)
    print("=" * 88)


async def print_sessions(conn: asyncpg.Connection) -> None:
    # Chunk counts aren't queryable here - unstructured document chunks
    # live only in Qdrant, not Postgres. Check the Qdrant web dashboard
    # for those.
    rows = await conn.fetch(
        """
        SELECT s.session_id, s.title, s.user_id, s.created_at,
               (SELECT count(*) FROM files f WHERE f.session_id = s.session_id) AS files,
               (SELECT count(*) FROM excel_sheets sh WHERE sh.session_id = s.session_id) AS sheets,
               (SELECT count(*) FROM messages m WHERE m.session_id = s.session_id) AS messages
        FROM sessions s
        ORDER BY s.created_at DESC
        """
    )
    print("=" * 88)
    print(f"SESSIONS - {len(rows)} total")
    print("=" * 88)
    for r in rows:
        print()
        print(SEP)
        print()
        print(f"[SESSION] {r['session_id']}  user={r['user_id']}  title={r['title']!r}  created={r['created_at']}")
        print(f"    files={r['files']}  sheets={r['sheets']}  messages={r['messages']}")
    print()
    print(SEP)
    print("=" * 88)


async def print_table_rows(conn: asyncpg.Connection, table_name: str, limit: int) -> None:
    tables = await _table_names(conn)
    if table_name not in tables:
        print(f"No such table: {table_name!r}. Run with no arguments to see the full table list.")
        return

    count = await _row_count(conn, table_name)
    rows = await conn.fetch(f'SELECT * FROM "{table_name}" LIMIT $1', limit)

    print("=" * 88)
    print(f"TABLE: {table_name}  ({count} row(s) total, showing up to {limit})")
    print("=" * 88)
    for i, row in enumerate(rows):
        print()
        print(SEP)
        print()
        print(f"[ROW {i + 1}/{len(rows)}]")
        for key, value in dict(row).items():
            print(f"    {key}: {value}")
    print()
    print(SEP)
    print("=" * 88)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", action="store_true", help="Show one row per chat session + what it owns.")
    parser.add_argument("--table", help="Dump rows from one specific table (app table or xlsx_* sheet table).")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to show with --table (default 20).")
    args = parser.parse_args()

    settings = get_settings()
    conn = await asyncpg.connect(_dsn(settings.database_url))
    try:
        if args.table:
            await print_table_rows(conn, args.table, args.limit)
        elif args.sessions:
            await print_sessions(conn)
        else:
            await print_overview(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
