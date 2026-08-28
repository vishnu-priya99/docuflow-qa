"""DDL + bulk insert for the per-sheet structured-data tables.

Table/column names reaching this module have already been through
sanitize.py and are restricted to [a-z0-9_], so building DDL strings from
them is safe (SQL identifiers cannot be bind-parameterized).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.excel.schema_inference import ColumnInference

_SQL_TYPE = {
    "TEXT": "TEXT",
    "INTEGER": "BIGINT",
    # Fixed scale, not bare NUMERIC: a Python float64 (e.g. from round(x, 2))
    # is rarely an exact binary value, and an unconstrained NUMERIC column
    # stores that exact binary value's full decimal expansion - SUM/AVG
    # over such a column then surfaces 15+ digits of float noise (e.g.
    # "2923028.00999999997657..." instead of "2923028.01"). Scale 4 keeps
    # plenty of precision for money/measurements while storing a clean value.
    "NUMERIC": "NUMERIC(18, 4)",
    "BOOLEAN": "BOOLEAN",
    "TIMESTAMP": "TIMESTAMP",
}


async def create_sheet_table(
    db: AsyncSession, *, table_name: str, columns: list[tuple[str, ColumnInference]]
) -> None:
    col_defs = ", ".join(f'"{physical}" {_SQL_TYPE[c.data_type]}' for physical, c in columns)
    ddl = f'CREATE TABLE "{table_name}" ("_row_id" BIGINT, {col_defs})'
    await db.execute(text(ddl))


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)) or pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


async def insert_sheet_rows(
    db: AsyncSession,
    *,
    table_name: str,
    df: pd.DataFrame,
    physical_names: list[str],
    column_types: list[str],
) -> int:
    if df.empty:
        return 0

    columns_sql = ", ".join(f'"{p}"' for p in ["_row_id", *physical_names])
    params_sql = ", ".join(f":{p}" for p in ["_row_id", *physical_names])
    insert_sql = f'INSERT INTO "{table_name}" ({columns_sql}) VALUES ({params_sql})'

    rows: list[dict[str, Any]] = []
    for row_id, (_, row) in enumerate(df.iterrows()):
        record: dict[str, Any] = {"_row_id": row_id}
        for original_col, physical, data_type in zip(df.columns, physical_names, column_types, strict=True):
            value = _to_python(row[original_col])
            if data_type == "TIMESTAMP" and value is not None and not isinstance(value, pd.Timestamp):
                try:
                    parsed = pd.to_datetime(value, errors="coerce", format="mixed")
                    value = None if pd.isna(parsed) else parsed.to_pydatetime()
                except (TypeError, ValueError):
                    value = None
            record[physical] = value
        rows.append(record)

    await db.execute(text(insert_sql), rows)
    return len(rows)


async def drop_session_tables(db: AsyncSession, table_names: list[str]) -> None:
    for table_name in table_names:
        await db.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
