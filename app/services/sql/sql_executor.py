from __future__ import annotations

import asyncio
import decimal
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)


class SQLExecutionError(RuntimeError):
    pass


@dataclass
class SQLExecutionResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


async def execute_readonly(
    db: AsyncSession, sql: str, *, timeout_seconds: int, max_rows: int
) -> SQLExecutionResult:
    try:
        result = await asyncio.wait_for(db.execute(text(sql)), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise SQLExecutionError(f"Query exceeded {timeout_seconds}s timeout.") from exc
    except SQLAlchemyError as exc:
        raise SQLExecutionError(f"Query failed: {exc}") from exc

    columns = list(result.keys())
    rows = [dict(zip(columns, row, strict=True)) for row in result.fetchmany(max_rows)]
    return SQLExecutionResult(columns=columns, rows=rows, row_count=len(rows))


def _format_value(value: Any) -> str:
    """Formats one result cell for the answer LLM. A Postgres NUMERIC
    column - e.g. an AVG() result - comes back as a Decimal carrying its
    full division precision (0.18076923076923076923, not 0.18), which
    reads as a raw computational artifact, not an answer. Rounding it
    here, once, deterministically, means the LLM is never the one
    deciding how much precision is reasonable - it only ever sees an
    already-sensible number. Integers (a COUNT() result), booleans, and
    non-numeric values pass through unchanged - nothing to round."""
    if isinstance(value, bool) or not isinstance(value, (float, decimal.Decimal)):
        return str(value)
    text = f"{round(float(value), 4):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def format_result_for_llm(result: SQLExecutionResult, sql: str, *, max_rows_shown: int = 20) -> str:
    """Renders a SQL result as evidence text for the answer-generation LLM.

    Always includes the executed SQL, not just the raw rows: a bare row
    list gives the LLM no way to verify that the rows satisfy a filter
    condition named in the question (e.g. rows are just names, question
    asks "which had Minor severity") - without the query visible, a
    strictly-grounded model may refuse rather than assume the filter was
    applied correctly.
    """
    header = f"Query executed: {sql}\n\n"
    if not result.rows:
        return header + "Database result: (no rows)"
    if len(result.rows) == 1 and len(result.columns) == 1:
        value = result.rows[0][result.columns[0]]
        return header + f"Database result: {_format_value(value)}"

    lines = [", ".join(result.columns)]
    for row in result.rows[:max_rows_shown]:
        lines.append(", ".join(_format_value(row[c]) for c in result.columns))
    truncated = "" if len(result.rows) <= max_rows_shown else f" (+{len(result.rows) - max_rows_shown} more rows)"
    return header + "Database result:\n" + "\n".join(lines) + truncated
