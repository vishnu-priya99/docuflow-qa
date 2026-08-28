from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.excel import ExcelSheet


async def get_session_sheets(db: AsyncSession, *, session_id: str, sheet_ids: list[str] | None = None) -> list[ExcelSheet]:
    stmt = (
        select(ExcelSheet)
        .where(ExcelSheet.session_id == session_id)
        .options(selectinload(ExcelSheet.columns), selectinload(ExcelSheet.workbook))
    )
    if sheet_ids:
        stmt = stmt.where(ExcelSheet.sheet_id.in_(sheet_ids))
    result = await db.execute(stmt)
    return list(result.scalars().all())


def build_schema_description(sheets: list[ExcelSheet]) -> str:
    """Renders each sheet's schema using its PHYSICAL (queryable) column
    names - never the original spreadsheet header text. Generated SQL is
    only ever validated against/executed with physical names (see
    sanitize.py), so showing anything else here would let the LLM write SQL
    that fails on a case-sensitive backend like PostgreSQL even though it
    might happen to work on SQLite's case-insensitive identifier matching.
    The original header is kept as a trailing comment for the LLM's context.

    Categorical columns also get their inferred description (which carries
    real sample values, in their real case) appended - without it, a model
    filtering on e.g. "TestParameter" has no way to know the stored values
    are Title Case ("Lumen Friction Coefficient") rather than whatever
    casing it guesses, and a case-sensitive equality filter then silently
    matches zero rows instead of erroring.
    """
    blocks = []
    for sheet in sheets:
        col_lines = []
        for c in sheet.columns:
            comments = []
            if c.physical_name != c.column_name:
                comments.append(f'"{c.column_name}"')
            if c.is_categorical and c.description:
                comments.append(c.description)
            note = f"  -- {' | '.join(comments)}" if comments else ""
            col_lines.append(f'  "{c.physical_name}" {c.data_type}{note}')
        cols = ",\n".join(col_lines)
        blocks.append(
            f"-- Sheet: {sheet.sheet_name} (from {sheet.workbook.filename if sheet.workbook else ''})\n"
            f"{sheet.table_name}(\n{cols}\n)"
        )
    return "\n\n".join(blocks)


def allowed_table_names(sheets: list[ExcelSheet]) -> set[str]:
    return {s.table_name for s in sheets}
