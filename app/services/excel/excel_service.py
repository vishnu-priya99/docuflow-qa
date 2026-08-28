"""Orchestrates the XLSX/CSV structured-data pipeline (spec section 4):

file -> parse -> detect sheets/table -> infer schema -> store rows in
Postgres -> generate semantic sheet representation -> embed into Qdrant.

The full workbook/CSV is never sent to the LLM - only the small semantic
summary text (embedded) and, later, the inferred schema description (used
for SQL generation) ever reach it.
"""
from __future__ import annotations

import io
import os

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.base import new_uuid
from app.models.excel import ExcelColumnSchema, ExcelSheet, ExcelWorkbook
from app.models.file import FileRecord
from app.services.embeddings.base import EmbeddingProvider
from app.services.excel.dynamic_tables import create_sheet_table, insert_sheet_rows
from app.services.excel.header_detection import normalize_sheet
from app.services.excel.sanitize import build_table_name, sanitize_column_name
from app.services.excel.schema_inference import infer_column
from app.vector.qdrant_client import QdrantService, SheetPayload

logger = get_logger(__name__)


def parse_workbook(*, file_type: str, filename: str, content: bytes) -> dict[str, pd.DataFrame]:
    """Reads every sheet raw (no assumed header row) and runs header
    detection on each - real-world spreadsheets often have a title/banner
    row above the actual headers (see header_detection.py)."""
    if file_type == "csv":
        raw = pd.read_csv(io.BytesIO(content), header=None)
        sheet_name = os.path.splitext(filename)[0] or "Sheet1"
        return {sheet_name: normalize_sheet(raw)}
    if file_type == "xlsx":
        raw_sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, engine="openpyxl")
        return {name: normalize_sheet(raw) for name, raw in raw_sheets.items()}
    raise ValueError(f"Not a structured-data file type: {file_type}")


def _build_semantic_summary(*, filename: str, sheet_name: str, row_count: int, columns) -> str:
    col_lines = "\n".join(f"- {c.column_name} ({c.data_type}): {c.description}" for _, c in columns)
    return (
        f"File: {filename}\nSheet: {sheet_name}\nRows: {row_count}\n"
        f"Columns:\n{col_lines}"
    )


async def ingest_excel(
    *,
    db: AsyncSession,
    qdrant: QdrantService,
    embedder: EmbeddingProvider,
    file_record: FileRecord,
    content: bytes,
) -> tuple[ExcelWorkbook, int]:
    """Returns (workbook, total_rows_ingested_across_all_sheets) - callers
    should treat 0 rows as an ingestion failure, not a silent success."""
    sheets = parse_workbook(file_type=file_record.file_type, filename=file_record.filename, content=content)

    workbook = ExcelWorkbook(
        workbook_id=new_uuid(),
        session_id=file_record.session_id,
        user_id=file_record.user_id,
        file_id=file_record.file_id,
        filename=file_record.filename,
    )
    db.add(workbook)

    points: list[tuple[str, list[float], dict]] = []
    total_rows = 0

    for sheet_name, df in sheets.items():
        df = df.dropna(axis="columns", how="all")
        sheet_id = new_uuid()
        table_name = build_table_name(session_id=file_record.session_id, sheet_name=sheet_name, sheet_id=sheet_id)

        existing_physical: set[str] = set()
        inferred: list[tuple[str, object]] = []
        for col in df.columns:
            physical = sanitize_column_name(str(col), existing=existing_physical)
            column_inference = infer_column(df[col], str(col))
            inferred.append((physical, column_inference))

        await create_sheet_table(db, table_name=table_name, columns=inferred)
        row_count = await insert_sheet_rows(
            db,
            table_name=table_name,
            df=df,
            physical_names=[p for p, _ in inferred],
            column_types=[c.data_type for _, c in inferred],
        )

        summary = _build_semantic_summary(
            filename=file_record.filename, sheet_name=sheet_name, row_count=row_count, columns=inferred
        )

        sheet = ExcelSheet(
            sheet_id=sheet_id,
            workbook_id=workbook.workbook_id,
            session_id=file_record.session_id,
            user_id=file_record.user_id,
            sheet_name=str(sheet_name),
            table_name=table_name,
            row_count=row_count,
            semantic_summary=summary,
        )
        db.add(sheet)

        for physical, column_inference in inferred:
            db.add(
                ExcelColumnSchema(
                    sheet_id=sheet_id,
                    column_name=column_inference.column_name,
                    physical_name=physical,
                    data_type=column_inference.data_type,
                    is_numeric=column_inference.is_numeric,
                    is_date=column_inference.is_date,
                    is_text=column_inference.is_text,
                    is_categorical=column_inference.is_categorical,
                    description=column_inference.description,
                )
            )

        vector = await embedder.embed_one(summary)
        payload = SheetPayload(
            user_id=file_record.user_id,
            session_id=file_record.session_id,
            file_id=file_record.file_id,
            filename=file_record.filename,
            workbook_id=workbook.workbook_id,
            sheet_id=sheet_id,
            sheet_name=str(sheet_name),
            table_name=table_name,
            text=summary,
        )
        points.append((new_uuid(), vector, payload.to_payload()))
        total_rows += row_count

        logger.info("=" * 88)
        logger.info("[EXCEL INGEST] %s | sheet '%s'", file_record.filename, sheet_name)
        logger.info("=" * 88)
        logger.info("")
        logger.info("[COLUMNS DETECTED] %d column(s), type inferred from the real data:", len(inferred))
        for physical, column_inference in inferred:
            suffix = f" | {column_inference.description}" if column_inference.description else ""
            logger.info(
                "  - %s -> %s (%s)%s",
                column_inference.column_name, physical, column_inference.data_type, suffix,
            )
        logger.info("")
        logger.info("*" * 50)
        logger.info("")
        logger.info(
            "[TABLE CREATED] a real Postgres table now holds this sheet's data:\n  %s (%d row(s) inserted)",
            table_name, row_count,
        )
        logger.info("")
        logger.info("*" * 50)
        logger.info("")
        # This is the ONLY thing about this sheet that ever reaches an
        # embedding model or gets stored in Qdrant - a small description of
        # what the sheet contains, not the sheet's actual data. Used purely
        # to let a HYBRID/multi-file question find the right sheet; the
        # real row data is only ever touched by a SQL query against the
        # table above, never read by an LLM.
        logger.info("[SEMANTIC SUMMARY] embedded into Qdrant for sheet discovery (not the sheet's data):")
        logger.info("")
        for line in summary.splitlines():
            logger.info(line)
        logger.info("")
        logger.info("*" * 50)
        logger.info("=" * 88)

    await qdrant.upsert(points)
    await db.flush()
    return workbook, total_rows
