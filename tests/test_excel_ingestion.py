"""Excel schema extraction and CSV ingestion.

Uploads go through the real HTTP API; schema/row assertions then query the
database directly to verify what the pipeline actually persisted.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import select, text

from app.db.session import SessionLocal
from app.models.excel import ExcelColumnSchema, ExcelSheet
from tests.conftest import upload_file
from tests.factories import make_csv_bytes, make_messy_xlsx_bytes, make_xlsx_bytes

pytestmark = pytest.mark.asyncio


async def _columns_by_name(session_id: str) -> dict[str, ExcelColumnSchema]:
    async with SessionLocal() as db:
        result = await db.execute(
            select(ExcelSheet).where(ExcelSheet.session_id == session_id)
        )
        sheets = list(result.scalars().all())
        assert sheets, "expected at least one ingested sheet"
        cols_result = await db.execute(
            select(ExcelColumnSchema).where(ExcelColumnSchema.sheet_id == sheets[0].sheet_id)
        )
        return {c.column_name: c for c in cols_result.scalars().all()}, sheets[0]


async def test_xlsx_schema_extraction_infers_types(client, session_id, auth_headers):
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Carol", "Dave", "Erin"],
            "Age": [30, 25, 35, 40, 28],
            "Department": ["IT", "HR", "IT", "Sales", "IT"],
            "Salary": [90000.5, 60000.25, 95000.75, 70000.1, 88000.4],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="employees.xlsx",
        content=make_xlsx_bytes({"Employees": df}),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "ready"

    columns, sheet = await _columns_by_name(session_id)
    assert sheet.sheet_name == "Employees"
    assert sheet.row_count == 5

    assert columns["Name"].data_type == "TEXT"
    assert columns["Name"].is_text is True

    assert columns["Age"].data_type == "INTEGER"
    assert columns["Age"].is_numeric is True

    assert columns["Salary"].data_type == "NUMERIC"
    assert columns["Salary"].is_numeric is True

    assert columns["Department"].is_categorical is True
    assert columns["Department"].description  # programmatically generated, non-empty

    # The actual rows landed in the dynamically-created Postgres/SQLite table.
    async with SessionLocal() as db:
        count = await db.scalar(text(f'SELECT COUNT(*) FROM "{sheet.table_name}"'))
    assert count == 5


async def test_csv_ingestion_creates_one_sheet(client, session_id, auth_headers):
    df = pd.DataFrame({"Product": ["Laptop", "Mouse", "Keyboard"], "Revenue": [1200.50, 25.0, 45.0]})
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="sales.csv",
        content=make_csv_bytes(df), content_type="text/csv",
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "ready"

    columns, sheet = await _columns_by_name(session_id)
    assert sheet.row_count == 3
    assert columns["Revenue"].data_type == "NUMERIC"
    assert columns["Product"].is_text is True


async def test_messy_xlsx_with_title_banner_finds_real_headers(client, session_id, auth_headers):
    """Real-world reports often have a merged title row and a blank row
    above the actual headers, plus a trailing footnote. A naive
    header=0 read would treat the title as column names and produce
    garbage - see app/services/excel/header_detection.py."""
    df = pd.DataFrame(
        {
            "Region": ["North America", "EMEA", "APAC"],
            "Product": ["CVC-14G", "CVC-14G", "PICC-4FR"],
            "UnitsSold": [4200, 5100, 950],
            "Revenue": [203700, 247350, 46075],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="regional_report.xlsx",
        content=make_messy_xlsx_bytes("Sales", df, title="Q1 2024 Regional Sales Report - CONFIDENTIAL"),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "ready"

    columns, sheet = await _columns_by_name(session_id)
    assert sheet.row_count == 3  # the footnote row must not be counted as data
    assert set(columns.keys()) == {"Region", "Product", "UnitsSold", "Revenue"}
    assert columns["Revenue"].is_numeric is True
    assert columns["UnitsSold"].data_type == "INTEGER"
    assert columns["Region"].is_text is True

    async with SessionLocal() as db:
        total = await db.scalar(text(f'SELECT SUM("revenue") FROM "{sheet.table_name}"'))
    assert total == 203700 + 247350 + 46075


async def test_date_columns_are_detected(client, session_id, auth_headers):
    df = pd.DataFrame(
        {
            "OrderDate": ["2024-01-05", "2024-02-14", "2024-03-01"],
            "Amount": [100, 200, 150],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="orders.csv",
        content=make_csv_bytes(df), content_type="text/csv",
    )
    assert resp.status_code == 201, resp.text

    columns, _sheet = await _columns_by_name(session_id)
    assert columns["OrderDate"].is_date is True
    assert columns["OrderDate"].data_type == "TIMESTAMP"
