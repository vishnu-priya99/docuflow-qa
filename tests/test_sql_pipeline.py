"""End-to-end STRUCTURED question answering: COUNT DISTINCT, SUM, AVG,
GROUP BY, date filtering, and SQL injection/read-only safety through the
full API -> LangGraph -> SQL validator -> Postgres/SQLite pipeline.

The LLM's SQL-generation step is swapped for a fully-controlled fake via
FastAPI dependency overrides, so these tests assert real pipeline behavior
(validation, execution, session-table scoping) without depending on how
good any particular LLM/heuristic is at writing SQL.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import select, text

from app.api.deps import get_llm
from app.db.session import SessionLocal
from app.main import app
from app.models.excel import ExcelSheet
from app.services.llm.mock_llm import MockLLMProvider
from tests.conftest import ask, upload_file
from tests.factories import make_csv_bytes, make_xlsx_bytes

pytestmark = pytest.mark.asyncio


@pytest.fixture
def override_llm():
    applied: list[bool] = []

    def _apply(provider):
        app.dependency_overrides[get_llm] = lambda: provider
        applied.append(True)

    yield _apply
    if applied:
        app.dependency_overrides.pop(get_llm, None)


async def _table_name_for(session_id: str) -> str:
    async with SessionLocal() as db:
        result = await db.execute(select(ExcelSheet).where(ExcelSheet.session_id == session_id))
        sheet = result.scalars().first()
        assert sheet is not None
        return sheet.table_name


async def _upload_employees(client, session_id, auth_headers) -> str:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Carol", "Dave", "Alice"],
            "Department": ["IT", "HR", "IT", "Sales", "IT"],
            "Salary": [90000, 60000, 95000, 70000, 91000],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="employees.xlsx",
        content=make_xlsx_bytes({"Employees": df}),
    )
    assert resp.status_code == 201, resp.text
    return await _table_name_for(session_id)


async def test_count_distinct(client, session_id, auth_headers, override_llm):
    table = await _upload_employees(client, session_id, auth_headers)
    question = "how many unique employee names are there"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={question: f'SELECT COUNT(DISTINCT "name") FROM {table}'},
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert result["question_type"] == "STRUCTURED"
    assert result["answer"].strip() == "4"  # Alice appears twice


async def test_sum(client, session_id, auth_headers, override_llm):
    table = await _upload_employees(client, session_id, auth_headers)
    question = "what is the total salary across all employees"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={question: f'SELECT SUM("salary") FROM {table}'},
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert float(result["answer"]) == 90000 + 60000 + 95000 + 70000 + 91000


async def test_average_with_filter(client, session_id, auth_headers, override_llm):
    table = await _upload_employees(client, session_id, auth_headers)
    question = "what is the average salary in the IT department"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={question: f'SELECT AVG("salary") FROM {table} WHERE "department" = \'IT\''},
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert float(result["answer"]) == pytest.approx((90000 + 95000 + 91000) / 3)


async def test_group_by(client, session_id, auth_headers, override_llm):
    table = await _upload_employees(client, session_id, auth_headers)
    question = "what is the total salary by department"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={
                question: f'SELECT "department", SUM("salary") AS total FROM {table} GROUP BY "department"'
            },
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert result["sources"][0]["row_count"] == 3  # IT, HR, Sales
    assert "Database" not in result["answer"]  # answer should be synthesized, not raw dump label


async def test_date_filtering(client, session_id, auth_headers, override_llm):
    df = pd.DataFrame(
        {
            "OrderDate": ["2024-01-05", "2024-03-14", "2024-03-20", "2024-05-01"],
            "Revenue": [100, 200, 300, 400],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="sales.csv", content=make_csv_bytes(df)
    )
    assert resp.status_code == 201, resp.text
    table = await _table_name_for(session_id)

    question = "what was total revenue in march 2024"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={
                question: (
                    f'SELECT SUM("revenue") FROM {table} '
                    "WHERE \"orderdate\" >= '2024-03-01' AND \"orderdate\" < '2024-04-01'"
                )
            },
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert float(result["answer"]) == 500  # 200 + 300


async def test_injected_sql_injection_attempt_is_blocked_end_to_end(client, session_id, auth_headers, override_llm):
    table = await _upload_employees(client, session_id, auth_headers)
    question = "please drop the employee table"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={question: f'SELECT 1; DROP TABLE {table};'},
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert result["answer"] == "I couldn't find that information in the uploaded files."

    # The table must still exist with all rows intact.
    async with SessionLocal() as db:
        count = await db.scalar(text(f'SELECT COUNT(*) FROM "{table}"'))
    assert count == 5


async def test_query_reaching_outside_session_table_is_blocked(client, session_id, auth_headers, override_llm):
    await _upload_employees(client, session_id, auth_headers)
    question = "show me all users in the system"
    override_llm(
        MockLLMProvider(
            route_answers={question: "STRUCTURED"},
            sql_answers={question: "SELECT * FROM users"},
        )
    )
    result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    assert result["answer"] == "I couldn't find that information in the uploaded files."
