"""HYBRID question end-to-end: Qdrant sheet discovery -> schema selection
scoped to the discovered sheet -> SQL generation/validation/execution ->
answer generation. Mirrors the spec's laptop-sales example."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import select

from app.api.deps import get_llm
from app.db.session import SessionLocal
from app.main import app
from app.models.excel import ExcelSheet
from app.services.llm.mock_llm import MockLLMProvider
from tests.conftest import ask, upload_file
from tests.factories import make_csv_bytes, make_txt_bytes

pytestmark = pytest.mark.asyncio


async def test_hybrid_question_combines_discovery_and_sql(client, session_id, auth_headers):
    # An unrelated document, so has_documents=True as well as has_structured_data=True.
    await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="notes.txt",
        content=make_txt_bytes("General company notes with nothing about sales figures."),
    )

    df = pd.DataFrame(
        {
            "Product": ["Laptop", "Laptop", "Mouse", "Laptop"],
            "Month": ["January", "March", "March", "March"],
            "Revenue": [1000, 1500, 20, 2500],
        }
    )
    resp = await upload_file(
        client, session_id=session_id, headers=auth_headers, filename="laptop_sales.csv", content=make_csv_bytes(df)
    )
    assert resp.status_code == 201, resp.text

    async with SessionLocal() as db:
        result = await db.execute(select(ExcelSheet).where(ExcelSheet.session_id == session_id))
        table = result.scalars().one().table_name

    question = "Find the laptop sales data and tell me the total revenue in March."
    app.dependency_overrides[get_llm] = lambda: MockLLMProvider(
        route_answers={question: "HYBRID"},
        sql_answers={
            question: f"SELECT SUM(\"revenue\") FROM {table} WHERE \"month\" = 'March'"
        },
    )
    try:
        result = await ask(client, session_id=session_id, headers=auth_headers, question=question)
    finally:
        app.dependency_overrides.pop(get_llm, None)

    assert result["question_type"] == "HYBRID"
    assert float(result["answer"]) == 1500 + 20 + 2500
    assert any(s.get("type") == "structured_query" for s in result["sources"])
