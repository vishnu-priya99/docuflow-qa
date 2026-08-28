"""Unit tests for sql_generator._clean_sql - stripping a NO_QUERY marker a
model can append onto an otherwise valid query when hedging on a
multi-part question, without discarding the real query underneath it."""
from __future__ import annotations

import pytest

from app.services.llm.mock_llm import MockLLMProvider
from app.services.sql.sql_generator import _clean_sql, generate_sql

SCHEMA = 'employees(\n  "name" TEXT\n)'


def test_clean_sql_strips_a_trailing_no_query_after_a_real_query():
    raw = "SELECT COUNT(*) FROM readings WHERE \"testparameter\" = 'X' NO_QUERY"
    assert _clean_sql(raw) == "SELECT COUNT(*) FROM readings WHERE \"testparameter\" = 'X'"


def test_clean_sql_leaves_a_genuine_no_query_refusal_untouched():
    assert _clean_sql("NO_QUERY") == "NO_QUERY"
    assert _clean_sql("  NO_QUERY  ") == "NO_QUERY"


def test_clean_sql_leaves_a_normal_query_untouched():
    raw = "SELECT COUNT(*) FROM employees"
    assert _clean_sql(raw) == raw


def test_clean_sql_only_strips_a_trailing_marker_not_one_mid_string():
    # NO_QUERY appearing only as a trailing hedge gets stripped; nothing
    # observed or expected mid-query, so this is intentionally narrow.
    raw = "SELECT 'NO_QUERY reason' AS note FROM employees"
    assert _clean_sql(raw) == raw


@pytest.mark.asyncio
async def test_generate_sql_returns_the_cleaned_query_end_to_end():
    llm = MockLLMProvider(
        sql_answers={"q": "SELECT COUNT(*) FROM employees WHERE \"name\" = 'X' NO_QUERY"}
    )
    result = await generate_sql(llm, question="q", schema_description=SCHEMA)
    assert result == "SELECT COUNT(*) FROM employees WHERE \"name\" = 'X'"


@pytest.mark.asyncio
async def test_generate_sql_still_returns_no_query_for_an_empty_schema():
    llm = MockLLMProvider()
    result = await generate_sql(llm, question="q", schema_description="")
    assert result == "NO_QUERY"
