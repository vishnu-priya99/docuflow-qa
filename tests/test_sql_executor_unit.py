"""Pure unit tests for format_result_for_llm - fast, deterministic, no DB
needed. Checks the executed SQL is always present in the formatted
evidence text: without it, a question naming a filter condition not
visibly reflected in the row values (e.g. "which customers had Minor
severity complaints?" against a bare list of names) gives the answer LLM
no way to verify the rows satisfy that condition. The actual grounding
behavior is an LLM property, not something a unit test can assert - these
tests only check the one thing it depends on.
"""
from __future__ import annotations

import decimal

from app.services.sql.sql_executor import SQLExecutionResult, _format_value, format_result_for_llm

SQL = 'SELECT "customer" FROM xlsx_complaints WHERE "severity" = \'Minor\''


def test_multi_row_result_includes_the_sql_query():
    result = SQLExecutionResult(
        columns=["customer"],
        rows=[{"customer": "Acme Corp"}, {"customer": "Beta Inc"}],
        row_count=2,
    )
    text = format_result_for_llm(result, SQL)
    assert SQL in text
    assert "Acme Corp" in text
    assert "Beta Inc" in text


def test_single_aggregate_result_includes_the_sql_query():
    result = SQLExecutionResult(columns=["count"], rows=[{"count": 7}], row_count=1)
    text = format_result_for_llm(result, SQL)
    assert SQL in text
    assert "Database result: 7" in text


def test_empty_result_includes_the_sql_query():
    result = SQLExecutionResult(columns=["customer"], rows=[], row_count=0)
    text = format_result_for_llm(result, SQL)
    assert SQL in text
    assert "(no rows)" in text


def test_row_truncation_still_works_with_sql_included():
    result = SQLExecutionResult(
        columns=["n"],
        rows=[{"n": i} for i in range(25)],
        row_count=25,
    )
    text = format_result_for_llm(result, SQL, max_rows_shown=20)
    assert SQL in text
    assert "+5 more rows" in text


def test_format_value_rounds_a_long_decimal_average():
    assert _format_value(decimal.Decimal("0.18076923076923076923")) == "0.1808"


def test_format_value_preserves_a_short_currency_decimal():
    assert _format_value(decimal.Decimal("661326.65")) == "661326.65"


def test_format_value_strips_trailing_zero_decimal_to_a_plain_integer():
    assert _format_value(decimal.Decimal("30.0000")) == "30"


def test_format_value_leaves_a_plain_int_count_untouched():
    assert _format_value(22) == "22"


def test_format_value_leaves_non_numeric_values_untouched():
    assert _format_value("Pass") == "Pass"
    assert _format_value(None) == "None"


def test_format_value_handles_a_plain_float_too():
    assert _format_value(0.18076923076923076923) == "0.1808"


def test_single_aggregate_result_rounds_a_long_decimal():
    result = SQLExecutionResult(
        columns=["avg"], rows=[{"avg": decimal.Decimal("0.18076923076923076923")}], row_count=1
    )
    text = format_result_for_llm(result, SQL)
    assert "Database result: 0.1808" in text
    assert "0.18076923076923076923" not in text
