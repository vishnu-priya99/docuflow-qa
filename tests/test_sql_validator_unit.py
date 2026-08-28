"""Pure unit tests for the SQL safety validator - fast, deterministic,
no DB/HTTP needed. Covers the read-only/injection-safety checklist item."""
from __future__ import annotations

import pytest

from app.services.sql.sql_validator import SQLValidationError, validate_sql

ALLOWED = {"xlsx_employees_ab12cd34"}


def test_valid_select_passes_and_gets_limited():
    result = validate_sql(
        'SELECT COUNT(DISTINCT "Name") FROM xlsx_employees_ab12cd34',
        allowed_tables=ALLOWED,
        max_rows=100,
    )
    assert "LIMIT 100" in result.sql.upper()
    assert result.tables == ["xlsx_employees_ab12cd34"]


def test_no_query_sentinel_raises():
    with pytest.raises(SQLValidationError):
        validate_sql("NO_QUERY", allowed_tables=ALLOWED, max_rows=100)


def test_empty_sql_raises():
    with pytest.raises(SQLValidationError):
        validate_sql("", allowed_tables=ALLOWED, max_rows=100)


@pytest.mark.parametrize(
    "malicious_sql",
    [
        'DROP TABLE xlsx_employees_ab12cd34',
        'DELETE FROM xlsx_employees_ab12cd34',
        'UPDATE xlsx_employees_ab12cd34 SET "Salary" = 0',
        'INSERT INTO xlsx_employees_ab12cd34 ("Name") VALUES (\'x\')',
        'TRUNCATE TABLE xlsx_employees_ab12cd34',
        'ALTER TABLE xlsx_employees_ab12cd34 ADD COLUMN hacked TEXT',
        'CREATE TABLE evil (id INTEGER)',
        'SELECT * FROM xlsx_employees_ab12cd34; DROP TABLE xlsx_employees_ab12cd34;',
        'SELECT 1; SELECT 2;',
        'GRANT ALL ON xlsx_employees_ab12cd34 TO public',
    ],
)
def test_write_and_ddl_statements_are_rejected(malicious_sql):
    with pytest.raises(SQLValidationError):
        validate_sql(malicious_sql, allowed_tables=ALLOWED, max_rows=100)


def test_stacked_select_then_drop_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql(
            'SELECT "Name" FROM xlsx_employees_ab12cd34; DROP TABLE users;',
            allowed_tables=ALLOWED,
            max_rows=100,
        )


def test_query_outside_session_tables_is_rejected():
    with pytest.raises(SQLValidationError, match="outside this session"):
        validate_sql("SELECT * FROM users", allowed_tables=ALLOWED, max_rows=100)


def test_subquery_reaching_outside_table_is_rejected():
    with pytest.raises(SQLValidationError, match="outside this session"):
        validate_sql(
            'SELECT "Name" FROM xlsx_employees_ab12cd34 '
            "WHERE \"Name\" IN (SELECT user_id FROM users)",
            allowed_tables=ALLOWED,
            max_rows=100,
        )


def test_existing_limit_larger_than_max_is_clamped():
    result = validate_sql(
        'SELECT "Name" FROM xlsx_employees_ab12cd34 LIMIT 999999',
        allowed_tables=ALLOWED,
        max_rows=50,
    )
    assert "LIMIT 50" in result.sql.upper()


def test_existing_limit_smaller_than_max_is_kept():
    result = validate_sql(
        'SELECT "Name" FROM xlsx_employees_ab12cd34 LIMIT 5',
        allowed_tables=ALLOWED,
        max_rows=500,
    )
    assert "LIMIT 5" in result.sql.upper()
