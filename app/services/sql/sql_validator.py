"""Defense-in-depth validation for LLM-generated SQL.

Never execute raw LLM SQL without going through this module first. Layers:
  1. Parse with sqlglot; reject anything that isn't exactly one SELECT.
  2. Reject any write/DDL node type anywhere in the parse tree.
  3. Restrict every referenced table to an explicit allow-list (the current
     session's structured-data tables) - this is what enforces session
     isolation for generated SQL, independent of the read-only DB role.
  4. Cap returned rows by injecting/lowering a LIMIT clause.

In production this is paired with a dedicated read-only PostgreSQL role
(DATABASE_URL_READONLY / scripts/init_readonly_role.sql) as a second,
DB-enforced layer - see app/db/session.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_FORBIDDEN_NODE_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
    exp.Merge,
    exp.Command,  # catch-all for statements sqlglot can't classify (e.g. GRANT variants)
    exp.Attach,
    exp.Pragma,
)


class SQLValidationError(ValueError):
    pass


@dataclass
class ValidatedQuery:
    sql: str
    tables: list[str]


# Fixed to Postgres, this project's only supported database (see
# config.py's _require_postgres) - not exposed as a parameter, since
# nothing here ever needs a different dialect.
_DIALECT = "postgres"


def validate_sql(
    raw_sql: str,
    *,
    allowed_tables: set[str],
    max_rows: int,
) -> ValidatedQuery:
    sql = raw_sql.strip().rstrip(";").strip()
    if not sql or sql.upper() == "NO_QUERY":
        raise SQLValidationError("NO_QUERY")

    try:
        statements = sqlglot.parse(sql, read=_DIALECT)
    except Exception as exc:  # noqa: BLE001 - sqlglot raises its own error types
        raise SQLValidationError(f"Could not parse generated SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SQLValidationError("Only read-only SELECT statements are allowed.")

    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise SQLValidationError(f"Statement contains a forbidden operation: {type(node).__name__}")

    tables = {t.name for t in statement.find_all(exp.Table)}
    disallowed = {t for t in tables if t not in allowed_tables}
    if disallowed:
        raise SQLValidationError(f"Query references tables outside this session: {sorted(disallowed)}")

    existing_limit = statement.args.get("limit")
    if existing_limit is None:
        statement = statement.limit(max_rows)
    else:
        try:
            current = int(existing_limit.expression.this)
            if current > max_rows:
                statement = statement.limit(max_rows)
        except (AttributeError, TypeError, ValueError):
            statement = statement.limit(max_rows)

    final_sql = statement.sql(dialect=_DIALECT)
    return ValidatedQuery(sql=final_sql, tables=sorted(tables))
