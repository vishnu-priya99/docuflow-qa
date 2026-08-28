from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # --- Input ---
    user_id: str
    session_id: str
    question: str

    # --- Load session / routing ---
    has_documents: bool
    has_structured_data: bool
    question_type: str  # SEMANTIC | STRUCTURED | HYBRID

    # --- Semantic path ---
    retrieved_chunks: list[dict[str, Any]]

    # --- Hybrid discovery ---
    discovered_sheet_ids: list[str]

    # --- Structured / hybrid path ---
    schema_description: str
    allowed_tables: list[str]
    generated_sql: str
    validated_sql: str
    sql_columns: list[str]
    sql_rows: list[dict[str, Any]]
    sql_row_count: int
    sql_error: str | None

    # --- Shared evidence assembled for the answer step ---
    context: str
    sources: list[dict[str, Any]]

    # --- Output ---
    answer: str
    error: str | None
