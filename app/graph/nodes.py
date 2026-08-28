"""LangGraph node implementations.

Each node is a plain async function: (state, config) -> partial state
update. Dependencies (DB session, Qdrant, embedder, LLM) come from
config["configurable"]["deps"] (see graph/deps.py) rather than the state
itself, since they aren't graph-serializable. The router node only
classifies the question - it never generates an answer itself.
"""
from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select

from app.core.logging import get_logger
from app.graph.deps import GraphDeps
from app.graph.state import GraphState
from app.models.excel import ExcelSheet
from app.models.file import FileRecord, SUPPORTED_DOCUMENT_TYPES
from app.services.retrieval.hybrid_retrieval import discover_sheets
from app.services.retrieval.reranker import rerank_results
from app.services.retrieval.semantic_retrieval import format_chunks_for_llm, retrieve_chunks
from app.services.sql import schema_service
from app.services.sql.sql_executor import SQLExecutionError, execute_readonly, format_result_for_llm
from app.services.sql.sql_generator import generate_sql
from app.services.sql.sql_validator import SQLValidationError, validate_sql

logger = get_logger(__name__)


def _deps(config: RunnableConfig) -> GraphDeps:
    return config["configurable"]["deps"]


async def load_session_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    session_id = state["session_id"]

    doc_count = await deps.db.scalar(
        select(func.count())
        .select_from(FileRecord)
        .where(
            FileRecord.session_id == session_id,
            FileRecord.status == "ready",
            FileRecord.file_type.in_(SUPPORTED_DOCUMENT_TYPES),
        )
    )
    sheet_count = await deps.db.scalar(
        select(func.count()).select_from(ExcelSheet).where(ExcelSheet.session_id == session_id)
    )
    return {"has_documents": bool(doc_count), "has_structured_data": bool(sheet_count)}


async def router_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    raw = await deps.llm.classify_question(
        question=state["question"],
        has_documents=state.get("has_documents", False),
        has_structured_data=state.get("has_structured_data", False),
    )
    question_type = raw.strip().upper()
    if question_type not in ("SEMANTIC", "STRUCTURED", "HYBRID"):
        question_type = "SEMANTIC" if state.get("has_documents") else "STRUCTURED"

    # Never route to a path with no data to serve it.
    if question_type == "STRUCTURED" and not state.get("has_structured_data"):
        question_type = "SEMANTIC" if state.get("has_documents") else "STRUCTURED"
    if question_type == "SEMANTIC" and not state.get("has_documents"):
        question_type = "STRUCTURED" if state.get("has_structured_data") else "SEMANTIC"
    if question_type == "HYBRID" and not (state.get("has_documents") and state.get("has_structured_data")):
        question_type = "STRUCTURED" if state.get("has_structured_data") else "SEMANTIC"

    logger.info(
        "[ROUTE] -> %s (has_documents=%s, has_structured_data=%s, raw_llm_response=%r)",
        question_type, state.get("has_documents", False), state.get("has_structured_data", False), raw,
    )
    return {"question_type": question_type}


async def semantic_retrieval_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    # Reranking is always on - RERANK_PROVIDER only accepts "cross_encoder"
    # or "llm" (see config.py), so a wide candidate pool is always fetched
    # and then narrowed by the reranker.
    deps = _deps(config)
    settings = deps.settings
    results = await retrieve_chunks(
        qdrant=deps.qdrant,
        embedder=deps.embedder,
        question=state["question"],
        user_id=state["user_id"],
        session_id=state["session_id"],
        top_k=settings.semantic_candidate_k,
    )
    results = await rerank_results(
        deps.reranker, question=state["question"], results=results, top_n=settings.semantic_top_k
    )
    context, sources = format_chunks_for_llm(results)
    return {"retrieved_chunks": results, "context": context, "sources": sources}


async def hybrid_discovery_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    results = await discover_sheets(
        qdrant=deps.qdrant,
        embedder=deps.embedder,
        question=state["question"],
        user_id=state["user_id"],
        session_id=state["session_id"],
    )
    sheet_ids = [r["payload"]["sheet_id"] for r in results if r.get("payload", {}).get("sheet_id")]
    return {"discovered_sheet_ids": sheet_ids}


async def schema_selection_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    sheet_ids = state.get("discovered_sheet_ids") or None
    sheets = await schema_service.get_session_sheets(deps.db, session_id=state["session_id"], sheet_ids=sheet_ids)
    allowed_tables = sorted(schema_service.allowed_table_names(sheets))
    schema_description = schema_service.build_schema_description(sheets)
    logger.info(
        "[SCHEMA] %d sheet(s) in scope%s: %s",
        len(sheets),
        " (Qdrant-discovered)" if sheet_ids else " (all session sheets)",
        [s.sheet_name for s in sheets],
    )
    # The schema text below is exactly what the LLM sees for SQL generation
    # (physical column names + inferred types + categorical sample values) -
    # logging it verbatim, not just the sheet names above, makes it possible
    # to see why the LLM wrote the SQL it did in [SQL] generated.
    logger.info("[SCHEMA] description sent to LLM:\n%s", schema_description)
    return {
        "schema_description": schema_description,
        "allowed_tables": allowed_tables,
    }


async def sql_generation_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    sql = await generate_sql(
        deps.llm, question=state["question"], schema_description=state.get("schema_description", "")
    )
    logger.info("[SQL] generated: %s", sql)
    return {"generated_sql": sql}


async def sql_validation_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    try:
        validated = validate_sql(
            state.get("generated_sql", ""),
            allowed_tables=set(state.get("allowed_tables", [])),
            max_rows=deps.settings.sql_max_rows,
        )
        logger.info("[SQL] validated: %s", validated.sql)
        return {"validated_sql": validated.sql, "sql_error": None}
    except SQLValidationError as exc:
        logger.warning("[SQL] validation REJECTED: %s", exc)
        return {"validated_sql": "", "sql_error": str(exc)}


async def sql_execution_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    # A HYBRID question can have real narrative evidence already gathered
    # by semantic_retrieval_node before this node runs - if the SQL half
    # fails or was never generated (e.g. NO_QUERY on a question the schema
    # alone can't fully answer), that evidence must survive, not be wiped
    # to empty, so the answer step can still fall back to an
    # evidence-based answer from the document half alone.
    if state.get("sql_error") or not state.get("validated_sql"):
        return {
            "context": state.get("context", ""), "sources": state.get("sources", []),
            "sql_rows": [], "sql_row_count": 0,
        }

    try:
        result = await execute_readonly(
            deps.readonly_db,
            state["validated_sql"],
            timeout_seconds=deps.settings.sql_query_timeout_seconds,
            max_rows=deps.settings.sql_max_rows,
        )
        logger.info("[SQL] executed -> %d row(s): %s", result.row_count, result.rows[:5])
    except SQLExecutionError as exc:
        logger.warning("[SQL] execution FAILED: %s", exc)
        return {
            "sql_error": str(exc), "context": state.get("context", ""), "sources": state.get("sources", []),
            "sql_rows": [], "sql_row_count": 0,
        }

    context = format_result_for_llm(result, state["validated_sql"])
    sources = [{"type": "structured_query", "sql": state["validated_sql"], "row_count": result.row_count}]

    # HYBRID also carries any narrative evidence already gathered.
    existing_context = state.get("context", "")
    if existing_context:
        context = f"{existing_context}\n\n{context}"
        sources = state.get("sources", []) + sources

    return {
        "sql_columns": result.columns,
        "sql_rows": result.rows,
        "sql_row_count": result.row_count,
        "context": context,
        "sources": sources,
    }


async def answer_generation_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    context = state.get("context", "")

    if state.get("sql_error") and not context:
        return {
            "answer": "I couldn't find that information in the uploaded files.",
            "sources": state.get("sources", []),
        }

    if not context.strip():
        return {
            "answer": "I couldn't find that information in the uploaded files.",
            "sources": state.get("sources", []),
        }

    answer = await deps.llm.generate_answer(
        question=state["question"], context=context, question_type=state.get("question_type", "SEMANTIC")
    )
    return {"answer": answer.strip(), "sources": state.get("sources", [])}
