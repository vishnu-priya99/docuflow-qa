"""Second-stage reranking for semantic retrieval.

Qdrant's embedding search is a single-stage nearest-neighbor lookup - it
reliably finds plausible candidates but, on its own, surfaces some
unrelated ones alongside the real match (nothing separates "somewhat
similar" from "actually answers this"). This module fetches a wider
candidate pool from Qdrant, then hands it to a pluggable RerankProvider
(a cross-encoder model, an LLM call, or a no-op - see
app/services/reranking/) to re-score by genuine relevance and keep only
the top few.
"""
from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.services.reranking.base import RerankProvider

logger = get_logger(__name__)


async def rerank_results(
    reranker: RerankProvider, *, question: str, results: list[dict[str, Any]], top_n: int
) -> list[dict[str, Any]]:
    """Reorders/filters Qdrant search results (each a {"score", "payload"}
    dict) by the reranker's relevance judgment. Falls back to the original
    embedding-similarity order, truncated to top_n, if the rerank call
    fails or raises - reranking is a quality improvement, never a hard
    dependency for retrieval to keep working."""
    if not results:
        return []

    candidates = [r["payload"].get("text", "") for r in results]
    try:
        order = await reranker.rerank(question=question, candidates=candidates, top_n=top_n)
    except Exception:  # noqa: BLE001 - reranking must never break retrieval
        logger.warning("Reranking failed; falling back to embedding-similarity order", exc_info=True)
        return results[:top_n]

    # An empty list here is a legitimate "none of these are actually
    # relevant" verdict from the reranker - that's different from the call
    # failing, and must NOT fall back to showing the raw candidates
    # anyway, or reranking could never actually filter anything out.
    return [results[i] for i in order]
