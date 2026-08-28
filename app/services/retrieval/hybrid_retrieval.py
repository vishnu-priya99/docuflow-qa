"""Sheet discovery for HYBRID questions: Qdrant identifies which
workbook/sheet a question is about before SQL generation runs against it.
"""
from __future__ import annotations

from typing import Any

from app.services.embeddings.base import EmbeddingProvider
from app.vector.qdrant_client import QdrantService


async def discover_sheets(
    *,
    qdrant: QdrantService,
    embedder: EmbeddingProvider,
    question: str,
    user_id: str,
    session_id: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    vector = await embedder.embed_one(question)
    return await qdrant.search(
        query_vector=vector,
        user_id=user_id,
        session_id=session_id,
        content_type="excel_sheet",
        top_k=top_k,
    )
