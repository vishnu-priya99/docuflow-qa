from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import LLMProvider
from app.services.reranking.base import RerankProvider
from app.vector.qdrant_client import QdrantService


@dataclass
class GraphDeps:
    """Request-scoped dependencies, threaded through LangGraph via
    RunnableConfig["configurable"]["deps"] rather than the graph state,
    since they aren't serializable/checkpoint-safe state."""

    db: AsyncSession
    readonly_db: AsyncSession
    qdrant: QdrantService
    embedder: EmbeddingProvider
    llm: LLMProvider
    reranker: RerankProvider
    settings: Settings
