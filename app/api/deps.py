"""FastAPI dependency providers.

Auth model: the login page only collects a user_id (no password, per spec).
The client stores it and sends it back on every request as the X-User-Id
header; get_current_user_id is the single choke point that establishes
"who is asking" for every downstream query, so session/data isolation can
be enforced consistently everywhere.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.db.session import get_db, get_readonly_db
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.factory import get_embedding_provider
from app.services.llm.base import LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.reranking.base import RerankProvider
from app.services.reranking.factory import get_rerank_provider
from app.services.storage.base import StorageBackend
from app.services.storage.factory import get_storage_backend
from app.vector.qdrant_client import QdrantService, get_qdrant_service

__all__ = [
    "get_current_user_id",
    "get_db",
    "get_readonly_db",
    "get_embedder",
    "get_llm",
    "get_reranker",
    "get_storage",
    "get_qdrant",
]


async def get_current_user_id(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> str:
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id header - log in first.",
        )
    return x_user_id.strip()


def get_embedder() -> EmbeddingProvider:
    return get_embedding_provider()


def get_llm() -> LLMProvider:
    return get_llm_provider()


def get_reranker() -> RerankProvider:
    return get_rerank_provider()


def get_storage() -> StorageBackend:
    return get_storage_backend()


def get_qdrant() -> QdrantService:
    return get_qdrant_service()
