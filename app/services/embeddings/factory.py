from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.embeddings.base import EmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "local":
        from app.services.embeddings.local_embeddings import LocalHashingEmbeddingProvider

        return LocalHashingEmbeddingProvider(dim=settings.embedding_dim)
    if settings.embedding_provider == "ollama":
        from app.services.embeddings.ollama_embeddings import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            keep_alive=settings.ollama_keep_alive,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
