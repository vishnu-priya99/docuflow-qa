from __future__ import annotations

import httpx

from app.services.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via a local Ollama server's batch /api/embed endpoint.

    EMBEDDING_DIM must match the chosen model's actual output size (e.g.
    768 for nomic-embed-text, 1024 for mxbai-embed-large) - QdrantService
    validates this at upsert time and raises a clear error on mismatch.
    """

    def __init__(self, base_url: str, model: str, dim: int, keep_alive: str = "30m") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self.dim = dim
        self._keep_alive = keep_alive
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.post(
            f"{self._base_url}/api/embed",
            # keep_alive: see Settings.ollama_keep_alive - without it, the
            # embedding model is also subject to Ollama's 5-minute default
            # unload, adding a reload delay to the next upload/question.
            json={"model": self._model, "input": texts, "keep_alive": self._keep_alive},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Ollama /api/embed returned no 'embeddings' field: {data}")
        return embeddings
