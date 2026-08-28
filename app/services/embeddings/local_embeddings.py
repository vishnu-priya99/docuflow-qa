from __future__ import annotations

import hashlib
import math
import re

from app.services.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider.

    Implements signed feature hashing (a bag-of-words vector projected into
    a fixed-size space via a hash function) so the app runs fully offline
    with no API key and no ML runtime. This trades semantic quality for
    zero external dependencies - swap EMBEDDING_PROVIDER=ollama (or add
    another provider) for production-grade retrieval quality.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector_for(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]
