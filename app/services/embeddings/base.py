from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Configurable embedding backend."""

    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
