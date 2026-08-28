from __future__ import annotations

from abc import ABC, abstractmethod


class RerankProvider(ABC):
    """Configurable second-stage relevance reranker for semantic retrieval.

    Decoupled from LLMProvider on purpose: reranking is a different kind of
    model (a cross-encoder scores query/passage pairs directly; an LLM
    reasons about relevance via a prompt) and should be swappable
    independently of which chat model is configured.
    """

    @abstractmethod
    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        """Return indices into ``candidates``, most relevant to ``question``
        first, length at most ``top_n``. May return fewer than ``top_n``
        (including none) when fewer candidates are actually relevant."""
