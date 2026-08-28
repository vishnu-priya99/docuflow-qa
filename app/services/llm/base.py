from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Configurable chat/generation backend.

    Exposes task-specific methods (rather than one raw ``complete``) so the
    LangGraph nodes stay provider-agnostic and a deterministic MockLLMProvider
    can back offline tests without parsing free-form prompt text.
    """

    @abstractmethod
    async def classify_question(
        self, *, question: str, has_documents: bool, has_structured_data: bool
    ) -> str:
        """Return exactly one of "SEMANTIC", "STRUCTURED", "HYBRID"."""

    @abstractmethod
    async def generate_sql(self, *, question: str, schema_description: str) -> str:
        """Return one SQL SELECT statement answering ``question`` against
        the given schema. No prose, no markdown fences."""

    @abstractmethod
    async def generate_answer(
        self, *, question: str, context: str, question_type: str
    ) -> str:
        """Return a short, direct answer grounded only in ``context``."""

    @abstractmethod
    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        """Return indices into ``candidates``, most relevant to ``question``
        first, length at most ``top_n``. May return fewer than ``top_n``
        (including none) when fewer candidates are actually relevant."""
