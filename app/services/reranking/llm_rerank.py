from __future__ import annotations

from app.core.logging import get_logger
from app.services.llm.base import LLMProvider
from app.services.reranking.base import RerankProvider

logger = get_logger(__name__)


class LLMRerankProvider(RerankProvider):
    """Reranks via a prompted call to an LLM (see LLMProvider.rerank /
    app/prompts/rerank_prompt.py) - the same provider type as LLM_PROVIDER
    (ollama/mock) but built with its own RERANK_LLM_MODEL setting (see
    app.services.llm.factory.get_rerank_llm_provider), independent of
    LLM_MODEL - RERANK_LLM_MODEL defaults to the same model as LLM_MODEL
    (see config.py for why: a smaller dedicated model sounds appealing but
    measured worse with local Ollama, which has to swap the loaded model
    on every question if the two differ). Zero extra dependencies, but
    still slower than a cross-encoder since it's a full generative model
    call per rerank."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        selected = await self._llm.rerank(question=question, candidates=candidates, top_n=top_n)

        logger.info("=" * 88)
        logger.info(
            '[RERANK] llm question="%s" | %d candidates -> %d selected (no continuous score - '
            "an LLM judgment, not a cross-encoder)",
            question, len(candidates), len(selected),
        )
        dropped = [i for i in range(len(candidates)) if i not in selected]
        for rank, i in enumerate(selected):
            logger.info("")
            logger.info("*" * 50)
            logger.info("")
            logger.info("[RERANK CANDIDATE] rank=%d index=%d KEEP", rank + 1, i)
            logger.info(candidates[i])
        for i in dropped:
            logger.info("")
            logger.info("*" * 50)
            logger.info("")
            logger.info("[RERANK CANDIDATE] index=%d drop", i)
            logger.info(candidates[i])
        logger.info("")
        logger.info("*" * 50)
        logger.info("=" * 88)

        return selected
