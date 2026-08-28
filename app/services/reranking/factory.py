from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.reranking.base import RerankProvider


@lru_cache
def get_rerank_provider() -> RerankProvider:
    settings = get_settings()
    if settings.rerank_provider == "cross_encoder":
        from app.services.reranking.cross_encoder_rerank import CrossEncoderRerankProvider

        return CrossEncoderRerankProvider(
            model_name=settings.rerank_model,
            min_score=settings.rerank_min_score,
            score_margin=settings.rerank_score_margin,
            offline=settings.rerank_offline,
        )
    if settings.rerank_provider == "llm":
        from app.services.llm.factory import get_rerank_llm_provider
        from app.services.reranking.llm_rerank import LLMRerankProvider

        return LLMRerankProvider(get_rerank_llm_provider())
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")
