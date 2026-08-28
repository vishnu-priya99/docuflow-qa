from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.llm.base import LLMProvider


def _build_llm_provider(*, model: str) -> LLMProvider:
    """Builds a provider for the given model name, respecting LLM_PROVIDER
    (ollama vs. mock) - shared by get_llm_provider() (answers/routing, uses
    LLM_MODEL) and get_rerank_llm_provider() (reranking, uses a
    RERANK_LLM_MODEL that can be a different, smaller model)."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from app.services.llm.ollama_llm import OllamaLLMProvider

        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=model,
            max_tokens=settings.llm_max_tokens,
            keep_alive=settings.ollama_keep_alive,
        )
    if settings.llm_provider == "mock":
        from app.services.llm.mock_llm import MockLLMProvider

        return MockLLMProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    return _build_llm_provider(model=get_settings().llm_model)


@lru_cache
def get_rerank_llm_provider() -> LLMProvider:
    return _build_llm_provider(model=get_settings().rerank_llm_model)
