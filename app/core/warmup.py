"""Eager model warmup at application startup - see app.main.lifespan.

Every model this app uses (the reranker, and - if configured - Ollama's
chat/embedding models) is otherwise loaded lazily on first real use, so the
first question after a fresh start pays the full model-load cost on top of
normal inference time. Warming up here moves that cost to startup, before
any real question is ever asked, so a live session's first answer isn't the
slow one.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embeddings.factory import get_embedding_provider
from app.services.llm.factory import get_llm_provider
from app.services.reranking.factory import get_rerank_provider

logger = get_logger(__name__)


async def _timed(label: str, awaitable: Awaitable[object]) -> None:
    start = time.monotonic()
    try:
        await awaitable
        logger.info("[WARMUP] %s ready in %.1fs", label, time.monotonic() - start)
    except Exception:
        # A warmup failure (e.g. Ollama not running yet) must not crash
        # startup - the first real request just pays the lazy-load cost as
        # a fallback, exactly the pre-warmup behavior.
        logger.warning("[WARMUP] %s failed - will load lazily on first use", label, exc_info=True)


async def warm_up_models() -> None:
    """Runs each applicable warmup one at a time, not concurrently - the
    reranker load and the two Ollama calls are all genuinely CPU-heavy on
    this kind of local, GPU-less setup, and running them at once means
    they compete for the same cores instead of each getting full use of
    them. Sequential costs more total wall time at startup (a one-time,
    acceptable cost) in exchange for each step finishing as fast as it
    can, and for not compounding into the same kind of CPU contention this
    project measured between the reranker and Ollama during a real
    request (see cross_encoder_rerank.py history)."""
    settings = get_settings()

    # get_rerank_provider() does the actual (synchronous, CPU-bound) model
    # load when RERANK_PROVIDER=cross_encoder - off the event loop via
    # to_thread so it doesn't block startup outright. Cheap/no-op for
    # RERANK_PROVIDER=llm (just wraps the LLM provider, no separate load).
    await _timed("reranker", asyncio.to_thread(get_rerank_provider))

    # Constructing an OllamaLLMProvider/OllamaEmbeddingProvider only builds
    # the lightweight Python wrapper - Ollama itself only loads a model
    # into its own memory on the first real request for it, so warming up
    # means sending one minimal real call, not just instantiating the
    # provider.
    if settings.llm_provider == "ollama":
        llm = get_llm_provider()
        await _timed(
            f"LLM ({settings.llm_model})",
            llm.classify_question(question="warmup", has_documents=False, has_structured_data=False),
        )

    if settings.embedding_provider == "ollama":
        embedder = get_embedding_provider()
        await _timed(f"embeddings ({settings.embedding_model})", embedder.embed(["warmup"]))
