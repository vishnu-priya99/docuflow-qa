"""Cross-encoder reranking via sentence-transformers.

A cross-encoder scores a (question, passage) pair directly through one
small model - purpose-trained for exactly this task - rather than asking a
full generative LLM to reason about relevance. Much faster (milliseconds
vs seconds-to-minutes for an LLM call) and gives a real relevance score
per pair instead of just an ordering.
"""
from __future__ import annotations

import asyncio
import os

from app.core.logging import get_logger
from app.services.reranking.base import RerankProvider

logger = get_logger(__name__)


def rank_and_filter(
    scores: list[float], *, top_n: int, min_score: float | None, score_margin: float | None = None
) -> list[int]:
    """Pure ranking logic, split out from the model call so it's testable
    without loading an actual cross-encoder: sorts indices by score
    descending, then applies whichever filters are set, then takes the
    top_n survivors.

    ``min_score``: an ABSOLUTE floor. Cross-encoder raw scores for MS
    MARCO-style models are not reliably comparable across passages of
    different length/style - a correct but verbose real passage can score
    lower than a short clean synthetic one, so a fixed floor can silently
    drop a correct answer.

    ``score_margin``: a RELATIVE cutoff - keep only candidates within
    ``score_margin`` of the top score in this batch. Adapts to whatever
    scale a given question/corpus produces, unlike a fixed floor. Failure
    mode: if nothing in the batch is relevant, scores can still cluster
    within one margin-width of each other, so this doesn't guarantee
    detecting "nothing is relevant" the way an absolute floor can.
    """
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if min_score is not None:
        ranked = [i for i in ranked if scores[i] >= min_score]
    if score_margin is not None and ranked:
        top_score = scores[ranked[0]]
        ranked = [i for i in ranked if scores[i] >= top_score - score_margin]
    return ranked[:top_n]


class CrossEncoderRerankProvider(RerankProvider):
    def __init__(
        self,
        model_name: str,
        min_score: float | None = None,
        score_margin: float | None = None,
        offline: bool = False,
    ) -> None:
        # Lazy import: sentence-transformers (+ torch) is a real dependency
        # weight-wise, only paid by whoever actually picks this provider -
        # RERANK_PROVIDER defaults to "llm" precisely so nobody is forced
        # to install it just to run the app.
        if offline:
            # huggingface_hub reads this real OS env var directly, not
            # anything our own Settings object passes in - see
            # config.py's rerank_offline for what this trades off.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._min_score = min_score
        self._score_margin = score_margin
        logger.info("Loaded cross-encoder reranker model %s", model_name)

    def _predict_sync(self, question: str, candidates: list[str]) -> list[float]:
        pairs = [(question, c) for c in candidates]
        return [float(s) for s in self._model.predict(pairs)]

    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        if not candidates:
            return []
        # CrossEncoder.predict() is a synchronous, CPU/GPU-bound call - run
        # it off the event loop so it doesn't block other requests.
        scores = await asyncio.to_thread(self._predict_sync, question, candidates)
        selected = rank_and_filter(
            scores, top_n=top_n, min_score=self._min_score, score_margin=self._score_margin
        )

        logger.info("=" * 88)
        logger.info(
            '[RERANK] cross_encoder question="%s" | %d candidates -> %d selected '
            "(min_score=%s, score_margin=%s)",
            question, len(candidates), len(selected), self._min_score, self._score_margin,
        )
        ranked_all = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        for rank, i in enumerate(ranked_all):
            tag = "KEEP" if i in selected else "drop"
            logger.info("")
            logger.info("*" * 50)
            logger.info("")
            logger.info("[RERANK CANDIDATE %d/%d] index=%d score=%.3f %s", rank + 1, len(ranked_all), i, scores[i], tag)
            logger.info(candidates[i])
        logger.info("")
        logger.info("*" * 50)
        logger.info("=" * 88)

        return selected
