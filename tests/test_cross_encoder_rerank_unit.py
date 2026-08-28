"""Tests for the cross-encoder reranker.

The ranking/filtering math is tested directly (fast, no model needed).
The real model is only exercised by test_cross_encoder_end_to_end_with_real_model,
which is skipped automatically if sentence-transformers isn't installed
(RERANK_PROVIDER defaults to "llm" precisely so it's optional) - the
model, once downloaded, is cached by huggingface_hub, so this doesn't
re-download on every run.
"""
from __future__ import annotations

import pytest

from app.services.reranking.cross_encoder_rerank import rank_and_filter


def test_rank_and_filter_orders_by_score_descending():
    scores = [0.1, 0.9, 0.5]
    assert rank_and_filter(scores, top_n=3, min_score=None) == [1, 2, 0]


def test_rank_and_filter_respects_top_n():
    scores = [0.1, 0.9, 0.5, 0.7]
    assert rank_and_filter(scores, top_n=2, min_score=None) == [1, 3]


def test_rank_and_filter_drops_below_min_score():
    scores = [4.13, -2.37, -10.5, -11.2]
    assert rank_and_filter(scores, top_n=5, min_score=0.0) == [0]


def test_rank_and_filter_min_score_can_yield_empty_result():
    scores = [-1.0, -2.0, -3.0]
    assert rank_and_filter(scores, top_n=5, min_score=0.0) == []


def test_rank_and_filter_no_min_score_always_fills_top_n():
    scores = [4.13, -2.37, -10.5, -11.2]
    assert rank_and_filter(scores, top_n=3, min_score=None) == [0, 1, 2]


def test_rank_and_filter_score_margin_isolates_clear_winner_negative_scale():
    # Real distribution captured from this project's own documents - top
    # score is negative, so an absolute floor like 0.0 would wrongly drop
    # every candidate here (the bug that motivated score_margin).
    scores = [-4.235, -11.379, -11.342, -11.297, -11.419, -11.215, -11.342, -11.165, -11.376]
    assert rank_and_filter(scores, top_n=5, min_score=None, score_margin=5.0) == [0]


def test_rank_and_filter_score_margin_isolates_clear_winner_positive_scale():
    # Same margin value, a completely different absolute scale (top score
    # positive) - margin adapts per-batch where a fixed floor can't.
    scores = [-11.289, 4.130, -11.260, -2.366, -10.585, -10.776]
    assert rank_and_filter(scores, top_n=5, min_score=None, score_margin=5.0) == [1]


def test_rank_and_filter_score_margin_wide_enough_admits_noise():
    scores = [-4.235, -11.379, -11.342, -11.297, -11.419, -11.215, -11.342, -11.165, -11.376]
    result = rank_and_filter(scores, top_n=5, min_score=None, score_margin=7.0)
    assert result == [0, 7, 5]


def test_rank_and_filter_score_margin_combines_with_min_score():
    # min_score is applied first, score_margin narrows further within the
    # survivors - a candidate can fail either filter independently.
    scores = [4.13, -2.37, -10.5, -11.2]
    assert rank_and_filter(scores, top_n=5, min_score=0.0, score_margin=1.0) == [0]


def test_rank_and_filter_score_margin_none_is_a_noop():
    scores = [0.1, 0.9, 0.5]
    assert rank_and_filter(scores, top_n=3, min_score=None, score_margin=None) == [1, 2, 0]


async def test_cross_encoder_end_to_end_with_real_model():
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    from app.services.reranking.cross_encoder_rerank import CrossEncoderRerankProvider

    provider = CrossEncoderRerankProvider(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", min_score=0.0
    )
    candidates = [
        "Employees get 15 vacation days per year.",
        "The coating friction issue was caused by a supplier change in the lumen coating process.",
        "Competitor A launched a coated CVC line in North America in Q2.",
    ]
    result = await provider.rerank(
        question="What was the root cause of the coating issue?", candidates=candidates, top_n=5
    )
    assert result == [1]
