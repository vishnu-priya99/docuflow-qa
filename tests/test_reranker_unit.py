"""Unit tests for the reranking layer: response parsing, the mock
provider's rerank heuristic, and the graceful-fallback contract."""
from __future__ import annotations

import pytest

from app.services.llm.mock_llm import MockLLMProvider
from app.services.llm.utils import parse_index_list
from app.services.retrieval.reranker import rerank_results


def test_parse_index_list_happy_path():
    assert parse_index_list("[3, 0, 5]", count=6) == [3, 0, 5]


def test_parse_index_list_ignores_out_of_range_and_duplicates():
    assert parse_index_list("[3, 99, 0, 3, -1]", count=4) == [3, 0]


def test_parse_index_list_handles_code_fence_and_prose():
    raw = "Here are the results:\n```json\n[2, 1]\n```"
    assert parse_index_list(raw, count=3) == [2, 1]


def test_parse_index_list_empty_array_is_a_legitimate_empty_result():
    assert parse_index_list("[]", count=5) == []


@pytest.mark.parametrize("garbage", ["not json at all", "", "null", "[not, valid, json]"])
def test_parse_index_list_never_raises_on_malformed_input(garbage):
    assert parse_index_list(garbage, count=5) == []


def test_parse_index_list_extracts_array_embedded_in_an_object():
    # Forgiving of a model that wraps the array instead of returning it bare.
    assert parse_index_list('{"indices": [1, 2]}', count=5) == [1, 2]


async def test_mock_rerank_prefers_explicit_override():
    llm = MockLLMProvider(rerank_answers={"q": [2, 0]})
    result = await llm.rerank(question="q", candidates=["a", "b", "c"], top_n=5)
    assert result == [2, 0]


async def test_mock_rerank_default_heuristic_ranks_by_word_overlap():
    llm = MockLLMProvider()
    result = await llm.rerank(
        question="what caused the coating friction issue",
        candidates=[
            "Quarterly revenue grew significantly this year.",  # index 0: irrelevant
            "The coating friction issue was caused by a supplier change.",  # index 1: relevant
            "Employees get 15 vacation days per year.",  # index 2: irrelevant
        ],
        top_n=5,
    )
    assert result[0] == 1
    assert 0 not in result and 2 not in result


class _FakeLLMThatFails:
    async def rerank(self, *, question, candidates, top_n):
        raise RuntimeError("simulated provider outage")


class _FakeLLMThatFindsNothingRelevant:
    async def rerank(self, *, question, candidates, top_n):
        return []


async def test_rerank_results_falls_back_to_original_order_on_failure():
    results = [{"score": 0.9, "payload": {"text": "a"}}, {"score": 0.8, "payload": {"text": "b"}}]
    out = await rerank_results(_FakeLLMThatFails(), question="q", results=results, top_n=5)
    assert out == results  # unchanged, not silently emptied


async def test_rerank_results_respects_a_genuine_empty_verdict():
    """A reranker legitimately deciding nothing is relevant must produce an
    empty result - NOT fall back to showing the raw candidates anyway,
    which would defeat the entire point of reranking."""
    results = [{"score": 0.9, "payload": {"text": "a"}}, {"score": 0.8, "payload": {"text": "b"}}]
    out = await rerank_results(
        _FakeLLMThatFindsNothingRelevant(), question="q", results=results, top_n=5
    )
    assert out == []


async def test_rerank_results_reorders_by_returned_indices():
    llm = MockLLMProvider(rerank_answers={"which one": [1, 0]})
    results = [{"score": 0.5, "payload": {"text": "first"}}, {"score": 0.9, "payload": {"text": "second"}}]
    out = await rerank_results(llm, question="which one", results=results, top_n=5)
    assert [r["payload"]["text"] for r in out] == ["second", "first"]
