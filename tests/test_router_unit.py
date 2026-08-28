"""Unit tests for question classification (SEMANTIC / STRUCTURED / HYBRID)."""
from __future__ import annotations

import pytest

from app.services.llm.mock_llm import MockLLMProvider

pytestmark = pytest.mark.asyncio


async def test_narrative_question_is_semantic():
    llm = MockLLMProvider()
    result = await llm.classify_question(
        question="What are the main recommendations in the report?",
        has_documents=True,
        has_structured_data=False,
    )
    assert result == "SEMANTIC"


async def test_aggregation_question_is_structured():
    llm = MockLLMProvider()
    result = await llm.classify_question(
        question="What is the average salary in IT?",
        has_documents=False,
        has_structured_data=True,
    )
    assert result == "STRUCTURED"


async def test_discovery_plus_aggregation_is_hybrid():
    llm = MockLLMProvider()
    result = await llm.classify_question(
        question="Find the laptop sales data and tell me the total revenue in March.",
        has_documents=True,
        has_structured_data=True,
    )
    assert result == "HYBRID"


async def test_never_routes_structured_without_structured_data():
    llm = MockLLMProvider()
    result = await llm.classify_question(
        question="How many unique names are there?", has_documents=True, has_structured_data=False
    )
    assert result != "STRUCTURED"


async def test_never_routes_semantic_without_documents():
    llm = MockLLMProvider()
    result = await llm.classify_question(
        question="Summarize the main findings.", has_documents=False, has_structured_data=True
    )
    assert result != "SEMANTIC"
