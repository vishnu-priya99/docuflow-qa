"""Qdrant session/user filtering - verified directly against QdrantService,
independent of the HTTP layer."""
from __future__ import annotations

import uuid

import pytest

from app.services.embeddings.factory import get_embedding_provider
from app.vector.qdrant_client import get_qdrant_service

pytestmark = pytest.mark.asyncio


async def test_search_never_crosses_session_boundary():
    qdrant = get_qdrant_service()
    embedder = get_embedding_provider()

    user_a, session_a = f"user_{uuid.uuid4().hex[:6]}", uuid.uuid4().hex
    user_b, session_b = f"user_{uuid.uuid4().hex[:6]}", uuid.uuid4().hex

    vec_a = await embedder.embed_one("apples and oranges are fruit")
    vec_b = await embedder.embed_one("apples and oranges are fruit")

    await qdrant.upsert(
        [
            (
                str(uuid.uuid4()),
                vec_a,
                {
                    "user_id": user_a,
                    "session_id": session_a,
                    "file_id": "f1",
                    "filename": "a.txt",
                    "file_type": "txt",
                    "document_id": "d1",
                    "chunk_id": "c1",
                    "chunk_index": 0,
                    "text": "fruit content for session A",
                    "content_type": "document_chunk",
                },
            )
        ]
    )
    await qdrant.upsert(
        [
            (
                str(uuid.uuid4()),
                vec_b,
                {
                    "user_id": user_b,
                    "session_id": session_b,
                    "file_id": "f2",
                    "filename": "b.txt",
                    "file_type": "txt",
                    "document_id": "d2",
                    "chunk_id": "c2",
                    "chunk_index": 0,
                    "text": "fruit content for session B",
                    "content_type": "document_chunk",
                },
            )
        ]
    )

    query_vec = await embedder.embed_one("apples and oranges are fruit")

    results_a = await qdrant.search(
        query_vector=query_vec, user_id=user_a, session_id=session_a, content_type="document_chunk"
    )
    assert len(results_a) == 1
    assert results_a[0]["payload"]["session_id"] == session_a

    results_b = await qdrant.search(
        query_vector=query_vec, user_id=user_b, session_id=session_b, content_type="document_chunk"
    )
    assert len(results_b) == 1
    assert results_b[0]["payload"]["session_id"] == session_b

    # Right user, wrong session -> nothing.
    cross = await qdrant.search(
        query_vector=query_vec, user_id=user_a, session_id=session_b, content_type="document_chunk"
    )
    assert cross == []

    # delete_session only removes the targeted session's points.
    await qdrant.delete_session(user_id=user_a, session_id=session_a)
    results_a_after = await qdrant.search(
        query_vector=query_vec, user_id=user_a, session_id=session_a, content_type="document_chunk"
    )
    assert results_a_after == []
    results_b_after = await qdrant.search(
        query_vector=query_vec, user_id=user_b, session_id=session_b, content_type="document_chunk"
    )
    assert len(results_b_after) == 1
