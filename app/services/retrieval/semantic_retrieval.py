"""Semantic (Qdrant) retrieval for SEMANTIC and HYBRID questions.

Every search is filtered by user_id + session_id (see QdrantService.search),
so results can never cross session/user boundaries. Retrieved payload
metadata is passed straight through to the LLM as citation evidence - the
LLM never invents source locations.
"""
from __future__ import annotations

from typing import Any

from app.services.embeddings.base import EmbeddingProvider
from app.vector.qdrant_client import QdrantService


async def retrieve_chunks(
    *,
    qdrant: QdrantService,
    embedder: EmbeddingProvider,
    question: str,
    user_id: str,
    session_id: str,
    file_id: str | None = None,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    vector = await embedder.embed_one(question)
    return await qdrant.search(
        query_vector=vector,
        user_id=user_id,
        session_id=session_id,
        content_type="document_chunk",
        file_id=file_id,
        top_k=top_k,
    )


# Below this many characters, a text match between two chunks is treated
# as coincidental rather than the chunker's own overlap (see
# app.services.ingestion.chunker._split_text) - real overlap is
# CHUNK_OVERLAP_CHARS-scale (config default 200), not a handful of shared
# words.
_MIN_OVERLAP_CHARS = 20
# How far into the next chunk to search for the previous chunk's tail.
# Wider than CHUNK_OVERLAP_CHARS alone so a repeated table header row
# prepended ahead of the overlap (see chunker.build_chunks_with_sizes)
# doesn't push the real overlap out of the search window.
_OVERLAP_SEARCH_WINDOW = 400


def _find_merge_point(a_text: str, b_text: str) -> int | None:
    """If a_text's tail is also present near the start of b_text - the
    chunker's own overlap between two adjacent pieces of the same parsed
    unit - returns the index into b_text where b's genuinely new content
    begins. None if no sufficiently long, confident match is found (the
    two chunks are then left unmerged rather than guessed at). Tries the
    longest possible overlap first so a short coincidental match can't
    win over a real one."""
    search_zone = b_text[:_OVERLAP_SEARCH_WINDOW]
    max_len = min(len(a_text), len(search_zone))
    for length in range(max_len, _MIN_OVERLAP_CHARS - 1, -1):
        pos = search_zone.find(a_text[-length:])
        if pos != -1:
            return pos + length
    return None


def _merge_adjacent_overlapping_chunks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merges selected chunks that are consecutive pieces of the same
    parsed unit (same document, consecutive chunk_index, and share the
    chunker's own text overlap) into one combined chunk.

    Two such chunks presented to the LLM as separate "[Source: ...]"
    blocks both contain the sentence at their shared boundary - which
    reads as two sources repeating the same fact, not one continuous
    passage split in two. Verified: a local model can (and did) treat the
    second chunk's genuinely new tail content as redundant and drop it
    entirely, truncating a list at the split point - and rewording the
    prompt's instructions did not fix it. Removing the ambiguity here,
    deterministically, before the LLM ever sees it, does not depend on
    the model correctly inferring what two identically-labeled sources
    from the same page/section actually mean."""
    # Grouped and walked in chunk_index order (not `results`' rerank-rank
    # order) so a lower-indexed chunk that happens to rerank below a
    # higher-indexed one still gets the chance to absorb it forward -
    # walking in rerank order let an already-visited later chunk block a
    # merge that an earlier chunk (visited afterwards) should have made.
    by_doc: dict[Any, list[int]] = {}
    for i, r in enumerate(results):
        by_doc.setdefault(r["payload"].get("document_id"), []).append(i)
    for idxs in by_doc.values():
        idxs.sort(key=lambda i: results[i]["payload"].get("chunk_index", -1))

    groups: list[dict[str, Any]] = []
    for idxs in by_doc.values():
        pos = 0
        while pos < len(idxs):
            i = idxs[pos]
            payload = dict(results[i]["payload"])
            score = results[i]["score"]
            best_rank = i  # lowest original (rerank-order) index absorbed, for final output ordering
            pos += 1
            while pos < len(idxs):
                j = idxs[pos]
                next_payload = results[j]["payload"]
                if next_payload.get("chunk_index") != payload.get("chunk_index", -999) + 1:
                    break
                merge_point = _find_merge_point(payload["text"], next_payload["text"])
                if merge_point is None:
                    break
                payload = {
                    **payload,
                    "text": payload["text"] + next_payload["text"][merge_point:],
                    "chunk_index": next_payload.get("chunk_index"),
                    "page_end": next_payload.get("page_end", payload.get("page_end")),
                }
                score = max(score, results[j]["score"])
                best_rank = min(best_rank, j)
                pos += 1
            groups.append({"payload": payload, "score": score, "_rank": best_rank})

    groups.sort(key=lambda g: g["_rank"])
    for g in groups:
        del g["_rank"]
    return groups


def _location_str(payload: dict[str, Any]) -> str:
    parts = []
    if payload.get("page_start") is not None:
        if payload.get("page_end") and payload["page_end"] != payload["page_start"]:
            parts.append(f"page {payload['page_start']}-{payload['page_end']}")
        else:
            parts.append(f"page {payload['page_start']}")
    if payload.get("slide_number") is not None:
        parts.append(f"slide {payload['slide_number']}")
    if payload.get("line_start") is not None:
        parts.append(f"line {payload['line_start']}")
    if payload.get("section"):
        parts.append(f"section: {payload['section']}")
    return f" ({', '.join(parts)})" if parts else ""


def format_chunks_for_llm(results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (context_text_for_llm, source_list_for_api_response)."""
    results = _merge_adjacent_overlapping_chunks(results)
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for r in results:
        payload = r["payload"]
        loc = _location_str(payload)
        context_parts.append(f"[Source: {payload['filename']}{loc}]\n{payload['text']}")
        sources.append(
            {
                "filename": payload.get("filename"),
                "file_type": payload.get("file_type"),
                "chunk_id": payload.get("chunk_id"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "section": payload.get("section"),
                "slide_number": payload.get("slide_number"),
                "slide_title": payload.get("slide_title"),
                "line_start": payload.get("line_start"),
                "line_end": payload.get("line_end"),
                "score": r.get("score"),
            }
        )
    return "\n\n".join(context_parts), sources
