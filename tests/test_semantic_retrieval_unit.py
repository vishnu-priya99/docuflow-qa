"""Unit tests for merging adjacent, overlapping selected chunks back into
one continuous passage before they reach the LLM - see
app.services.retrieval.semantic_retrieval._merge_adjacent_overlapping_chunks."""
from __future__ import annotations

from app.services.retrieval.semantic_retrieval import (
    _find_merge_point,
    _merge_adjacent_overlapping_chunks,
    format_chunks_for_llm,
)


def _result(*, document_id: str, chunk_index: int, text: str, score: float = 0.5, **extra):
    payload = {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": text,
        "filename": "doc.pdf",
        "file_type": "pdf",
        "chunk_id": f"{document_id}-{chunk_index}",
        **extra,
    }
    return {"score": score, "payload": payload}


def test_find_merge_point_locates_a_genuine_overlap():
    a = "the quick brown fox jumps over the lazy dog while it is raining heavily outside today"
    b = "the lazy dog while it is raining heavily outside today and the cat watches from the porch"
    point = _find_merge_point(a, b)
    assert point is not None
    assert b[point:] == " and the cat watches from the porch"


def test_find_merge_point_returns_none_below_minimum_length():
    assert _find_merge_point("...ends with the", "the start of something else") is None


def test_find_merge_point_returns_none_with_no_overlap_at_all():
    assert _find_merge_point("completely unrelated leading text here", "totally different trailing content") is None


def test_merges_two_adjacent_overlapping_chunks_from_the_same_document():
    a_text = "Corrective Actions:\n1. Reverted to the qualified supplier oven temperature setpoint for all lots."
    b_text = "1. Reverted to the qualified supplier oven temperature setpoint for all lots.\n2. Retrained operators."
    results = [
        _result(document_id="doc-1", chunk_index=6, text=a_text, section="3.0", page_start=2, page_end=2),
        _result(document_id="doc-1", chunk_index=7, text=b_text, section="3.0", page_start=2, page_end=2),
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 1
    assert merged[0]["payload"]["text"] == a_text + "\n2. Retrained operators."


def test_does_not_merge_non_adjacent_chunk_indices():
    a_text = "the quick brown fox jumps over the lazy dog while it is raining heavily outside"
    b_text = "the lazy dog while it is raining heavily outside and the cat watches from the porch"
    results = [
        _result(document_id="doc-1", chunk_index=1, text=a_text),
        _result(document_id="doc-1", chunk_index=9, text=b_text),  # not chunk_index + 1
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 2


def test_does_not_merge_chunks_from_different_documents():
    a_text = "the quick brown fox jumps over the lazy dog while it is raining heavily outside"
    b_text = "the lazy dog while it is raining heavily outside and the cat watches from the porch"
    results = [
        _result(document_id="doc-1", chunk_index=0, text=a_text),
        _result(document_id="doc-2", chunk_index=1, text=b_text),
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 2


def test_does_not_merge_adjacent_chunks_with_no_real_text_overlap():
    results = [
        _result(document_id="doc-1", chunk_index=0, text="Section on complaints and their root causes here."),
        _result(document_id="doc-1", chunk_index=1, text="A completely different paragraph about approvals."),
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 2


def test_merges_through_a_prepended_table_header_row():
    """A table split across chunks gets its header row repeated ahead of
    the real overlap (see chunker.build_chunks_with_sizes) - the merge
    search window must look past that repeated header, not just at
    b_text's literal first characters."""
    a_text = "Lot Number | Complaints | Status\nCVC-0298 | 1 | Unrelated\nCVC-0305 | 2 | Unrelated"
    b_text = "Lot Number | Complaints | Status\nCVC-0305 | 2 | Unrelated\nCVC-0311 | 5 | Root Cause"
    results = [
        _result(document_id="doc-1", chunk_index=3, text=a_text),
        _result(document_id="doc-1", chunk_index=4, text=b_text),
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 1
    assert merged[0]["payload"]["text"] == a_text + "\nCVC-0311 | 5 | Root Cause"


def test_three_way_chain_merges_into_one_block():
    a = "one two three four five six seven eight nine ten eleven twelve thirteen"
    b = "seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
    c = "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one"
    results = [
        _result(document_id="doc-1", chunk_index=0, text=a),
        _result(document_id="doc-1", chunk_index=1, text=b),
        _result(document_id="doc-1", chunk_index=2, text=c),
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    assert len(merged) == 1
    assert merged[0]["payload"]["text"] == (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
        " fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one"
    )


def test_merges_correctly_when_reranked_order_differs_from_chunk_index_order():
    """Reranking sorts by relevance, not document order - the higher-index
    half of an overlapping pair can legitimately outrank (and so appear
    before, in `results`) the lower-index half. The merge must still
    happen based on chunk_index adjacency, not `results` position."""
    a_text = "Corrective Actions:\n1. Reverted to the qualified supplier oven temperature setpoint for all lots."
    b_text = "1. Reverted to the qualified supplier oven temperature setpoint for all lots.\n2. Retrained operators."
    results = [
        _result(document_id="doc-1", chunk_index=7, text=b_text, score=0.9),  # ranked first, higher chunk_index
        _result(document_id="doc-1", chunk_index=9, text="unrelated risk assessment paragraph text here", score=0.7),
        _result(document_id="doc-1", chunk_index=6, text=a_text, score=0.5),  # ranked last, lower chunk_index
    ]
    merged = _merge_adjacent_overlapping_chunks(results)
    texts = [m["payload"]["text"] for m in merged]
    assert a_text + "\n2. Retrained operators." in texts
    assert len(merged) == 2
    # The merged block keeps the best (first) rank among its members - here that's index 0 (score 0.9).
    assert merged[0]["payload"]["text"] == a_text + "\n2. Retrained operators."


def test_format_chunks_for_llm_emits_one_source_block_for_a_merged_pair():
    a_text = "Corrective Actions:\n1. Reverted the oven temperature setpoint for all affected lots produced."
    b_text = "1. Reverted the oven temperature setpoint for all affected lots produced.\n2. Retrained operators."
    results = [
        _result(document_id="doc-1", chunk_index=6, text=a_text, section="3.0", page_start=2, page_end=2),
        _result(document_id="doc-1", chunk_index=7, text=b_text, section="3.0", page_start=2, page_end=2),
    ]
    context, sources = format_chunks_for_llm(results)
    assert context.count("[Source:") == 1
    assert "2. Retrained operators." in context
    assert len(sources) == 1
