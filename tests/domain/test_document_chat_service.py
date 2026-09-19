import pytest

from app.core.config import settings
from app.domain.document_chat_service import DocumentChatService
from app.models.document_session import DocumentChunk


def _service() -> DocumentChatService:
    # Bypasses __init__ so the test never constructs real Gemini/LLM clients
    # (those need a live GOOGLE_API_KEY) — _retrieve_chunks/_build_context/
    # _cosine_similarity only touch their arguments and module-level settings.
    return DocumentChatService.__new__(DocumentChatService)


def _chunk(index: int, embedding: list[float], content: str = "content") -> DocumentChunk:
    return DocumentChunk(index=index, content=content, embedding=embedding, metadata={})


def test_cosine_similarity_identical_vectors_is_one():
    assert DocumentChatService._cosine_similarity([2.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert DocumentChatService._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vector_without_dividing_by_zero():
    assert DocumentChatService._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_retrieve_chunks_filters_out_results_below_the_relevance_floor(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.5)
    monkeypatch.setattr(settings, "document_retrieval_top_k", 5)
    service = _service()
    chunks = [
        _chunk(0, [1.0, 0.0]),  # identical to query -> similarity 1.0
        _chunk(1, [0.0, 1.0]),  # orthogonal to query -> similarity 0.0
    ]
    retrieved = service._retrieve_chunks(chunks, [1.0, 0.0])
    assert [c.index for c in retrieved] == [0]


def test_retrieve_chunks_returns_nothing_when_all_scores_are_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.9)
    service = _service()
    chunks = [_chunk(0, [0.0, 1.0])]
    retrieved = service._retrieve_chunks(chunks, [1.0, 0.0])
    assert retrieved == []


def test_retrieve_chunks_still_respects_top_k_among_relevant_matches(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.0)
    monkeypatch.setattr(settings, "document_retrieval_top_k", 2)
    service = _service()
    chunks = [_chunk(i, [1.0, 0.0]) for i in range(5)]
    retrieved = service._retrieve_chunks(chunks, [1.0, 0.0])
    assert len(retrieved) == 2


def test_build_context_returns_explicit_no_match_marker_when_nothing_retrieved():
    service = _service()
    context = service._build_context([])
    assert "No relevant excerpts" in context


def test_build_context_joins_chunk_content_with_blank_line():
    service = _service()
    chunks = [_chunk(0, [], content="First part."), _chunk(1, [], content="Second part.")]
    context = service._build_context(chunks)
    assert context == "First part.\n\nSecond part."


def test_build_context_skips_blank_chunks():
    service = _service()
    chunks = [_chunk(0, [], content="   "), _chunk(1, [], content="Real content.")]
    context = service._build_context(chunks)
    assert context == "Real content."


def test_build_context_truncates_to_the_configured_character_budget(monkeypatch):
    monkeypatch.setattr(settings, "document_max_context_chars", 10)
    service = _service()
    chunks = [_chunk(0, [], content="A" * 20)]
    context = service._build_context(chunks)
    assert len(context) == 10


def _chunk_at(index, content, metadata=None, embedding=None):
    return DocumentChunk(
        index=index,
        content=content,
        embedding=embedding if embedding is not None else [],
        metadata=metadata or {},
    )


def test_chunk_label_uses_page_for_pdf_and_section_for_epub():
    service = _service()
    assert service._chunk_label(_chunk_at(0, "x", {"page": 12})) == "[Page 12]"
    assert service._chunk_label(_chunk_at(0, "x", {"chapter_index": 3})) == "[Section 3]"
    assert (
        service._chunk_label(_chunk_at(0, "x", {"chapter_index": 3, "chapter_title": "Storm Warning"}))
        == "[Section 3: Storm Warning]"
    )
    assert (
        service._chunk_label(_chunk_at(0, "x", {"chapter_index": 3, "chapter_title": "ch03"}))
        == "[Section 3]"
    )
    assert service._chunk_label(_chunk_at(0, "x", {})) == ""


def test_build_context_prefixes_each_excerpt_with_its_location_label():
    service = _service()
    context = service._build_context([_chunk_at(0, "Hello.", {"page": 2})])
    assert context == "[Page 2]\nHello."


def test_select_context_chunks_skips_a_chunk_that_does_not_fit_but_keeps_smaller_ones(monkeypatch):
    monkeypatch.setattr(settings, "document_max_context_chars", 30)
    service = _service()
    ranked = [
        _chunk_at(0, "A" * 20),   # fits
        _chunk_at(1, "B" * 20),   # would exceed the budget -> skipped whole, not cut
        _chunk_at(2, "C" * 5),    # smaller chunk further down still fits
    ]
    selected = service._select_context_chunks(ranked)
    assert [c.index for c in selected] == [0, 2]


def test_select_context_chunks_returns_document_order_not_rank_order():
    service = _service()
    ranked = [_chunk_at(7, "late"), _chunk_at(2, "early"), _chunk_at(5, "middle")]
    assert [c.index for c in service._select_context_chunks(ranked)] == [2, 5, 7]


def test_select_context_chunks_always_keeps_the_first_chunk_even_if_oversized(monkeypatch):
    monkeypatch.setattr(settings, "document_max_context_chars", 10)
    service = _service()
    selected = service._select_context_chunks([_chunk_at(0, "A" * 50)])
    assert [c.index for c in selected] == [0]


def test_retrieve_chunks_uses_mmr_to_avoid_near_duplicates(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.0)
    monkeypatch.setattr(settings, "document_mmr_lambda", 0.5)
    service = _service()
    chunks = [
        _chunk_at(0, "a", embedding=[1.0, 0.0]),
        _chunk_at(1, "a-dup", embedding=[0.99, -0.02]),
        _chunk_at(2, "b", embedding=[0.0, 1.0]),
    ]
    retrieved = service._retrieve_chunks(chunks, [0.8, 0.6], top_k=2)
    assert [c.index for c in retrieved] == [0, 2]


def test_retrieve_chunks_lambda_one_disables_mmr(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.0)
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    service = _service()
    chunks = [
        _chunk_at(0, "a", embedding=[1.0, 0.0]),
        _chunk_at(1, "a-dup", embedding=[0.99, -0.02]),
        _chunk_at(2, "b", embedding=[0.0, 1.0]),
    ]
    retrieved = service._retrieve_chunks(chunks, [0.8, 0.6], top_k=2)
    assert [c.index for c in retrieved] == [0, 1]


def test_retrieve_chunks_explicit_top_k_overrides_the_default(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.0)
    monkeypatch.setattr(settings, "document_retrieval_top_k", 1)
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    service = _service()
    chunks = [_chunk_at(i, "x", embedding=[1.0, 0.0]) for i in range(6)]
    assert len(service._retrieve_chunks(chunks, [1.0, 0.0], top_k=4)) == 4
