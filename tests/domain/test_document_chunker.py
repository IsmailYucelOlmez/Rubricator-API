from app.domain.document_chunker import DocumentChunker, _even_sample


class _FakeSegment:
    def __init__(self, content: str, metadata: dict) -> None:
        self.content = content
        self.metadata = metadata


def test_even_sample_returns_all_when_under_limit():
    items = [(f"chunk-{i}", {}) for i in range(5)]
    assert _even_sample(items, limit=10) == items


def test_even_sample_keeps_first_and_last():
    items = [(f"chunk-{i}", {}) for i in range(100)]
    sampled = _even_sample(items, limit=10)
    assert len(sampled) == 10
    assert sampled[0] == items[0]
    assert sampled[-1] == items[-1]


def test_even_sample_single_limit_picks_middle():
    items = [(f"chunk-{i}", {}) for i in range(10)]
    assert _even_sample(items, limit=1) == [items[5]]


def test_even_sample_non_positive_limit_returns_input_unchanged():
    items = [(f"chunk-{i}", {}) for i in range(5)]
    assert _even_sample(items, limit=0) is items


def test_even_sample_preserves_ascending_order_across_the_whole_range():
    # This is the property the sampler exists for: coverage across the
    # entire document, not just the opening pages.
    items = [(f"chunk-{i}", {}) for i in range(37)]
    sampled = _even_sample(items, limit=13)
    assert len(sampled) == 13
    original_indices = [items.index(entry) for entry in sampled]
    assert original_indices == sorted(original_indices)
    assert len(set(original_indices)) == 13


def test_split_segments_tags_each_piece_with_its_source_metadata():
    chunker = DocumentChunker()
    segments = [
        _FakeSegment("A" * 50, {"page": 1}),
        _FakeSegment("B" * 50, {"page": 2}),
    ]
    result = chunker.split_segments(segments)
    assert not result.truncated
    assert {meta["page"] for _, meta in result.items} == {1, 2}


def test_split_segments_skips_blank_pieces():
    chunker = DocumentChunker()
    segments = [_FakeSegment("   ", {"page": 1}), _FakeSegment("Real content.", {"page": 2})]
    result = chunker.split_segments(segments)
    assert len(result.items) == 1
    assert result.items[0][0] == "Real content."


def test_split_segments_truncates_and_flags_when_over_document_max_chunks(monkeypatch):
    import app.domain.document_chunker as chunker_module

    monkeypatch.setattr(chunker_module.settings, "document_max_chunks", 3)
    chunker = DocumentChunker()
    segments = [_FakeSegment(f"segment number {i}", {"index": i}) for i in range(10)]
    result = chunker.split_segments(segments)
    assert result.truncated is True
    assert len(result.items) == 3


def test_build_chunks_pairs_items_with_embeddings_by_position():
    chunker = DocumentChunker()
    items = [("text-a", {"page": 1}), ("text-b", {"page": 2})]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    chunks = chunker.build_chunks(items, embeddings)
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].content == "text-a"
    assert chunks[0].embedding == [0.1, 0.2]
    assert chunks[1].metadata == {"page": 2}


def test_build_chunks_truncates_to_the_shorter_of_items_or_embeddings():
    chunker = DocumentChunker()
    items = [("a", {}), ("b", {}), ("c", {})]
    embeddings = [[0.0], [0.1]]
    chunks = chunker.build_chunks(items, embeddings)
    assert len(chunks) == 2


def test_meaningful_chapter_title_rejects_generated_file_names():
    from app.domain.document_chunker import meaningful_chapter_title

    for generated in ("ch03", "chapter_12", "Section0005", "split_005", "index_split_012", "part 3", "", None):
        assert meaningful_chapter_title(generated) is None


def test_meaningful_chapter_title_keeps_real_titles():
    from app.domain.document_chunker import meaningful_chapter_title

    assert meaningful_chapter_title("The Gathering Storm") == "The Gathering Storm"
    assert meaningful_chapter_title("Yolculuk Başlıyor") == "Yolculuk Başlıyor"


def test_embedding_text_prefixes_meaningful_titles_only():
    assert (
        DocumentChunker.embedding_text("Body text.", {"chapter_title": "The Gathering Storm"})
        == "The Gathering Storm\n\nBody text."
    )
    assert DocumentChunker.embedding_text("Body text.", {"chapter_title": "ch03"}) == "Body text."
    assert DocumentChunker.embedding_text("Body text.", {"page": 4}) == "Body text."
