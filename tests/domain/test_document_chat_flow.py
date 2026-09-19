from datetime import datetime, timezone
from types import SimpleNamespace

from app.core.config import settings
from app.data.session_store.memory import InMemorySessionStore
from app.domain.document_chat_service import DocumentChatService
from app.models.document_session import ChatTurn, DocumentChunk, DocumentSession


class _FakeEmbeddings:
    def __init__(self, vector):
        self._vector = vector

    def embed_query(self, _text):
        return self._vector


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content="answer")


def _session(chunks, with_history=False):
    now = datetime.now(timezone.utc)
    session = DocumentSession(
        session_id="s1",
        format="pdf",
        filename="book.pdf",
        created_at=now,
        expires_at=now.replace(year=now.year + 1),
        last_accessed_at=now,
        page_count=10,
        chapter_count=None,
        word_count=100,
        chunk_count=len(chunks),
        chunks=chunks,
        status="ready",
    )
    if with_history:
        session.chat_turns = [
            ChatTurn(role="user", content="Who is Anna?", created_at=now),
            ChatTurn(role="assistant", content="The protagonist.", created_at=now),
        ]
    return session


def _service(session, query_vector=(1.0, 0.0)):
    store = InMemorySessionStore()
    store.put(session)
    service = DocumentChatService.__new__(DocumentChatService)
    service._session_store = store
    service._embedding_client = _FakeEmbeddings(list(query_vector))
    service._chat_llm = _FakeLLM()
    return service


def _chunks(n):
    return [
        DocumentChunk(index=i, content=f"chunk {i} text", embedding=[1.0, 0.0], metadata={"page": i + 1})
        for i in range(n)
    ]


def test_self_contained_question_skips_the_condense_llm_call(monkeypatch):
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    service = _service(_session(_chunks(3), with_history=True))
    service.chat("s1", "What is the name of the ship Captain Ahab commands?")
    assert len(service._chat_llm.calls) == 1  # only the answer call


def test_follow_up_question_still_gets_condensed(monkeypatch):
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    service = _service(_session(_chunks(3), with_history=True))
    service.chat("s1", "why?")
    assert len(service._chat_llm.calls) == 2  # condense + answer


def test_complex_questions_retrieve_more_chunks_than_simple_ones(monkeypatch):
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    monkeypatch.setattr(settings, "document_retrieval_top_k", 2)
    monkeypatch.setattr(settings, "document_retrieval_top_k_complex", 6)

    simple = _service(_session(_chunks(10))).chat("s1", "Where does the story begin exactly?")
    complex_ = _service(_session(_chunks(10))).chat("s1", "Summarize the whole story for me")

    assert len(simple.sources) == 2
    assert len(complex_.sources) == 6


def test_no_relevant_chunks_means_no_sources_and_an_explicit_marker(monkeypatch):
    monkeypatch.setattr(settings, "document_retrieval_min_score", 0.5)
    service = _service(_session(_chunks(3)), query_vector=(0.0, 1.0))  # orthogonal to every chunk
    result = service.chat("s1", "Something the document never mentions at all?")
    assert result.sources == []
    prompt = service._chat_llm.calls[0][1].content
    assert "No relevant excerpts" in prompt


def test_sources_only_list_chunks_that_actually_fit_in_the_context(monkeypatch):
    monkeypatch.setattr(settings, "document_mmr_lambda", 1.0)
    monkeypatch.setattr(settings, "document_retrieval_top_k", 5)
    monkeypatch.setattr(settings, "document_max_context_chars", 60)
    long_chunks = [
        DocumentChunk(index=i, content="x" * 40, embedding=[1.0, 0.0], metadata={"page": i + 1})
        for i in range(5)
    ]
    service = _service(_session(long_chunks))
    result = service.chat("s1", "Where does the story begin exactly?")
    # 60 chars fit one 40-char excerpt (+ label), not a second one.
    assert len(result.sources) == 1

