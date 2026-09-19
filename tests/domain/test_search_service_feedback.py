import logging
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.data.repositories.feedback_repository import FeedbackRepository, FeedbackVotes
from app.domain.search_service import SemanticSearchService


class _Events(list):
    """Shared, ordered record of what the fakes were asked to do."""


class _FakeEmbeddings:
    def __init__(self, events):
        self._events = events

    def embed_query(self, _query):
        self._events.append("embed")
        return [1.0, 0.0]


class _FakeCatalog:
    def __init__(self, events, rows):
        self._events = events
        self._rows = rows
        self.calls = []

    def search_by_embedding(self, *args, **kwargs):
        self._events.append("catalog")
        self.calls.append((args, kwargs))
        return list(self._rows)


class _FakeFeedback:
    def __init__(self, events, votes=None, future=None):
        self._events = events
        self._votes = votes or {}
        self._future = future
        self.lookups = []

    def get_votes_async(self, query, language=None):
        self._events.append("feedback")
        self.lookups.append((query, language))
        if self._future is not None:
            return self._future
        done = Future()
        done.set_result(self._votes)
        return done


def _rows():
    return [
        {"isbn13": "1", "title": "A", "authors": "x", "description": "d", "similarity": 0.90},
        {"isbn13": "2", "title": "B", "authors": "x", "description": "d", "similarity": 0.89},
    ]


def _service(votes=None, future=None):
    events = _Events()
    catalog = _FakeCatalog(events, _rows())
    feedback = _FakeFeedback(events, votes, future)
    service = SemanticSearchService(
        embeddings=_FakeEmbeddings(events),
        google_client=SimpleNamespace(),
        query_rewriter=SimpleNamespace(),
        query_cache=SimpleNamespace(),
        catalog=catalog,
        feedback=feedback,
    )
    return service, catalog, feedback, events


@pytest.fixture(autouse=True)
def _feedback_settings(monkeypatch):
    monkeypatch.setattr(settings, "feedback_weight", 0.05)
    monkeypatch.setattr(settings, "feedback_prior_strength", 5.0)
    monkeypatch.setattr(settings, "feedback_min_votes", 3)


def test_flag_off_never_touches_feedback(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", False)
    service, catalog, feedback, events = _service({"2": FeedbackVotes(up=9, down=0)})
    service.search("detective novel")
    assert feedback.lookups == []
    assert "feedback" not in events
    assert catalog.calls[0][1]["score_adjustments"] is None


def test_flag_on_passes_vote_adjustments_to_the_catalog(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, catalog, _, _ = _service(
        {"2": FeedbackVotes(up=9, down=0), "1": FeedbackVotes(up=0, down=9)}
    )
    service.search("detective novel")
    adjustments = catalog.calls[0][1]["score_adjustments"]
    assert adjustments["2"] > 0 > adjustments["1"]


def test_lookup_uses_the_original_query_and_language(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, _, feedback, _ = _service({})
    service.search("  Detective Novel? ", language="tr")
    assert feedback.lookups == [("  Detective Novel? ", "tr")]


def test_lookup_starts_before_the_embedding_so_the_two_overlap(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, _, _, events = _service({})
    service.search("detective novel")
    assert events == ["feedback", "embed", "catalog"]


def test_votes_below_the_minimum_send_no_adjustments(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, catalog, _, _ = _service({"1": FeedbackVotes(up=2, down=0)})
    service.search("detective novel")
    assert catalog.calls[0][1]["score_adjustments"] is None


def test_a_slow_lookup_is_abandoned_and_the_search_still_returns(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    monkeypatch.setattr(settings, "feedback_lookup_timeout_seconds", 0.05)
    never_done = Future()  # simulates a Supabase call that does not answer in time
    service, catalog, _, _ = _service(future=never_done)

    results, _ = service.search("detective novel")

    assert [r.isbn13 for r in results] == ["1", "2"]
    assert catalog.calls[0][1]["score_adjustments"] is None


def test_a_failed_lookup_does_not_break_the_search(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    failed = Future()
    failed.set_exception(RuntimeError("supabase exploded"))
    service, catalog, _, _ = _service(future=failed)

    results, _ = service.search("detective novel")

    assert len(results) == 2
    assert catalog.calls[0][1]["score_adjustments"] is None


def test_applied_adjustments_are_logged_without_the_query_text(monkeypatch, caplog):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, _, _, _ = _service({"2": FeedbackVotes(up=9, down=0)})
    with caplog.at_level(logging.INFO):
        service.search("confidential detective novel")
    assert "feedback_applied adjusted_books=1" in caplog.text
    assert "confidential" not in caplog.text


def test_empty_queries_start_no_lookup(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    service, _, feedback, _ = _service({})
    assert service.search("   ") == ([], None)
    assert feedback.lookups == []


def test_end_to_end_with_the_real_repository_reorders_near_ties(monkeypatch):
    """Real FeedbackRepository + a stub Supabase client, fake catalog that honours adjustments."""
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)

    class _Client:
        def rpc(self, name, params):
            assert name == "get_semantic_feedback"
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(data=[{"isbn13": "2", "up": 20, "down": 0}])
            )

    events = _Events()

    class _AdjustingCatalog:
        def search_by_embedding(self, *args, score_adjustments=None, **kwargs):
            rows = _rows()
            adjustments = score_adjustments or {}
            return sorted(
                rows,
                key=lambda r: r["similarity"] + adjustments.get(r["isbn13"], 0.0),
                reverse=True,
            )

    service = SemanticSearchService(
        embeddings=_FakeEmbeddings(events),
        google_client=SimpleNamespace(),
        query_rewriter=SimpleNamespace(),
        query_cache=SimpleNamespace(),
        catalog=_AdjustingCatalog(),
        feedback=FeedbackRepository(client=_Client()),
    )
    results, _ = service.search("detective novel")
    assert [r.isbn13 for r in results] == ["2", "1"]
    assert results[0].similarity == 0.89  # reported similarity is still the raw score


def test_search_thread_is_not_blocked_by_a_pending_lookup_beyond_the_timeout(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    monkeypatch.setattr(settings, "feedback_lookup_timeout_seconds", 0.05)
    release = threading.Event()

    class _SlowClient:
        def rpc(self, *_args):
            def execute():
                release.wait(timeout=5)
                return SimpleNamespace(data=[])

            return SimpleNamespace(execute=execute)

    events = _Events()
    service = SemanticSearchService(
        embeddings=_FakeEmbeddings(events),
        google_client=SimpleNamespace(),
        query_rewriter=SimpleNamespace(),
        query_cache=SimpleNamespace(),
        catalog=_FakeCatalog(events, _rows()),
        feedback=FeedbackRepository(client=_SlowClient()),
    )
    import time

    started = time.monotonic()
    results, _ = service.search("detective novel")
    elapsed = time.monotonic() - started
    release.set()

    assert len(results) == 2
    assert elapsed < 1.0
