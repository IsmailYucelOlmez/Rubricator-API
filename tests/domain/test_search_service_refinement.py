import logging
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.data.repositories.feedback_repository import FeedbackVotes
from app.domain.search_service import SemanticSearchService

BASE = [1.0, 0.0, 0.0]
REL_VEC = [0.0, 1.0, 0.0]
IRR_VEC = [0.0, 0.0, 1.0]


class _Embeddings:
    def embed_query(self, _query):
        return list(BASE)


class _Catalog:
    supports_feedback_refinement = True

    def __init__(self, vectors=None, fail_vectors=False):
        self.vectors = vectors if vectors is not None else {"111": REL_VEC, "222": IRR_VEC}
        self.fail_vectors = fail_vectors
        self.vector_requests = []
        self.search_calls = []

    def get_vectors(self, isbns):
        self.vector_requests.append(list(isbns))
        return {} if self.fail_vectors else {i: v for i, v in self.vectors.items() if i in isbns}

    def search_by_embedding(self, *args, **kwargs):
        self.search_calls.append((args, kwargs))
        return [
            {"isbn13": "1", "title": "A", "authors": "x", "description": "d", "similarity": 0.9}
        ]


class _PlainCatalog:
    """A backend without refinement support (e.g. the Supabase catalog)."""

    def __init__(self):
        self.search_calls = []

    def search_by_embedding(self, *args, **kwargs):
        self.search_calls.append((args, kwargs))
        return []


def _service(catalog, feedback=None):
    return SemanticSearchService(
        embeddings=_Embeddings(),
        google_client=SimpleNamespace(),
        query_rewriter=SimpleNamespace(),
        query_cache=SimpleNamespace(),
        catalog=catalog,
        feedback=feedback or SimpleNamespace(),
    )


def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", False)
    monkeypatch.setattr(settings, "refine_relevant_weight", 0.5)
    monkeypatch.setattr(settings, "refine_irrelevant_weight", 0.3)


def test_no_marks_leaves_the_search_untouched():
    catalog = _Catalog()
    _service(catalog).search("detective novel")
    args, kwargs = catalog.search_calls[0]
    assert args[0] == BASE
    assert "exclude_isbns" not in kwargs
    assert catalog.vector_requests == []


def test_relevant_marks_pull_the_query_vector_toward_them():
    catalog = _Catalog()
    _service(catalog).search("detective novel", relevant_isbns=["111"])
    sent = catalog.search_calls[0][0][0]
    assert _cos(sent, REL_VEC) > _cos(BASE, REL_VEC)
    assert "exclude_isbns" not in catalog.search_calls[0][1]


def test_irrelevant_marks_push_the_vector_away_and_are_excluded_from_results():
    catalog = _Catalog()
    base = [1.0, 0.0, 1.0]
    service = _service(catalog)
    service.embeddings = SimpleNamespace(embed_query=lambda _q: list(base))
    service.search("detective novel", irrelevant_isbns=["222"])
    args, kwargs = catalog.search_calls[0]
    assert _cos(args[0], IRR_VEC) < _cos(base, IRR_VEC)
    assert kwargs["exclude_isbns"] == ["222"]


def test_relevant_books_are_kept_in_the_results_not_excluded():
    catalog = _Catalog()
    _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    assert catalog.search_calls[0][1]["exclude_isbns"] == ["222"]


def test_a_book_marked_both_ways_counts_as_irrelevant():
    catalog = _Catalog()
    _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["111"])
    assert catalog.search_calls[0][1]["exclude_isbns"] == ["111"]
    assert catalog.vector_requests == [["111"]]  # requested once, as irrelevant


def test_vectors_are_fetched_in_one_lookup():
    catalog = _Catalog()
    _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    assert catalog.vector_requests == [["111", "222"]]


def test_marks_missing_from_the_catalog_are_skipped_but_still_excluded():
    catalog = _Catalog(vectors={})
    _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    args, kwargs = catalog.search_calls[0]
    assert args[0] == BASE  # nothing to refine with
    assert kwargs["exclude_isbns"] == ["222"]


def test_a_failed_vector_lookup_fails_open_but_keeps_the_exclusions():
    catalog = _Catalog(fail_vectors=True)
    results, _ = _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    args, kwargs = catalog.search_calls[0]
    assert args[0] == BASE
    assert kwargs["exclude_isbns"] == ["222"]
    assert len(results) == 1


def test_a_refinement_error_falls_back_to_the_original_vector(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("dimension mismatch")

    monkeypatch.setattr("app.domain.search_service.refine_query_vector", boom)
    catalog = _Catalog()
    results, _ = _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    assert catalog.search_calls[0][0][0] == BASE
    assert len(results) == 1


def test_unsupported_backends_ignore_marks_and_never_get_the_extra_argument(caplog):
    catalog = _PlainCatalog()
    with caplog.at_level(logging.WARNING):
        _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    args, kwargs = catalog.search_calls[0]
    assert args[0] == BASE
    assert "exclude_isbns" not in kwargs
    assert "query_refinement skipped" in caplog.text


def test_zero_weights_disable_the_pull_but_exclusion_still_applies(monkeypatch):
    monkeypatch.setattr(settings, "refine_relevant_weight", 0.0)
    monkeypatch.setattr(settings, "refine_irrelevant_weight", 0.0)
    catalog = _Catalog()
    _service(catalog).search("q", relevant_isbns=["111"], irrelevant_isbns=["222"])
    args, kwargs = catalog.search_calls[0]
    assert args[0] == pytest.approx(BASE, abs=1e-6)
    assert kwargs["exclude_isbns"] == ["222"]


def test_refinement_is_logged_without_the_query_text(caplog):
    with caplog.at_level(logging.INFO):
        _service(_Catalog()).search(
            "confidential detective novel", relevant_isbns=["111"], irrelevant_isbns=["222"]
        )
    assert "query_refined relevant=1/1 irrelevant=1/1 excluded=1" in caplog.text
    assert "confidential" not in caplog.text


def test_refinement_and_vote_adjustments_can_apply_together(monkeypatch):
    monkeypatch.setattr(settings, "feedback_rerank_enabled", True)
    monkeypatch.setattr(settings, "feedback_min_votes", 3)
    done = Future()
    done.set_result({"1": FeedbackVotes(up=9, down=0)})
    feedback = SimpleNamespace(get_votes_async=lambda *_a: done)

    catalog = _Catalog()
    _service(catalog, feedback).search("q", irrelevant_isbns=["222"])
    kwargs = catalog.search_calls[0][1]
    assert kwargs["score_adjustments"]["1"] > 0
    assert kwargs["exclude_isbns"] == ["222"]
