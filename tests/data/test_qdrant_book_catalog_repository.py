from types import SimpleNamespace

from app.core.config import settings
from app.data.repositories.qdrant_book_catalog_repository import QdrantBookCatalogRepository


def _point(isbn, score, vector):
    return SimpleNamespace(
        score=score,
        vector=vector,
        payload={"isbn13": isbn, "title": isbn, "authors": "A", "description": "d"},
    )


class _FakeClient:
    def __init__(self, points):
        self._points = points
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points[: kwargs["limit"]])


def _repo(points):
    # Bypass __init__: it would build a real Gemini client and Qdrant connection.
    repo = QdrantBookCatalogRepository.__new__(QdrantBookCatalogRepository)
    repo._client = _FakeClient(points)
    return repo


POOL = [
    _point("1", 0.95, [1.0, 0.0]),
    _point("2", 0.94, [0.99, 0.01]),  # near-duplicate of 1
    _point("3", 0.80, [0.0, 1.0]),
]


def test_mmr_fetches_the_full_candidate_pool_with_vectors(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 0.5)
    repo = _repo(POOL)
    repo.search_by_embedding([1.0, 0.0], None, limit=2, initial_k=50)
    call = repo._client.calls[0]
    assert call["limit"] == 50
    assert call["with_vectors"] is True


def test_mmr_replaces_a_near_duplicate_with_a_diverse_result(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 0.5)
    repo = _repo(POOL)
    rows = repo.search_by_embedding([1.0, 0.0], None, limit=2, initial_k=50)
    assert [row["isbn13"] for row in rows] == ["1", "3"]
    assert rows[0]["similarity"] == 0.95


def test_lambda_one_keeps_the_original_top_limit_query_without_vectors(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    repo = _repo(POOL)
    rows = repo.search_by_embedding([1.0, 0.0], None, limit=2, initial_k=50)
    call = repo._client.calls[0]
    assert call["limit"] == 2
    assert call["with_vectors"] is False
    assert [row["isbn13"] for row in rows] == ["1", "2"]


def test_missing_vectors_fall_back_to_plain_top_limit(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 0.5)
    pool = [_point(str(i), 0.9 - i / 100, None) for i in range(5)]
    repo = _repo(pool)
    rows = repo.search_by_embedding([1.0, 0.0], None, limit=2, initial_k=50)
    assert [row["isbn13"] for row in rows] == ["0", "1"]


def test_pool_not_larger_than_limit_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 0.5)
    repo = _repo(POOL[:2])
    rows = repo.search_by_embedding([1.0, 0.0], None, limit=5, initial_k=50)
    assert [row["isbn13"] for row in rows] == ["1", "2"]

