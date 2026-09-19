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


def _ranked_pool(count=40):
    # Descending scores; each book points in its own direction so MMR has no duplicates to drop.
    return [
        _point(f"97800000000{i:02d}", 0.90 - i * 0.005, [1.0, float(i)])
        for i in range(count)
    ]


def test_feedback_pulls_a_voted_up_book_from_below_the_cutoff_into_the_results(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    pool = _ranked_pool()
    repo = _repo(pool)
    target = pool[30].payload["isbn13"]  # 31st most similar, far outside limit=16

    baseline = repo.search_by_embedding([1.0, 0.0], None, limit=16, initial_k=50)
    assert target not in [row["isbn13"] for row in baseline]

    boosted = repo.search_by_embedding(
        [1.0, 0.0], None, limit=16, initial_k=50, score_adjustments={target: 0.2}
    )
    assert boosted[0]["isbn13"] == target


def test_feedback_fetches_the_full_pool_even_when_mmr_is_off(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    repo = _repo(_ranked_pool())
    repo.search_by_embedding([1.0, 0.0], None, limit=16, initial_k=50, score_adjustments={"x": 0.1})
    call = repo._client.calls[0]
    assert call["limit"] == 50
    assert call["with_vectors"] is False  # vectors are only needed for MMR


def test_no_adjustments_keeps_the_original_small_query(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    repo = _repo(_ranked_pool())
    repo.search_by_embedding([1.0, 0.0], None, limit=16, initial_k=50, score_adjustments=None)
    assert repo._client.calls[0]["limit"] == 16
    repo.search_by_embedding([1.0, 0.0], None, limit=16, initial_k=50, score_adjustments={})
    assert repo._client.calls[1]["limit"] == 16


def test_a_negative_adjustment_demotes_a_book(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    pool = _ranked_pool()
    repo = _repo(pool)
    top = pool[0].payload["isbn13"]
    rows = repo.search_by_embedding(
        [1.0, 0.0], None, limit=5, initial_k=50, score_adjustments={top: -0.5}
    )
    assert top not in [row["isbn13"] for row in rows]


def test_reported_similarity_stays_the_raw_vector_score(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    pool = _ranked_pool()
    repo = _repo(pool)
    target = pool[30]
    rows = repo.search_by_embedding(
        [1.0, 0.0], None, limit=3, initial_k=50, score_adjustments={target.payload["isbn13"]: 0.2}
    )
    assert rows[0]["similarity"] == target.score


def test_adjustments_for_books_outside_the_pool_are_ignored(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    repo = _repo(_ranked_pool())
    rows = repo.search_by_embedding(
        [1.0, 0.0], None, limit=3, initial_k=50, score_adjustments={"9789999999999": 0.5}
    )
    assert [row["isbn13"] for row in rows] == [
        p.payload["isbn13"] for p in _ranked_pool()[:3]
    ]


def test_feedback_changes_which_diverse_book_mmr_picks(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 0.5)
    pool = [
        _point("1", 0.95, [1.0, 0.0]),
        _point("2", 0.94, [0.99, 0.01]),  # near-duplicate of 1
        _point("3", 0.80, [0.0, 1.0]),    # diverse
        _point("4", 0.70, [0.0, -1.0]),   # equally diverse, but less similar
    ]

    without = _repo(pool).search_by_embedding([1.0, 0.0], None, limit=2, initial_k=50)
    assert [row["isbn13"] for row in without] == ["1", "3"]

    with_votes = _repo(pool).search_by_embedding(
        [1.0, 0.0], None, limit=2, initial_k=50, score_adjustments={"4": 0.12}
    )
    assert [row["isbn13"] for row in with_votes] == ["1", "4"]
