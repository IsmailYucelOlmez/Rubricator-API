from types import SimpleNamespace

from app.core.config import settings
from app.data.datasources.qdrant import isbn_to_point_id
from app.data.repositories.qdrant_book_catalog_repository import QdrantBookCatalogRepository


def _point(isbn, score):
    return SimpleNamespace(
        score=score,
        vector=None,
        payload={"isbn13": isbn, "title": isbn, "authors": "A", "description": "d"},
    )


def _record(isbn, vector):
    return SimpleNamespace(payload={"isbn13": isbn}, vector=vector)


class _Client:
    def __init__(self, records=None, error=None):
        self.records = records or []
        self.error = error
        self.retrieve_calls = []
        self.query_calls = []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        if self.error:
            raise self.error
        return self.records

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=[_point("1", 0.9), _point("2", 0.8)])


def _repo(client):
    # Bypass __init__: it would build a real Gemini client and a Qdrant connection.
    repo = QdrantBookCatalogRepository.__new__(QdrantBookCatalogRepository)
    repo._client = client
    return repo


def test_get_vectors_maps_isbns_to_stored_embeddings():
    client = _Client([_record("111", [1.0, 0.0]), _record("222", [0.0, 1.0])])
    vectors = _repo(client).get_vectors(["111", "222"])
    assert vectors == {"111": [1.0, 0.0], "222": [0.0, 1.0]}
    call = client.retrieve_calls[0]
    assert call["ids"] == [isbn_to_point_id("111"), isbn_to_point_id("222")]
    assert call["with_vectors"] is True


def test_get_vectors_skips_missing_or_non_list_vectors():
    client = _Client([_record("111", None), _record("222", [0.5, 0.5])])
    assert _repo(client).get_vectors(["111", "222", "333"]) == {"222": [0.5, 0.5]}


def test_get_vectors_fails_open_on_errors():
    assert _repo(_Client(error=RuntimeError("qdrant down"))).get_vectors(["111"]) == {}


def test_get_vectors_without_isbns_makes_no_call():
    client = _Client()
    assert _repo(client).get_vectors([]) == {}
    assert client.retrieve_calls == []


def test_repository_advertises_feedback_refinement_support():
    assert QdrantBookCatalogRepository.supports_feedback_refinement is True


def test_exclude_isbns_become_a_must_not_filter(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    client = _Client()
    _repo(client).search_by_embedding([1.0, 0.0], None, 2, 50, exclude_isbns=["1", "9"])
    query_filter = client.query_calls[0]["query_filter"]
    assert query_filter.must is None
    (condition,) = query_filter.must_not
    assert condition.key == "isbn13"
    assert condition.match.any == ["1", "9"]


def test_exclusion_combines_with_category_and_language_filters(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    client = _Client()
    _repo(client).search_by_embedding(
        [1.0, 0.0], "Fiction", 2, 50, language="tr", exclude_isbns=["1"]
    )
    query_filter = client.query_calls[0]["query_filter"]
    assert {c.key for c in query_filter.must} == {"simple_category", "language"}
    assert [c.key for c in query_filter.must_not] == ["isbn13"]


def test_no_exclusions_keeps_the_previous_filter_shape(monkeypatch):
    monkeypatch.setattr(settings, "search_mmr_lambda", 1.0)
    client = _Client()
    repo = _repo(client)
    repo.search_by_embedding([1.0, 0.0], None, 2, 50)
    assert client.query_calls[0]["query_filter"] is None
    repo.search_by_embedding([1.0, 0.0], "Fiction", 2, 50)
    query_filter = client.query_calls[1]["query_filter"]
    assert query_filter.must_not is None
    assert [c.key for c in query_filter.must] == ["simple_category"]
