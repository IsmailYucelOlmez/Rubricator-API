from types import SimpleNamespace

from app.data.repositories.book_catalog_repository import SupabaseBookCatalogRepository


class _Rpc:
    def __init__(self, rows):
        self._rows = rows

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _Client:
    def __init__(self, rows):
        self._rows = rows

    def rpc(self, _name, _params):
        return _Rpc(self._rows)


def _repo(rows):
    repo = SupabaseBookCatalogRepository.__new__(SupabaseBookCatalogRepository)
    repo._client = _Client(rows)
    return repo


ROWS = [
    {"isbn13": "1", "similarity": 0.90},
    {"isbn13": "2", "similarity": 0.89},
    {"isbn13": "3", "similarity": 0.70},
]


def test_without_adjustments_rows_keep_the_rpc_order():
    rows = _repo(ROWS).search_by_embedding([0.0], None, 3, 50)
    assert [r["isbn13"] for r in rows] == ["1", "2", "3"]


def test_adjustments_reorder_the_returned_rows():
    rows = _repo(ROWS).search_by_embedding([0.0], None, 3, 50, score_adjustments={"2": 0.05})
    assert [r["isbn13"] for r in rows] == ["2", "1", "3"]


def test_adjustments_never_change_the_reported_similarity():
    rows = _repo(ROWS).search_by_embedding([0.0], None, 3, 50, score_adjustments={"3": 0.5})
    assert rows[0]["isbn13"] == "3"
    assert rows[0]["similarity"] == 0.70
