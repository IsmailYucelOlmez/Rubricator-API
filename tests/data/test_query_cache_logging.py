import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.data.repositories.query_cache_repository import QueryCacheRepository, cache_stats


class _FakeQuery:
    def __init__(self, row):
        self._row = row

    def table(self, _name):
        return self

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        return SimpleNamespace(data=self._row)


def _repo(row):
    repo = QueryCacheRepository.__new__(QueryCacheRepository)
    repo._client = _FakeQuery(row)
    repo._ttl = 3600
    return repo


@pytest.fixture(autouse=True)
def _reset_stats():
    cache_stats.reset()


def _row(expires_in_seconds):
    expires = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return {
        "isbn13_list": ["9780000000001"],
        "rewrite_json": None,
        "expires_at": expires.isoformat(),
    }


def test_hit_is_counted_and_logged(caplog):
    with caplog.at_level(logging.INFO):
        assert _repo(_row(600)).get("q", "advanced", "All", "All") is not None
    assert (cache_stats.hits, cache_stats.misses) == (1, 0)
    assert "query_cache result=hit" in caplog.text
    assert "hit_rate=100%" in caplog.text


def test_missing_row_counts_as_a_miss(caplog):
    with caplog.at_level(logging.INFO):
        assert _repo(None).get("q", "advanced", "All", "All") is None
    assert (cache_stats.hits, cache_stats.misses) == (0, 1)
    assert "query_cache result=miss" in caplog.text


def test_expired_row_counts_as_a_miss(caplog):
    with caplog.at_level(logging.INFO):
        assert _repo(_row(-600)).get("q", "advanced", "All", "All") is None
    assert (cache_stats.hits, cache_stats.misses) == (0, 1)
    assert "query_cache result=expired" in caplog.text


def test_hit_rate_accumulates_across_lookups(caplog):
    with caplog.at_level(logging.INFO):
        _repo(_row(600)).get("a", "advanced", "All", "All")
        _repo(None).get("b", "advanced", "All", "All")
    assert "hit_rate=50%" in caplog.text


def test_log_lines_never_contain_the_query_text(caplog):
    with caplog.at_level(logging.INFO):
        _repo(None).get("my very private query", "advanced", "All", "All")
    assert "private" not in caplog.text
