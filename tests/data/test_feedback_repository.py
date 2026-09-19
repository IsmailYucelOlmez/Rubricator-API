import logging
import threading
import time
from types import SimpleNamespace

from app.data.repositories import feedback_repository
from app.data.repositories.feedback_repository import FeedbackRepository, FeedbackVotes


class _FakeRpc:
    def __init__(self, client, name, params):
        self._client = client
        self._name = name
        self._params = params

    def execute(self):
        return self._client.run(self._name, self._params)


class _FakeClient:
    def __init__(self, rows=None, error=None, gate=None):
        self.rows = rows or []
        self.error = error
        self.gate = gate
        self.calls = []

    def rpc(self, name, params):
        return _FakeRpc(self, name, params)

    def run(self, name, params):
        self.calls.append((name, params))
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.rows)


ROWS = [
    {"isbn13": "9780000000001", "up": 4, "down": 1},
    {"isbn13": "9780000000002", "up": 0, "down": 3},
]


def test_votes_are_parsed_into_a_dict_keyed_by_isbn():
    repo = FeedbackRepository(client=_FakeClient(ROWS))
    votes = repo.get_votes("detective novel", "tr", timeout=2)
    assert votes == {
        "9780000000001": FeedbackVotes(up=4, down=1),
        "9780000000002": FeedbackVotes(up=0, down=3),
    }


def test_rpc_receives_the_raw_query_and_language():
    client = _FakeClient(ROWS)
    FeedbackRepository(client=client).get_votes("  Detective Novel? ", "tr", timeout=2)
    assert client.calls == [
        ("get_semantic_feedback", {"p_query": "  Detective Novel? ", "p_language": "tr"})
    ]


def test_second_lookup_for_the_same_query_is_served_from_cache():
    client = _FakeClient(ROWS)
    repo = FeedbackRepository(client=client)
    repo.get_votes("detective novel", "tr", timeout=2)
    repo.get_votes("Detective   NOVEL", "tr", timeout=2)  # same query after cache normalization
    assert len(client.calls) == 1


def test_empty_results_are_cached_too():
    client = _FakeClient([])
    repo = FeedbackRepository(client=client)
    assert repo.get_votes("rare query", timeout=2) == {}
    assert repo.get_votes("rare query", timeout=2) == {}
    assert len(client.calls) == 1


def test_language_is_part_of_the_cache_key():
    client = _FakeClient(ROWS)
    repo = FeedbackRepository(client=client)
    repo.get_votes("same words", "tr", timeout=2)
    repo.get_votes("same words", "en", timeout=2)
    assert len(client.calls) == 2


def test_cache_entries_expire(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(feedback_repository.time, "monotonic", lambda: now[0])
    client = _FakeClient(ROWS)
    repo = FeedbackRepository(client=client, ttl_seconds=60)
    repo.get_votes("q", timeout=2)
    now[0] += 30
    repo.get_votes("q", timeout=2)
    assert len(client.calls) == 1
    now[0] += 31
    repo.get_votes("q", timeout=2)
    assert len(client.calls) == 2


def test_supabase_errors_fail_open_and_are_not_cached():
    client = _FakeClient(error=RuntimeError("supabase down"))
    repo = FeedbackRepository(client=client)
    assert repo.get_votes("q", timeout=2) == {}
    client.error = None
    client.rows = ROWS
    assert len(repo.get_votes("q", timeout=2)) == 2  # retried, not stuck on the failure


def test_slow_lookup_times_out_then_still_fills_the_cache():
    gate = threading.Event()
    client = _FakeClient(ROWS, gate=gate)
    repo = FeedbackRepository(client=client)

    started = time.monotonic()
    assert repo.get_votes("q", timeout=0.05) == {}
    assert time.monotonic() - started < 1.0  # did not wait for the slow call

    gate.set()
    for _ in range(100):  # the background lookup completes and warms the cache
        cached = repo.get_votes("q", timeout=0.05)
        if cached:
            break
        time.sleep(0.02)
    assert len(cached) == 2
    assert len(client.calls) == 1


def test_async_lookup_lets_the_caller_overlap_other_work():
    client = _FakeClient(ROWS)
    repo = FeedbackRepository(client=client)
    future = repo.get_votes_async("q")
    assert len(FeedbackRepository.resolve(future, timeout=2)) == 2


def test_cache_hit_returns_an_already_completed_future():
    repo = FeedbackRepository(client=_FakeClient(ROWS))
    repo.get_votes("q", timeout=2)
    assert repo.get_votes_async("q").done()


def test_lookup_is_logged_without_the_query_text(caplog):
    repo = FeedbackRepository(client=_FakeClient(ROWS))
    with caplog.at_level(logging.INFO):
        repo.get_votes("my private query", timeout=2)
    assert "feedback_lookup ms=" in caplog.text
    assert "rows=2" in caplog.text
    assert "private" not in caplog.text


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(feedback_repository, "_MAX_CACHE_ENTRIES", 3)
    repo = FeedbackRepository(client=_FakeClient(ROWS))
    for i in range(6):
        repo.get_votes(f"query {i}", timeout=2)
    assert len(repo._cache) == 3


def test_concurrent_lookups_for_the_same_query_share_one_rpc():
    gate = threading.Event()
    client = _FakeClient(ROWS, gate=gate)
    repo = FeedbackRepository(client=client)

    first = repo.get_votes_async("q")
    second = repo.get_votes_async("Q ")  # same query while the first is still running
    assert second is first

    gate.set()
    assert len(FeedbackRepository.resolve(first, timeout=2)) == 2
    assert len(client.calls) == 1


def test_inflight_entry_is_released_after_completion_so_failures_can_retry():
    client = _FakeClient(error=RuntimeError("down"))
    repo = FeedbackRepository(client=client)
    FeedbackRepository.resolve(repo.get_votes_async("q"), timeout=2)
    for _ in range(100):
        if not repo._inflight:
            break
        time.sleep(0.01)
    assert repo._inflight == {}


def test_malformed_rows_fail_open_instead_of_breaking_the_search():
    repo = FeedbackRepository(client=_FakeClient([{"isbn13": "9780000000001", "up": None, "down": 1}]))
    assert repo.get_votes("q", timeout=2) == {}


def test_resolve_never_raises_even_if_the_lookup_future_failed():
    from concurrent.futures import Future

    failed = Future()
    failed.set_exception(ValueError("boom"))
    assert FeedbackRepository.resolve(failed, timeout=1) == {}
