import threading
import time
from types import SimpleNamespace

import pytest

from app.data.datasources import gemini
from app.data.datasources.gemini import GlobalEmbeddingLimiter, embed_batch_with_retry


class _FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _limiter(clock, max_concurrent=2, rpm=0):
    return GlobalEmbeddingLimiter(max_concurrent, rpm, clock=clock.now, sleep=clock.sleep)


def test_slot_caps_in_flight_requests_across_threads():
    limiter = GlobalEmbeddingLimiter(max_concurrent=2)
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def worker():
        with limiter.slot():
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert state["max"] == 2


def test_pacing_spaces_consecutive_requests():
    clock = _FakeClock()
    limiter = _limiter(clock, rpm=6)  # one request every 10s
    with limiter.slot():
        pass
    with limiter.slot():
        pass
    assert clock.sleeps == [10.0]


def test_no_pacing_when_rpm_is_zero():
    clock = _FakeClock()
    limiter = _limiter(clock, rpm=0)
    for _ in range(5):
        with limiter.slot():
            pass
    assert clock.sleeps == []


def test_penalize_delays_every_request_that_has_not_started():
    clock = _FakeClock()
    limiter = _limiter(clock)
    limiter.penalize(30)
    with limiter.slot():
        pass
    assert clock.sleeps == [30.0]


def test_slot_is_released_when_the_request_raises():
    limiter = GlobalEmbeddingLimiter(max_concurrent=1)
    with pytest.raises(RuntimeError):
        with limiter.slot():
            raise RuntimeError("boom")
    with limiter.slot():  # would deadlock if the semaphore leaked
        pass


class _FlakyEmbeddings:
    def __init__(self, failures):
        self._failures = list(failures)
        self.calls = 0

    def embed_documents(self, texts, batch_size):
        self.calls += 1
        if self._failures:
            raise RuntimeError(self._failures.pop(0))
        return [[0.1, 0.2] for _ in texts]


def _patched(monkeypatch, clock):
    limiter = _limiter(clock)
    monkeypatch.setattr(gemini, "_limiter", limiter)
    monkeypatch.setattr(gemini, "time", SimpleNamespace(sleep=clock.sleep))
    return limiter


def test_rate_limited_batch_retries_and_cools_down_the_shared_limiter(monkeypatch):
    clock = _FakeClock()
    limiter = _patched(monkeypatch, clock)
    embeddings = _FlakyEmbeddings(["429 RESOURCE_EXHAUSTED. Please retry in 3s"])

    vectors = embed_batch_with_retry(embeddings, ["a", "b"])

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert embeddings.calls == 2
    assert limiter._cooldown_until == 65.0
    assert 65.0 in clock.sleeps


def test_transient_errors_retry_without_a_shared_cooldown(monkeypatch):
    clock = _FakeClock()
    limiter = _patched(monkeypatch, clock)
    embeddings = _FlakyEmbeddings(["503 Service Unavailable"])

    embed_batch_with_retry(embeddings, ["a"])

    assert embeddings.calls == 2
    assert limiter._cooldown_until == 0.0


def test_non_retryable_errors_propagate_immediately(monkeypatch):
    clock = _FakeClock()
    _patched(monkeypatch, clock)
    embeddings = _FlakyEmbeddings(["400 bad request"])
    with pytest.raises(RuntimeError, match="400"):
        embed_batch_with_retry(embeddings, ["a"])
    assert embeddings.calls == 1


def test_query_embedding_429_cools_down_the_batch_lane(monkeypatch):
    clock = _FakeClock()
    limiter = _patched(monkeypatch, clock)

    client = gemini.GeminiEmbeddingClient.__new__(gemini.GeminiEmbeddingClient)

    class _Failing:
        def embed_query(self, _query):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    client._embeddings = _Failing()
    assert client.embed_query("hello") is None
    assert limiter._cooldown_until > 0
