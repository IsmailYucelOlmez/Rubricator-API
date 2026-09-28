import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core import rate_limit as rate_limit_module
from app.core.config import settings
from app.core.rate_limit import SlidingWindowRateLimiter


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(rate_limit_module.time, "monotonic", fake)
    return fake


# --- limiter ------------------------------------------------------------------


def test_allows_up_to_the_limit_then_blocks(clock):
    limiter = SlidingWindowRateLimiter(lambda: 3)
    assert [limiter.hit("a") for _ in range(3)] == [None, None, None]
    assert limiter.hit("a") == pytest.approx(60.0)


def test_keys_have_separate_budgets(clock):
    limiter = SlidingWindowRateLimiter(lambda: 1)
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None
    assert limiter.hit("a") is not None


def test_budget_frees_up_as_old_hits_leave_the_window(clock):
    limiter = SlidingWindowRateLimiter(lambda: 2)
    limiter.hit("a")
    clock.now += 30
    limiter.hit("a")
    assert limiter.hit("a") == pytest.approx(30.0)
    clock.now += 30
    assert limiter.hit("a") is None


def test_blocked_hits_do_not_extend_the_wait(clock):
    limiter = SlidingWindowRateLimiter(lambda: 1)
    limiter.hit("a")
    for _ in range(5):
        clock.now += 10
        limiter.hit("a")
    assert limiter.hit("a") == pytest.approx(10.0)


def test_zero_limit_disables_limiting(clock):
    limiter = SlidingWindowRateLimiter(lambda: 0)
    assert all(limiter.hit("a") is None for _ in range(100))


def test_limit_is_read_on_every_hit(clock):
    limit = {"value": 1}
    limiter = SlidingWindowRateLimiter(lambda: limit["value"])
    limiter.hit("a")
    assert limiter.hit("a") is not None
    limit["value"] = 5
    assert limiter.hit("a") is None


def test_idle_keys_are_swept(clock):
    limiter = SlidingWindowRateLimiter(lambda: 5)
    limiter.hit("a")
    clock.now += 61
    limiter.hit("b")
    assert set(limiter._hits) == {"b"}


# --- dependency over HTTP -------------------------------------------------------


def _client(limit: int) -> TestClient:
    app = FastAPI()
    limiter = SlidingWindowRateLimiter(lambda: limit)

    @app.get("/limited", dependencies=[Depends(rate_limit_module.rate_limit(limiter))])
    def limited() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_dependency_answers_429_with_retry_after(clock):
    client = _client(limit=2)
    assert [client.get("/limited").status_code for _ in range(2)] == [200, 200]
    response = client.get("/limited")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


# --- wiring on the Gemini-backed endpoints --------------------------------------


@pytest.fixture
def semantic(monkeypatch):
    with (
        patch("app.domain.search_service.SemanticSearchService"),
        patch("app.data.repositories.search_log_repository.SearchLogRepository"),
    ):
        sys.modules.pop("app.api.routers.semantic", None)
        module = importlib.import_module("app.api.routers.semantic")
    module._search_service = SimpleNamespace(search=lambda **_: ([], None))
    module._search_logger = SimpleNamespace(log_anonymous=lambda **_: None)
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "api_key", "")
    yield module
    sys.modules.pop("app.api.routers.semantic", None)


@pytest.fixture
def trbooks(monkeypatch):
    with patch("app.domain.description_generator.TrbookDescriptionGenerator"):
        sys.modules.pop("app.api.routers.trbooks", None)
        module = importlib.import_module("app.api.routers.trbooks")
    module._description_generator = SimpleNamespace(generate=lambda **_: "Kısa açıklama.")
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "api_key", "")
    yield module
    sys.modules.pop("app.api.routers.trbooks", None)


def _app(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_semantic_search_is_rate_limited(semantic, monkeypatch, clock):
    monkeypatch.setattr(settings, "search_max_requests_per_minute", 2)
    client = _app(semantic.router)
    codes = [
        client.post("/api/v1/semantic/search", json={"query": "detective novel"}).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


def test_description_generation_is_rate_limited(trbooks, monkeypatch, clock):
    monkeypatch.setattr(settings, "description_max_requests_per_minute", 1)
    client = _app(trbooks.router)
    body = {"title": "Tutunamayanlar", "author": "Oğuz Atay", "isbn": "9789754700114"}
    codes = [
        client.post("/api/v1/trbooks/generate-description", json=body).status_code
        for _ in range(2)
    ]
    assert codes == [200, 429]


def test_rejected_api_key_does_not_use_up_the_budget(semantic, monkeypatch, clock):
    monkeypatch.setattr(settings, "api_key", "secret")
    monkeypatch.setattr(settings, "search_max_requests_per_minute", 1)
    client = _app(semantic.router)
    for _ in range(3):
        assert client.post("/api/v1/semantic/search", json={"query": "x"}).status_code == 401
    response = client.post(
        "/api/v1/semantic/search",
        json={"query": "x"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
