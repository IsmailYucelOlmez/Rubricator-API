import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.models.schemas import SemanticSearchRequest


class _StubService:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [], None


@pytest.fixture
def semantic():
    """Import the router with its heavy module-level services stubbed.

    The router builds a search service (Gemini + catalog clients) at import time,
    which needs real credentials; these tests only care about request wiring.
    """
    with (
        patch("app.domain.search_service.SemanticSearchService"),
        patch("app.data.repositories.search_log_repository.SearchLogRepository"),
    ):
        sys.modules.pop("app.api.routers.semantic", None)
        module = importlib.import_module("app.api.routers.semantic")
    yield module
    sys.modules.pop("app.api.routers.semantic", None)


def _call(semantic, monkeypatch, payload):
    stub = _StubService()
    monkeypatch.setattr(semantic, "_search_service", stub)
    monkeypatch.setattr(semantic, "_search_logger", SimpleNamespace(log_anonymous=lambda **_: None))
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    semantic.semantic_search(SemanticSearchRequest.model_validate(payload), None)
    return stub.calls[0]


def test_router_forwards_the_marked_books_to_the_service(semantic, monkeypatch):
    call = _call(
        semantic,
        monkeypatch,
        {
            "query": "detective novel",
            "feedback": {"relevant": ["9780134685991"], "irrelevant": ["9780000000001"]},
        },
    )
    assert call["relevant_isbns"] == ["9780134685991"]
    assert call["irrelevant_isbns"] == ["9780000000001"]


def test_router_passes_none_when_the_client_sends_no_feedback(semantic, monkeypatch):
    call = _call(semantic, monkeypatch, {"query": "detective novel"})
    assert call["relevant_isbns"] is None
    assert call["irrelevant_isbns"] is None
