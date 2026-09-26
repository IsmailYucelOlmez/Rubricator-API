import asyncio
import importlib
import io
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile
from starlette.datastructures import Headers

from app.core.config import settings

USER_A = "11111111-1111-1111-1111-111111111111"
USER_B = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def sessions(monkeypatch):
    """Import the router with its heavy module-level service stubbed."""
    with patch("app.domain.document_chat_service.DocumentChatService"):
        sys.modules.pop("app.api.routers.sessions", None)
        module = importlib.import_module("app.api.routers.sessions")
    module._session_times.clear()
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "document_max_sessions_per_user_hour", 2)
    monkeypatch.setattr(settings, "document_max_sessions_per_ip_hour", 3)
    yield module
    module._session_times.clear()
    sys.modules.pop("app.api.routers.sessions", None)


def _request(headers: dict[str, str] | None = None, host: str = "203.0.113.7"):
    return SimpleNamespace(
        headers=Headers(headers or {}),
        client=SimpleNamespace(host=host) if host else None,
    )


class _FakeChatService:
    def create_pending_session(self, filename, doc_format):
        return SimpleNamespace(
            session_id="s-1",
            format=doc_format,
            filename=filename,
            expires_at="2099-01-01T00:00:00Z",
            status="processing",
            page_count=None,
            chapter_count=None,
            word_count=0,
            chunk_count=0,
            truncated=False,
            question_count=0,
        )

    def process_session(self, *args, **kwargs):  # pragma: no cover - background
        pass


def _upload(sessions, headers=None, host="203.0.113.7"):
    """Run create_session end to end with a tiny valid PDF."""
    sessions._chat_service = _FakeChatService()
    file = UploadFile(
        file=io.BytesIO(b"%PDF-1.4 tiny"),
        filename="book.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    return asyncio.run(
        sessions.create_session(_request(headers, host), BackgroundTasks(), file, None)
    )


# --- which bucket a request lands in ------------------------------------------


def test_forwarded_account_id_selects_a_per_account_bucket(sessions):
    key, limit, label = sessions._rate_limit_subject(_request({"X-User-Id": USER_A}))
    assert key == f"user:{USER_A}"
    assert limit == settings.document_max_sessions_per_user_hour
    assert label == "account"


def test_account_id_is_case_insensitive(sessions):
    upper = sessions._rate_limit_subject(_request({"X-User-Id": USER_A.upper()}))[0]
    lower = sessions._rate_limit_subject(_request({"X-User-Id": USER_A}))[0]
    assert upper == lower


def test_without_the_header_the_ip_is_the_bucket(sessions):
    key, limit, label = sessions._rate_limit_subject(_request(host="198.51.100.9"))
    assert key == "ip:198.51.100.9"
    assert limit == settings.document_max_sessions_per_ip_hour
    assert label == "IP address"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not-a-uuid", "x" * 500, "../../etc/passwd", USER_A + "0", "1111-1111"],
)
def test_malformed_ids_fall_back_to_the_ip_bucket(sessions, bad):
    """A junk header must neither open a bucket of its own nor crash."""
    key = sessions._rate_limit_subject(_request({"X-User-Id": bad}))[0]
    assert key == "ip:203.0.113.7"


def test_no_client_and_no_header_is_still_handled(sessions):
    assert sessions._rate_limit_subject(_request(host=None))[0] == "ip:unknown"


# --- the limit itself ---------------------------------------------------------


def test_accounts_behind_one_proxy_ip_no_longer_share_a_limit(sessions):
    """The bug this fixes: every user arrives from the edge function's IP."""
    for _ in range(2):
        _upload(sessions, {"X-User-Id": USER_A})
    with pytest.raises(HTTPException) as blocked:
        _upload(sessions, {"X-User-Id": USER_A})
    assert blocked.value.status_code == 429
    assert "account" in blocked.value.detail

    # Same source IP, different account: unaffected by A's limit.
    assert _upload(sessions, {"X-User-Id": USER_B}).sessionId == "s-1"


def test_requests_without_an_id_still_share_the_ip_limit(sessions):
    for _ in range(3):
        _upload(sessions)
    with pytest.raises(HTTPException) as blocked:
        _upload(sessions)
    assert blocked.value.status_code == 429
    assert "IP address" in blocked.value.detail


def test_a_spoofed_or_junk_id_cannot_dodge_the_ip_limit(sessions):
    """Rotating garbage header values must not mint fresh buckets."""
    for i in range(3):
        _upload(sessions, {"X-User-Id": f"junk-{i}"})
    with pytest.raises(HTTPException) as blocked:
        _upload(sessions, {"X-User-Id": "junk-again"})
    assert blocked.value.status_code == 429


def test_old_entries_expire_and_idle_buckets_are_dropped(sessions):
    from datetime import datetime, timedelta, timezone

    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    sessions._session_times["user:gone"] = [stale, stale]
    sessions._check_rate_limit("user:gone", 2, "account")  # not blocked: all expired
    assert "user:gone" not in sessions._session_times

    for _ in range(2):
        sessions._record_session("user:busy")
    with pytest.raises(HTTPException):
        sessions._check_rate_limit("user:busy", 2, "account")
    assert len(sessions._session_times["user:busy"]) == 2


def test_a_rejected_request_is_not_recorded(sessions):
    for _ in range(2):
        _upload(sessions, {"X-User-Id": USER_A})
    with pytest.raises(HTTPException):
        _upload(sessions, {"X-User-Id": USER_A})
    assert len(sessions._session_times[f"user:{USER_A}"]) == 2
