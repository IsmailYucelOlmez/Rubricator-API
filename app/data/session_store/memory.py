import asyncio
import logging
import threading
from datetime import datetime, timezone

from app.core.config import settings
from app.data.session_store.base import compute_expires_at
from app.models.document_session import DocumentSession

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DocumentSession] = {}
        self._lock = threading.RLock()
        self._cleanup_task: asyncio.Task | None = None

    def put(self, session: DocumentSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> DocumentSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if self._is_expired(session):
                self._sessions.pop(session_id, None)
                return None
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def touch(self, session: DocumentSession) -> None:
        with self._lock:
            now = datetime.now(timezone.utc)
            session.last_accessed_at = now
            session.expires_at = compute_expires_at(
                created_at=session.created_at,
                last_accessed_at=now,
                idle_ttl_minutes=settings.session_idle_ttl_minutes,
                max_ttl_minutes=settings.session_max_ttl_minutes,
            )
            self._sessions[session.session_id] = session

    def count(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._sessions)

    def start_cleanup_task(self) -> None:
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session cleanup task started")

    def stop_cleanup_task(self) -> None:
        if self._cleanup_task is None:
            return
        self._cleanup_task.cancel()
        self._cleanup_task = None
        logger.info("Session cleanup task stopped")

    def _is_expired(self, session: DocumentSession) -> bool:
        now = datetime.now(timezone.utc)
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return now >= expires_at

    def _purge_expired(self) -> None:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_expired(session)
        ]
        for session_id in expired_ids:
            del self._sessions[session_id]
        if expired_ids:
            logger.info("Purged %d expired session(s)", len(expired_ids))

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(settings.session_cleanup_interval_seconds)
                with self._lock:
                    self._purge_expired()
        except asyncio.CancelledError:
            return
