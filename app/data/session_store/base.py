from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from app.models.document_session import DocumentSession


class SessionStore(Protocol):
    def put(self, session: DocumentSession) -> None: ...

    def get(self, session_id: str) -> DocumentSession | None: ...

    def delete(self, session_id: str) -> bool: ...

    def touch(self, session: DocumentSession) -> None: ...

    def count(self) -> int: ...

    def start_cleanup_task(self) -> None: ...

    def stop_cleanup_task(self) -> None: ...


def compute_expires_at(
    created_at: datetime,
    last_accessed_at: datetime,
    idle_ttl_minutes: int,
    max_ttl_minutes: int,
) -> datetime:
    now = datetime.now(timezone.utc)
    idle_expiry = last_accessed_at + timedelta(minutes=idle_ttl_minutes)
    max_expiry = created_at + timedelta(minutes=max_ttl_minutes)
    return min(idle_expiry, max_expiry, now + timedelta(minutes=idle_ttl_minutes))
