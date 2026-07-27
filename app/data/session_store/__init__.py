from app.core.config import settings
from app.data.session_store.memory import InMemorySessionStore

_session_store: InMemorySessionStore | None = None


def get_session_store() -> InMemorySessionStore:
    global _session_store
    if _session_store is None:
        backend = settings.session_store_backend
        if backend == "redis":
            raise NotImplementedError(
                "Redis session store is not implemented yet; set SESSION_STORE_BACKEND=memory"
            )
        _session_store = InMemorySessionStore()
    return _session_store
