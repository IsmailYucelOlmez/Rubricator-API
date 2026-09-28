import math
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import HTTPException, Request


class SlidingWindowRateLimiter:
    """In-process sliding-window limiter: at most `limit()` hits per key per window.

    State lives in this process only, like the session store — fine for a single
    instance, but every extra instance or worker gets its own budget.
    """

    def __init__(self, limit: Callable[[], int], window_seconds: float = 60.0) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def hit(self, key: str) -> float | None:
        """Records a hit for `key`; returns seconds to wait if it's over the limit."""
        limit = self._limit()
        if limit <= 0:
            return None
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            self._sweep(now, cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                return hits[0] + self._window - now
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _sweep(self, now: float, cutoff: float) -> None:
        # Drop idle keys once per window so the dict doesn't grow with every caller ever seen.
        if now - self._last_sweep < self._window:
            return
        self._last_sweep = now
        for key in [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_ip(request: Request) -> str:
    # request.client is the direct peer. Behind a reverse proxy, run uvicorn with
    # --proxy-headers --forwarded-allow-ips=<proxy ip> so this is the real client;
    # X-Forwarded-For isn't read here because any caller can forge it.
    return request.client.host if request.client else "unknown"


def rate_limit(limiter: SlidingWindowRateLimiter) -> Callable[[Request], None]:
    """FastAPI dependency that answers 429 (with Retry-After) once a client IP is over the limit."""

    def dependency(request: Request) -> None:
        retry_after = limiter.hit(f"ip:{client_ip(request)}")
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests; try again shortly",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    return dependency
