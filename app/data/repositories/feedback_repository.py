import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.data.datasources.supabase import get_supabase_client

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="feedback-lookup")
_MAX_CACHE_ENTRIES = 1000


@dataclass(frozen=True)
class FeedbackVotes:
    up: int
    down: int


class FeedbackRepository:
    """Read side of the relevance-feedback data (Supabase `get_semantic_feedback` RPC).

    Lookups never break a search: errors and timeouts yield "no votes". Results,
    including "no votes", are cached briefly so repeated queries cost nothing.
    Users write votes straight to Supabase from the app (submit_semantic_feedback).
    """

    def __init__(self, client: Any | None = None, ttl_seconds: float | None = None) -> None:
        self._client = client
        self._ttl = settings.feedback_cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._lock = threading.RLock()
        self._inflight: dict[tuple[str, str], Future] = {}
        self._cache: OrderedDict[tuple[str, str], tuple[float, dict[str, FeedbackVotes]]] = (
            OrderedDict()
        )

    def get_votes_async(self, query: str, language: str | None = None) -> Future:
        """Start (or instantly satisfy from cache) a vote lookup.

        Lets the caller overlap the lookup with slower work, e.g. the query embedding.
        """
        key = self._cache_key(query, language)
        with self._lock:
            cached = self._cache_get(key)
            if cached is not None:
                done: Future = Future()
                done.set_result(cached)
                return done
            pending = self._inflight.get(key)
            if pending is not None:
                # A slow lookup is already running for this query; share it instead
                # of piling up duplicate RPCs.
                return pending
            future = _executor.submit(self._fetch, key, query, language)
            self._inflight[key] = future
            future.add_done_callback(lambda _f, k=key: self._forget_inflight(k))
            return future

    def _forget_inflight(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._inflight.pop(key, None)

    def get_votes(
        self,
        query: str,
        language: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, FeedbackVotes]:
        return self.resolve(self.get_votes_async(query, language), timeout)

    @staticmethod
    def resolve(future: Future, timeout: float | None = None) -> dict[str, FeedbackVotes]:
        wait = settings.feedback_lookup_timeout_seconds if timeout is None else timeout
        try:
            return future.result(timeout=wait)
        except FutureTimeout:
            # The lookup keeps running and will populate the cache for the next search.
            logger.warning("feedback_lookup timed out after %.0fms; continuing without votes", wait * 1000)
            return {}
        except Exception as error:
            logger.warning("feedback_lookup failed: %s", error)
            return {}

    def _fetch(
        self,
        key: tuple[str, str],
        query: str,
        language: str | None,
    ) -> dict[str, FeedbackVotes]:
        started = time.monotonic()
        try:
            client = self._client or get_supabase_client()
            result = client.rpc(
                "get_semantic_feedback",
                {"p_query": query, "p_language": language},
            ).execute()
            votes = {
                str(row["isbn13"]): FeedbackVotes(up=int(row["up"]), down=int(row["down"]))
                for row in (result.data or [])
                if row.get("isbn13")
            }
        except Exception as error:
            logger.warning("feedback_lookup failed: %s", error)
            return {}

        self._cache_set(key, votes)
        logger.info(
            "feedback_lookup ms=%d rows=%d",
            round((time.monotonic() - started) * 1000),
            len(votes),
        )
        return votes

    @staticmethod
    def _cache_key(query: str, language: str | None) -> tuple[str, str]:
        # Only a local cache key: coarser than the SQL key would just mean fewer hits.
        return " ".join(query.casefold().split()), (language or "").strip().lower()

    def _cache_get(self, key: tuple[str, str]) -> dict[str, FeedbackVotes] | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            stored_at, votes = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return votes

    def _cache_set(self, key: tuple[str, str], votes: dict[str, FeedbackVotes]) -> None:
        with self._lock:
            self._cache[key] = (time.monotonic(), votes)
            self._cache.move_to_end(key)
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
