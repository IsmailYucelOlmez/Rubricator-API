import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings
from app.data.datasources.supabase import get_supabase_client
from app.models.schemas import RewriteResultModel

logger = logging.getLogger(__name__)


def make_cache_key(query: str, mode: str, category: str, tone: str) -> str:
    normalized = "|".join(
        [
            query.strip().lower(),
            mode.strip().lower(),
            category.strip().lower(),
            tone.strip().lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class QueryCacheRepository:
    """Persistence: semantic_query_cache table."""

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._ttl = settings.semantic_query_cache_ttl_seconds

    def get(
        self,
        query: str,
        mode: str,
        category: str,
        tone: str,
    ) -> tuple[list[str], RewriteResultModel | None] | None:
        cache_key = make_cache_key(query, mode, category, tone)
        try:
            result = (
                self._client.table("semantic_query_cache")
                .select("isbn13_list, rewrite_json, expires_at")
                .eq("cache_key", cache_key)
                .maybe_single()
                .execute()
            )
        except Exception as error:
            logger.warning("Query cache read failed: %s", error)
            return None

        if result is None or not result.data:
            return None

        row = result.data

        expires_at = row.get("expires_at")
        if expires_at:
            try:
                expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if expiry <= datetime.now(timezone.utc):
                    return None
            except ValueError:
                pass

        isbns = [str(isbn) for isbn in (row.get("isbn13_list") or []) if str(isbn)]
        rewrite_payload = row.get("rewrite_json")
        rewrite = None
        if rewrite_payload:
            try:
                rewrite = RewriteResultModel.model_validate(rewrite_payload)
            except Exception as error:
                logger.warning("Invalid cached rewrite payload: %s", error)

        return isbns, rewrite

    def set(
        self,
        query: str,
        mode: str,
        category: str,
        tone: str,
        isbns: list[str],
        rewrite: RewriteResultModel | None = None,
    ) -> None:
        cache_key = make_cache_key(query, mode, category, tone)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
        payload: dict[str, Any] = {
            "cache_key": cache_key,
            "isbn13_list": isbns,
            "rewrite_json": rewrite.model_dump() if rewrite else None,
            "expires_at": expires_at.isoformat(),
        }
        try:
            self._client.table("semantic_query_cache").upsert(payload).execute()
        except Exception as error:
            logger.warning("Query cache write failed: %s", error)
