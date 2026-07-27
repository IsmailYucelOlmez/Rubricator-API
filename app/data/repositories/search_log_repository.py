import logging

from app.data.datasources.supabase import get_supabase_client

logger = logging.getLogger(__name__)


class SearchLogRepository:
    """Persistence: semantic_search_logs (server-side anonymous rows)."""

    def __init__(self) -> None:
        self._client = get_supabase_client()

    def log_anonymous(
        self,
        query: str,
        mode: str,
        category: str | None,
        tone: str | None,
        result_count: int,
    ) -> None:
        trimmed = query.strip()
        if not trimmed:
            return
        try:
            self._client.table("semantic_search_logs").insert(
                {
                    "user_id": None,
                    "query": trimmed,
                    "mode": mode,
                    "category": category,
                    "tone": tone,
                    "result_count": result_count,
                }
            ).execute()
        except Exception as error:
            logger.warning("Semantic search log insert failed: %s", error)
