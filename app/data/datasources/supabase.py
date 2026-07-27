import logging

from supabase import Client, create_client

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None
_CATALOG_PAGE_SIZE = 1000


def get_supabase_client() -> Client:
    global _client
    if _client is None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _client


def fetch_all_catalog_isbns(client: Client | None = None) -> set[str]:
    """Load all ISBNs from book_catalog (PostgREST defaults to 1000 rows per request)."""
    db = client or get_supabase_client()
    isbns: set[str] = set()
    offset = 0

    while True:
        result = (
            db.table("book_catalog")
            .select("isbn13")
            .range(offset, offset + _CATALOG_PAGE_SIZE - 1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            break
        isbns.update(str(row["isbn13"]) for row in rows if row.get("isbn13"))
        if len(rows) < _CATALOG_PAGE_SIZE:
            break
        offset += _CATALOG_PAGE_SIZE

    return isbns
