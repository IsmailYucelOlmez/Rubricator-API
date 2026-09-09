import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from app.core.config import settings
from app.data.datasources.gemini import GeminiEmbeddingClient
from app.data.datasources.google_books import GoogleBooksClient
from app.data.repositories.book_catalog_repository import get_catalog_repository
from app.data.repositories.query_cache_repository import QueryCacheRepository
from app.domain.book_normalizer import normalize_volumes
from app.domain.query_rewriter import QueryRewriter
from app.models.schemas import RewriteResultModel, SemanticBookResult

logger = logging.getLogger(__name__)

TONE_COLUMNS = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


class SemanticSearchService:
    """Domain/business logic: orchestrates datasources and repositories."""

    def __init__(
        self,
        embeddings: GeminiEmbeddingClient | None = None,
        google_client: GoogleBooksClient | None = None,
        query_rewriter: QueryRewriter | None = None,
        query_cache: QueryCacheRepository | None = None,
        catalog: Any | None = None,
    ) -> None:
        self.embeddings = embeddings or GeminiEmbeddingClient()
        self.google_client = google_client or GoogleBooksClient()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.query_cache = query_cache or QueryCacheRepository()
        self.catalog = catalog or get_catalog_repository(self.embeddings)

    def search(
        self,
        query: str,
        mode: Literal["simple", "advanced"] = "simple",
        category: str = "All",
        tone: str = "All",
        limit: int | None = None,
        initial_top_k: int | None = None,
        language: str | None = None,
    ) -> tuple[list[SemanticBookResult], str | None]:
        search_query = query.strip()[: settings.max_query_length]
        if not search_query:
            return [], None

        # Google-ingested books lack emotion_scores; tone sort/rewrite would be misleading.
        if mode == "advanced":
            tone = "All"

        # Turkish mode never calls Google Books: the local catalog already
        # covers Turkish books, and Google Books results skew English anyway.
        effective_mode = "simple" if language == "tr" else mode

        rewritten: str | None = None
        query_vector: list[float] | None = None

        if effective_mode == "advanced":
            rewrite_result, query_vector = self._fetch_and_ingest(
                search_query,
                category=category,
                tone=tone,
            )
            if rewrite_result:
                rewritten = rewrite_result.effective_local_query(search_query)
                search_query = rewritten

        if query_vector is None:
            query_vector = self.embeddings.embed_query(search_query)
        if query_vector is None:
            raise RuntimeError("Failed to generate query embedding")

        final_limit = min(limit or settings.default_limit, settings.max_limit)
        initial_k = initial_top_k or settings.initial_top_k

        rows = self.catalog.search_by_embedding(
            query_vector,
            category if category != "All" else None,
            final_limit,
            initial_k,
            language,
        )
        rows = self._apply_tone_sort(rows, tone)
        rows = rows[:final_limit]

        results = [self._row_to_result(row) for row in rows]
        return results, rewritten

    def _fetch_and_ingest(
        self,
        query: str,
        category: str,
        tone: str,
    ) -> tuple[RewriteResultModel | None, list[float] | None]:
        cached = self.query_cache.get(query, "advanced", category, tone)
        if cached is not None:
            _cached_isbns, rewrite = cached
            local_query = rewrite.effective_local_query(query) if rewrite else query.strip()
            query_vector = self._embed_query(local_query)
            return rewrite, query_vector

        with ThreadPoolExecutor(max_workers=2) as executor:
            rewrite_future = executor.submit(self.query_rewriter.rewrite, query, category, tone)
            existing_isbns_future = executor.submit(self.catalog.list_existing_isbns)
            rewrite_result = rewrite_future.result()
            existing_isbns = existing_isbns_future.result()

        api_queries = (
            rewrite_result.effective_api_queries(query) if rewrite_result else [query.strip()]
        )
        local_query = (
            rewrite_result.effective_local_query(query) if rewrite_result else query.strip()
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            volumes_future = executor.submit(self.google_client.search_many, api_queries)
            query_vector_future = executor.submit(self._embed_query, local_query)
            volumes = volumes_future.result()
            query_vector = query_vector_future.result()

        books = normalize_volumes(
            volumes,
            existing_isbns,
            max_books=settings.max_new_books,
            category=category,
        )

        if not books:
            self.query_cache.set(query, "advanced", category, tone, [], rewrite_result)
            return rewrite_result, query_vector

        ingested = self.catalog.upsert_books(books)
        ingested_isbns = [book.isbn13 for book in ingested] if ingested else [book.isbn13 for book in books]
        self.query_cache.set(query, "advanced", category, tone, ingested_isbns, rewrite_result)
        return rewrite_result, query_vector

    def _embed_query(self, query: str) -> list[float] | None:
        query = query.strip()
        if not query:
            return None
        return self.embeddings.embed_query(query)

    def _row_to_result(self, row: dict[str, Any]) -> SemanticBookResult:
        thumbnail = row.get("thumbnail_url")
        cover = f"{thumbnail}&fife=w800" if thumbnail else None
        authors = row.get("authors") or ""
        author = authors.split(";")[0].strip() if authors else "Unknown author"
        source = str(row.get("source") or "local")
        return SemanticBookResult(
            isbn13=str(row.get("isbn13") or ""),
            title=row.get("title") or "Unknown title",
            author=author,
            description=row.get("description") or "",
            coverImageUrl=cover,
            category=row.get("simple_category"),
            similarity=float(row["similarity"]) if row.get("similarity") is not None else None,
            source=source,
        )

    def _apply_tone_sort(self, rows: list[dict[str, Any]], tone: str) -> list[dict[str, Any]]:
        if not rows or tone == "All":
            return rows

        column = TONE_COLUMNS.get(tone)
        if not column:
            return rows

        def tone_key(row: dict[str, Any]) -> tuple[int, float]:
            scores = row.get("emotion_scores") or {}
            value = scores.get(column)
            if value is None:
                return (0, 0.0)
            try:
                return (1, float(value))
            except (TypeError, ValueError):
                return (0, 0.0)

        return sorted(rows, key=tone_key, reverse=True)


# Backward-compatible alias
HybridRecommender = SemanticSearchService
