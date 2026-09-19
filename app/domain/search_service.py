import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from app.core.config import settings
from app.data.datasources.gemini import GeminiEmbeddingClient
from app.data.datasources.google_books import GoogleBooksClient
from app.data.repositories.book_catalog_repository import get_catalog_repository
from app.data.repositories.feedback_repository import FeedbackRepository
from app.data.repositories.query_cache_repository import QueryCacheRepository
from app.domain.book_normalizer import normalize_volumes
from app.domain.feedback_scoring import compute_adjustments
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
        feedback: FeedbackRepository | None = None,
    ) -> None:
        self.embeddings = embeddings or GeminiEmbeddingClient()
        self.google_client = google_client or GoogleBooksClient()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.query_cache = query_cache or QueryCacheRepository()
        self.catalog = catalog or get_catalog_repository(self.embeddings)
        self.feedback = feedback or FeedbackRepository()

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

        # The TR catalog's simple_category values don't line up with the app's
        # Fiction/Nonfiction taxonomy: most Kitapyurdu-sourced rows have no
        # category at all, and the Kaggle-sourced rows use a different, unmapped
        # taxonomy (literature/academic/business_and_economy/...). Filtering by
        # category would silently exclude most of the TR catalog either way, so
        # skip the filter entirely for Turkish searches.
        effective_category = "All" if language == "tr" else category

        rewritten: str | None = None
        query_vector: list[float] | None = None

        # Votes depend only on the query text the user typed, so start the lookup now
        # and let it overlap with the rewrite/embedding work below.
        feedback_lookup = (
            self.feedback.get_votes_async(query, language)
            if settings.feedback_rerank_enabled
            else None
        )

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

        adjustments = self._feedback_adjustments(feedback_lookup)

        rows = self.catalog.search_by_embedding(
            query_vector,
            effective_category if effective_category != "All" else None,
            final_limit,
            initial_k,
            language,
            score_adjustments=adjustments or None,
        )
        rows = self._apply_tone_sort(rows, tone)
        rows = rows[:final_limit]

        results = [self._row_to_result(row) for row in rows]
        similarities = [r.similarity for r in results if r.similarity is not None]
        logger.info(
            "semantic_search mode=%s language=%s category=%s tone=%s rewritten=%s "
            "results=%d top_similarity=%s lowest_similarity=%s query_chars=%d",
            effective_mode,
            language,
            effective_category,
            tone,
            rewritten is not None,
            len(results),
            f"{max(similarities):.3f}" if similarities else "n/a",
            f"{min(similarities):.3f}" if similarities else "n/a",
            len(search_query),
        )
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
            logger.info("advanced_search cache=hit rewrite=%s", rewrite is not None)
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

        logger.info(
            "advanced_search cache=miss rewrite=%s api_queries=%d volumes=%d new_candidates=%d",
            rewrite_result is not None,
            len(api_queries),
            len(volumes),
            len(books),
        )

        if not books:
            self.query_cache.set(query, "advanced", category, tone, [], rewrite_result)
            return rewrite_result, query_vector

        ingested = self.catalog.upsert_books(books)
        logger.info("advanced_search upserted=%d of %d candidates", len(ingested), len(books))
        ingested_isbns = [book.isbn13 for book in ingested] if ingested else [book.isbn13 for book in books]
        self.query_cache.set(query, "advanced", category, tone, ingested_isbns, rewrite_result)
        return rewrite_result, query_vector

    def _feedback_adjustments(self, lookup: Any | None) -> dict[str, float]:
        if lookup is None:
            return {}
        adjustments = compute_adjustments(FeedbackRepository.resolve(lookup))
        if adjustments:
            logger.info(
                "feedback_applied adjusted_books=%d max_abs_delta=%.4f",
                len(adjustments),
                max(abs(delta) for delta in adjustments.values()),
            )
        return adjustments

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
