import logging
from typing import Any

from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.mmr import mmr_select
from app.data.datasources.gemini import GeminiEmbeddingClient
from app.data.datasources.qdrant import fetch_all_catalog_isbns, get_qdrant_client, isbn_to_point_id
from app.data.repositories.book_catalog_repository import parse_volume_id_from_thumbnail
from app.models.book_record import BookRecord

logger = logging.getLogger(__name__)


class QdrantBookCatalogRepository:
    """Persistence: Qdrant book_catalog collection."""

    def __init__(self, embeddings: GeminiEmbeddingClient | None = None) -> None:
        self._embeddings = embeddings or GeminiEmbeddingClient()
        self._client = get_qdrant_client()

    def list_existing_isbns(self) -> set[str]:
        try:
            return fetch_all_catalog_isbns(self._client)
        except Exception as error:
            logger.warning("Failed to load catalog ISBNs: %s", error)
            return set()

    def upsert_books(self, books: list[BookRecord]) -> list[BookRecord]:
        if not books:
            return []

        existing = self.list_existing_isbns()
        pending = [book for book in books if book.isbn13 not in existing]
        if not pending:
            return []

        texts = [book.tagged_description for book in pending]
        try:
            vectors = self._embeddings.embed_documents(texts)
        except Exception as error:
            logger.warning("Embedding failed for dynamic books: %s", error)
            return []

        points = [
            self._book_to_point(book, vector)
            for book, vector in zip(pending, vectors, strict=True)
        ]

        try:
            self._client.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
                wait=True,
            )
        except Exception as error:
            logger.warning("Catalog upsert failed: %s", error)
            return []

        return pending

    def search_by_embedding(
        self,
        query_vector: list[float],
        category: str | None,
        limit: int,
        initial_k: int,
        language: str | None = None,
        score_adjustments: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[qmodels.FieldCondition] = []
        if category:
            conditions.append(
                qmodels.FieldCondition(key="simple_category", match=qmodels.MatchValue(value=category))
            )
        if language:
            conditions.append(
                qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=language))
            )
        query_filter = qmodels.Filter(must=conditions) if conditions else None

        limit = max(limit, 1)
        adjustments = score_adjustments or {}
        use_mmr = settings.search_mmr_lambda < 1.0
        # MMR and feedback both need a candidate pool larger than `limit`: MMR to
        # diversify from, feedback so a voted-up book below the cutoff can move in.
        result_limit = (
            max(initial_k, limit) if (use_mmr or adjustments) else min(limit, initial_k)
        )
        response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=result_limit,
            with_payload=True,
            with_vectors=use_mmr,
            # Without explicit search_params, Qdrant skips rescoring the INT8-quantized
            # ANN candidates against full-precision vectors, which measurably degrades
            # top-k quality (verified empirically against exact/brute-force search).
            search_params=qmodels.SearchParams(
                hnsw_ef=128,
                quantization=qmodels.QuantizationSearchParams(rescore=True, oversampling=2.0),
            ),
        )
        points = self._select(response.points, limit, adjustments)
        return [self._point_to_row(point) for point in points]

    @staticmethod
    def _select(points: list[Any], limit: int, adjustments: dict[str, float]) -> list[Any]:
        """Pick `limit` points from the candidate pool.

        Relevance is the vector score plus any feedback adjustment (the score reported
        to clients stays the raw vector score). With MMR on and vectors present the
        pool is diversified; otherwise it is plain top-`limit` by adjusted relevance.
        """
        relevance = [
            point.score + adjustments.get(str((point.payload or {}).get("isbn13")), 0.0)
            for point in points
        ]
        by_relevance = sorted(range(len(points)), key=lambda i: relevance[i], reverse=True)

        vectors = [point.vector for point in points]
        can_diversify = (
            settings.search_mmr_lambda < 1.0
            and len(points) > limit
            and all(isinstance(vector, list) for vector in vectors)
        )
        if not can_diversify:
            return [points[i] for i in by_relevance[:limit]]

        picked = mmr_select(vectors, relevance, limit, settings.search_mmr_lambda)
        return [points[i] for i in picked]

    @staticmethod
    def _point_to_row(point: Any) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "isbn13": payload.get("isbn13"),
            "title": payload.get("title"),
            "authors": payload.get("authors"),
            "description": payload.get("description"),
            "thumbnail_url": payload.get("thumbnail_url"),
            "simple_category": payload.get("simple_category"),
            "emotion_scores": payload.get("emotion_scores") or {},
            "similarity": point.score,
            "source": payload.get("source"),
            "language": payload.get("language"),
        }

    @staticmethod
    def _book_to_point(book: BookRecord, vector: list[float]) -> qmodels.PointStruct:
        return qmodels.PointStruct(
            id=isbn_to_point_id(book.isbn13),
            vector=vector,
            payload={
                "isbn13": book.isbn13,
                "isbn10": book.isbn10,
                "title": book.title,
                "authors": book.authors,
                "description": book.description,
                "thumbnail_url": book.thumbnail,
                "google_volume_id": parse_volume_id_from_thumbnail(book.thumbnail),
                "simple_category": book.simple_categories or "Unknown",
                "emotion_scores": {},
                "source": book.source,
                # BookRecord has no language field; the Supabase column defaulted
                # dynamically-ingested rows to 'en' — Qdrant has no column defaults,
                # so this must be written explicitly or language="tr" filtering
                # would silently drop these rows.
                "language": "en",
            },
        )
