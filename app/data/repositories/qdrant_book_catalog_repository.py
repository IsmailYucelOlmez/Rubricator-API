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
        use_mmr = settings.search_mmr_lambda < 1.0
        # With MMR we need a candidate pool larger than `limit` to diversify from.
        result_limit = max(initial_k, limit) if use_mmr else min(limit, initial_k)
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
        points = response.points
        if use_mmr:
            points = self._diversify(points, limit)
        return [self._point_to_row(point) for point in points]

    @staticmethod
    def _diversify(points: list[Any], limit: int) -> list[Any]:
        """MMR over the candidate pool; falls back to plain top-`limit` if vectors are missing."""
        if len(points) <= limit:
            return points
        vectors = [point.vector for point in points]
        if any(not isinstance(vector, list) for vector in vectors):
            return points[:limit]
        picked = mmr_select(
            vectors,
            [point.score for point in points],
            limit,
            settings.search_mmr_lambda,
        )
        return [points[index] for index in picked]

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
