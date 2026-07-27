import logging
import re
from typing import Any

from app.data.datasources.gemini import GeminiEmbeddingClient
from app.data.datasources.supabase import fetch_all_catalog_isbns, get_supabase_client
from app.models.book_record import BookRecord

logger = logging.getLogger(__name__)


def parse_volume_id_from_thumbnail(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else None


class BookCatalogRepository:
    """Persistence: book_catalog table + semantic_search_books RPC."""

    def __init__(self, embeddings: GeminiEmbeddingClient | None = None) -> None:
        self._embeddings = embeddings or GeminiEmbeddingClient()
        self._client = get_supabase_client()

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

        rows: list[dict[str, Any]] = []
        for book, vector in zip(pending, vectors, strict=True):
            rows.append(self._book_to_row(book, vector))

        try:
            self._client.table("book_catalog").upsert(rows, on_conflict="isbn13").execute()
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
    ) -> list[dict[str, Any]]:
        rpc_result = self._client.rpc(
            "semantic_search_books",
            {
                "p_query_embedding": query_vector,
                "p_category": category,
                "p_limit": limit,
                "p_initial_k": initial_k,
            },
        ).execute()
        return rpc_result.data or []

    @staticmethod
    def _book_to_row(book: BookRecord, vector: list[float]) -> dict[str, Any]:
        return {
            "isbn13": book.isbn13,
            "isbn10": book.isbn10,
            "title": book.title,
            "authors": book.authors,
            "description": book.description,
            "thumbnail_url": book.thumbnail,
            "google_volume_id": parse_volume_id_from_thumbnail(book.thumbnail),
            "simple_category": book.simple_categories or "Unknown",
            "emotion_scores": {},
            "embedding": vector,
            "source": book.source,
        }
