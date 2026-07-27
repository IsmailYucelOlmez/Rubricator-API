import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
GOOGLE_BOOKS_MAX_RESULTS = 20
GOOGLE_BOOKS_TIMEOUT_SECONDS = 10


class GoogleBooksClient:
    """External datasource: Google Books volumes API."""

    def __init__(self, api_key: str | None = None, timeout: int = GOOGLE_BOOKS_TIMEOUT_SECONDS):
        self.api_key = api_key if api_key is not None else settings.google_books_api_key or os.getenv(
            "GOOGLE_BOOKS_API_KEY"
        )
        self.timeout = timeout
        if not self.api_key:
            logger.warning(
                "GoogleBooksClient running without an API key; requests use the low "
                "anonymous quota and may be silently rate-limited."
            )

    def search(self, query: str, max_results: int = GOOGLE_BOOKS_MAX_RESULTS) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if len(query) > settings.max_query_length:
            query = query[: settings.max_query_length]

        params: dict[str, Any] = {
            "q": query,
            "maxResults": max_results,
            "printType": "books",
            "langRestrict": "en",
        }
        if self.api_key:
            params["key"] = self.api_key

        try:
            response = requests.get(
                GOOGLE_BOOKS_BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.warning("Google Books API timed out for query: %s", query[:80])
            return []
        except requests.RequestException as error:
            logger.warning("Google Books API error: %s", error)
            return []

        payload = response.json()
        items = payload.get("items") or []
        return [item.get("volumeInfo", {}) for item in items if item.get("volumeInfo")]

    def search_many(self, queries: list[str], max_results: int = GOOGLE_BOOKS_MAX_RESULTS) -> list[dict[str, Any]]:
        unique_queries = [query.strip() for query in queries if query.strip()]
        if not unique_queries:
            return []
        if len(unique_queries) == 1:
            return self.search(unique_queries[0], max_results=max_results)

        merged: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        with ThreadPoolExecutor(max_workers=len(unique_queries)) as executor:
            futures = {
                executor.submit(self.search, query, max_results): query for query in unique_queries
            }
            for future in as_completed(futures):
                try:
                    volumes = future.result()
                except Exception as error:
                    logger.warning("Google Books parallel search failed: %s", error)
                    continue

                for volume in volumes:
                    dedupe_key = self._volume_dedupe_key(volume)
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    merged.append(volume)

        return merged

    @staticmethod
    def _volume_dedupe_key(volume: dict[str, Any]) -> str:
        title = (volume.get("title") or "").strip().lower()
        authors = volume.get("authors") or []
        first_author = authors[0].strip().lower() if authors else ""
        return f"{title}|{first_author}"
