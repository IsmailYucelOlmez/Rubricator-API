import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

PAUSE_SECONDS = 65
TRANSIENT_PAUSE_SECONDS = 30
MAX_RETRIES = 5
MAX_TOTAL_WAIT_SECONDS = 300

ProgressCallback = Callable[[int, int], None]


def _parse_retry_seconds(error: Exception) -> int:
    match = re.search(r"retry in ([0-9.]+)s", str(error), re.IGNORECASE)
    if match:
        return max(int(float(match.group(1))) + 1, PAUSE_SECONDS)
    return PAUSE_SECONDS


class DailyQuotaExhaustedError(RuntimeError):
    """Gemini embedding free-tier daily request limit reached."""


def _is_daily_quota_exhausted(error: Exception) -> bool:
    message = str(error)
    return (
        "embed_content_free_tier" in message
        and ("PerDay" in message or "RequestsPerDay" in message)
    )


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _is_transient_error(error: Exception) -> bool:
    message = str(error)
    return any(
        token in message
        for token in ("502", "503", "504", "Bad Gateway", "Service Unavailable", "Gateway Timeout")
    )


def _retry_wait_seconds(error: Exception) -> int:
    if _is_rate_limit_error(error):
        return _parse_retry_seconds(error)
    return TRANSIENT_PAUSE_SECONDS


def _is_retryable_error(error: Exception) -> bool:
    return _is_rate_limit_error(error) or _is_transient_error(error)


def embed_batch_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
    max_retries: int = MAX_RETRIES,
    max_total_wait: int = MAX_TOTAL_WAIT_SECONDS,
) -> list[list[float]]:
    attempts = 0
    waited = 0
    while True:
        try:
            return embeddings.embed_documents(texts, batch_size=len(texts))
        except Exception as error:
            if _is_daily_quota_exhausted(error):
                raise DailyQuotaExhaustedError(
                    "Gemini free tier günlük embedding kotası doldu (~1000/gün). "
                    "Tier 1 kullanıyorsanız .env içindeki GOOGLE_API_KEY'in doğru anahtar "
                    "olduğunu AI Studio'dan doğrulayın."
                ) from error
            if not _is_retryable_error(error):
                raise
            attempts += 1
            wait = _retry_wait_seconds(error)
            if attempts > max_retries or waited + wait > max_total_wait:
                logger.warning(
                    "Giving up embedding after %d retryable attempt(s) (~%ds waited).",
                    attempts,
                    waited,
                )
                raise
            reason = "Rate limited" if _is_rate_limit_error(error) else "Transient API error"
            logger.warning(
                "%s — waiting %ds before retry (%d/%d)...",
                reason,
                wait,
                attempts,
                max_retries,
            )
            time.sleep(wait)
            waited += wait


def embed_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
    max_retries: int = MAX_RETRIES,
    max_total_wait: int = MAX_TOTAL_WAIT_SECONDS,
    concurrency: int | None = None,
    batch_size: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[list[float]]:
    """Embed texts with parallel batch requests and per-batch retry."""
    if not texts:
        return []

    effective_batch_size = max(1, batch_size or settings.document_embedding_batch_size)
    effective_concurrency = max(1, concurrency or settings.document_embedding_concurrency)
    batches = [
        texts[index : index + effective_batch_size]
        for index in range(0, len(texts), effective_batch_size)
    ]

    if len(batches) == 1:
        vectors = embed_batch_with_retry(
            embeddings,
            batches[0],
            max_retries=max_retries,
            max_total_wait=max_total_wait,
        )
        if on_progress:
            on_progress(len(texts), len(texts))
        return vectors

    batch_results: list[list[list[float]] | None] = [None] * len(batches)
    completed = 0

    def _run_batch(batch_index: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        vectors = embed_batch_with_retry(
            embeddings,
            batch,
            max_retries=max_retries,
            max_total_wait=max_total_wait,
        )
        return batch_index, vectors

    workers = min(effective_concurrency, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_batch, batch_index, batch)
            for batch_index, batch in enumerate(batches)
        ]
        for future in as_completed(futures):
            batch_index, vectors = future.result()
            batch_results[batch_index] = vectors
            completed += len(vectors)
            if on_progress:
                on_progress(completed, len(texts))

    ordered: list[list[float]] = []
    for batch_vectors in batch_results:
        if batch_vectors is None:
            raise RuntimeError("Embedding batch failed unexpectedly")
        ordered.extend(batch_vectors)
    return ordered


class GeminiEmbeddingClient:
    """External datasource: Google Gemini embeddings API."""

    def __init__(self) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=768,
        )

    def embed_query(self, query: str) -> list[float] | None:
        query = query.strip()
        if not query:
            return None
        try:
            return self._embeddings.embed_query(query)
        except Exception as error:
            logger.warning("Query embedding failed for %r: %s", query[:80], error)
            return None

    def embed_documents(
        self,
        texts: list[str],
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        return embed_with_retry(self._embeddings, texts, on_progress=on_progress)
