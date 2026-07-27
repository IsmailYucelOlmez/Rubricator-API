from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    google_books_api_key: str = ""
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    semantic_query_cache_ttl_seconds: int = 3600
    max_new_books: int = 5
    cors_origins: str = "*"
    api_key: str = ""

    embedding_model: str = "models/gemini-embedding-001"
    rewrite_model: str = "gemini-2.5-flash"
    max_query_length: int = 500
    initial_top_k: int = 50
    default_limit: int = 16
    max_limit: int = 32

    # Document chat — limits
    document_max_file_size_mb: int = 20
    document_max_pages: int = 500
    document_max_chapters: int = 1000
    document_max_words: int = 1_500_000
    document_max_chars: int = 7_500_000
    document_chunk_size: int = 1000
    document_chunk_overlap: int = 200
    document_max_chunks: int = 1500
    document_max_question_length: int = 500
    document_max_questions_per_session: int = 10
    document_retrieval_top_k: int = 5
    document_max_context_chars: int = 12_000
    document_max_chat_turns_memory: int = 4
    document_chat_model: str = "gemini-2.5-flash"
    document_chat_temperature: float = 0.2
    document_chat_max_output_tokens: int = 1024
    document_embedding_concurrency: int = 15
    document_embedding_batch_size: int = 50

    # Session TTL
    session_idle_ttl_minutes: int = 45
    session_max_ttl_minutes: int = 120
    session_cleanup_interval_seconds: int = 60

    # Optional rate limits
    document_max_concurrent_sessions: int = 25
    document_max_sessions_per_ip_hour: int = 10

    # Session store backend
    redis_url: str = ""
    session_store_backend: Literal["memory", "redis"] = "memory"

    @property
    def document_max_file_size_bytes(self) -> int:
        return self.document_max_file_size_mb * 1024 * 1024

    @property
    def document_allowed_formats(self) -> tuple[str, ...]:
        return ("pdf", "epub")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()

DATA_DIR = ROOT_DIR / "data"
