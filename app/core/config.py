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

    environment: Literal["development", "production"] = "development"

    google_api_key: str = ""
    google_books_api_key: str = ""
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    cluster_endpoint: str = ""
    cluster_api_key: str = ""
    qdrant_collection: str = "book_catalog"
    catalog_backend: Literal["supabase", "qdrant"] = "supabase"
    semantic_query_cache_ttl_seconds: int = 3600
    max_new_books: int = 5
    cors_origins: str = "*"
    api_key: str = ""

    embedding_model: str = "models/gemini-embedding-001"
    # Process-wide cap on in-flight Gemini embedding batch requests, shared by all
    # document sessions, catalog ingests and scripts (per-session pools stay separate).
    embedding_max_concurrency: int = 10
    # Process-wide pacing of embedding batch requests; 0 = unlimited. Set to your
    # Gemini tier's embedding requests-per-minute to avoid 429 bursts.
    embedding_requests_per_minute: int = 0
    rewrite_model: str = "gemini-2.5-flash"
    description_model: str = "gemini-2.5-flash"
    description_temperature: float = 0.6
    description_max_output_tokens: int = 400
    max_query_length: int = 500
    initial_top_k: int = 50
    default_limit: int = 16
    max_limit: int = 32
    # In-session query refinement (Rocchio) from relevant/irrelevant marks in a search request.
    refine_relevant_weight: float = 0.5
    refine_irrelevant_weight: float = 0.3
    # Relevance feedback re-ranking: off until enough votes exist to be worth applying.
    feedback_rerank_enabled: bool = False
    # Largest score shift a book can get from votes (cosine similarity units).
    feedback_weight: float = 0.05
    # Shrinks tiny vote counts toward 0: adj = (up - down) / (up + down + prior).
    feedback_prior_strength: float = 5.0
    # Fewer votes than this on a (query, book) pair are ignored entirely.
    feedback_min_votes: int = 3
    # Relevance-feedback vote lookup (Supabase get_semantic_feedback RPC).
    feedback_cache_ttl_seconds: int = 60
    # Longest a search waits for votes; on timeout it proceeds without them.
    feedback_lookup_timeout_seconds: float = 0.15
    # Book search MMR (Qdrant backend): fetch initial_top_k candidates, diversify down to limit.
    # 1.0 = pure relevance (MMR off).
    search_mmr_lambda: float = 0.7

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
    # Used instead of document_retrieval_top_k for summary/analysis-style questions.
    document_retrieval_top_k_complex: int = 10
    # MMR diversity for chunk selection: 1.0 = pure relevance (MMR off), lower = more diverse.
    document_mmr_lambda: float = 0.7
    # MMR re-ranks the best top_k * multiplier relevant chunks down to top_k.
    document_mmr_fetch_multiplier: int = 4
    # Cosine similarity floor for a chunk to be considered relevant. Gemini
    # embedding similarities for genuinely on-topic excerpts typically land
    # well above this; it's set conservatively low to avoid dropping true
    # matches and should be tightened once real query/score logs are
    # available (see docs/retrieval-eval plan).
    document_retrieval_min_score: float = 0.3
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

    def ensure_production_ready(self) -> None:
        """Fail fast instead of silently serving LLM/embedding endpoints without auth."""
        if self.environment == "production" and not self.api_key:
            raise RuntimeError(
                "API_KEY must be set when ENVIRONMENT=production — otherwise the "
                "LLM/embedding-backed endpoints are open with no authentication. "
                "Set API_KEY, or set ENVIRONMENT=development for local/unauthenticated use."
            )

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
