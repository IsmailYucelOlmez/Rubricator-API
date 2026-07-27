from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: Literal["simple", "advanced"] = "simple"
    category: str = "All"
    tone: str = "All"
    limit: int = Field(default=16, ge=1, le=32)


class SemanticBookResult(BaseModel):
    isbn13: str
    title: str
    author: str
    description: str
    coverImageUrl: str | None = None
    category: str | None = None
    similarity: float | None = None
    source: str = "local"


class SemanticSearchMeta(BaseModel):
    mode: str
    queryRewritten: str | None = None
    resultCount: int


class SemanticSearchResponse(BaseModel):
    results: list[SemanticBookResult]
    meta: SemanticSearchMeta


class HealthResponse(BaseModel):
    status: str = "ok"


class SessionLimitsResponse(BaseModel):
    maxQuestionsRemaining: int


class CreateSessionResponse(BaseModel):
    sessionId: str
    format: Literal["pdf", "epub"]
    filename: str
    expiresAt: datetime
    status: Literal["processing", "ready", "failed"] = "processing"
    pageCount: int | None = None
    chapterCount: int | None = None
    wordCount: int = 0
    chunkCount: int = 0
    truncated: bool = False
    limits: SessionLimitsResponse


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatSourceResponse(BaseModel):
    chunkIndex: int
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSourceResponse]
    sessionExpiresAt: datetime
    questionsRemaining: int


class SessionStatusResponse(BaseModel):
    sessionId: str
    format: Literal["pdf", "epub"]
    expiresAt: datetime
    chunkCount: int
    questionCount: int
    questionsRemaining: int
    status: Literal["processing", "ready", "failed"] = "ready"
    errorMessage: str | None = None
    chunksEmbedded: int = 0
    chunksTotal: int = 0
    pageCount: int | None = None
    chapterCount: int | None = None
    wordCount: int = 0
    truncated: bool = False


class RewriteResultModel(BaseModel):
    detected_language: str = ""
    english_summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    google_books_queries: list[str] = Field(default_factory=list)
    genre_hint: Literal["fiction", "nonfiction", "unknown"] = "unknown"

    @field_validator("keywords", "google_books_queries", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def effective_local_query(self, fallback: str) -> str:
        summary = self.english_summary.strip()
        return summary if summary else fallback.strip()

    def effective_api_queries(self, fallback: str) -> list[str]:
        from app.core.config import settings

        queries = [query.strip() for query in self.google_books_queries if query.strip()]
        if queries:
            return queries[:2]

        keywords = [keyword.strip() for keyword in self.keywords if keyword.strip()]
        if keywords:
            return [" ".join(keywords[:6])]

        fallback = fallback.strip()
        return [fallback] if fallback else []
