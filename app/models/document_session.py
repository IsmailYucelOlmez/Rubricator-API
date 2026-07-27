from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SessionStatus = Literal["processing", "ready", "failed"]


@dataclass
class DocumentChunk:
    index: int
    content: str
    embedding: list[float]
    metadata: dict


@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


@dataclass
class DocumentSession:
    session_id: str
    format: Literal["pdf", "epub"]
    filename: str
    created_at: datetime
    expires_at: datetime
    last_accessed_at: datetime
    page_count: int | None
    chapter_count: int | None
    word_count: int
    chunk_count: int
    chunks: list[DocumentChunk]
    chat_turns: list[ChatTurn] = field(default_factory=list)
    question_count: int = 0
    truncated: bool = False
    status: SessionStatus = "ready"
    error_message: str | None = None
    chunks_total: int = 0
    chunks_embedded: int = 0
