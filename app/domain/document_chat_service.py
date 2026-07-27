import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.data.datasources.gemini import DailyQuotaExhaustedError, GeminiEmbeddingClient
from app.data.session_store import get_session_store
from app.data.session_store.base import compute_expires_at
from app.domain.document_chunker import DocumentChunker
from app.domain.document_extractors.epub_extractor import EpubExtractor
from app.domain.document_extractors.pdf_extractor import PdfExtractor
from app.models.document_session import ChatTurn, DocumentChunk, DocumentSession

logger = logging.getLogger(__name__)

CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

ANSWER_SYSTEM_PROMPT = """You answer questions about the uploaded document "{filename}" using ONLY the provided document excerpts.
- Treat the excerpts as coming from that document alone.
- Ignore publisher ads, catalogs, "also by this author" blurbs, or other book promotions if they appear in the excerpts.
- If the answer is not in the context, say you don't know — do not invent a different book or plot.
- Keep direct quotes under 2 sentences; prefer paraphrase.
- Respond in the same language as the user's question."""

EXCERPT_MAX_CHARS = 200


@dataclass
class ChatAnswer:
    answer: str
    sources: list[dict]
    session_expires_at: datetime
    questions_remaining: int


class SessionNotReadyError(RuntimeError):
    """Raised when chat is attempted before embeddings finish."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class DocumentChatService:
    def __init__(self) -> None:
        self._embedding_client = GeminiEmbeddingClient()
        self._chunker = DocumentChunker()
        self._pdf_extractor = PdfExtractor()
        self._epub_extractor = EpubExtractor()
        self._session_store = get_session_store()
        self._chat_llm = ChatGoogleGenerativeAI(
            model=settings.document_chat_model,
            temperature=settings.document_chat_temperature,
            max_output_tokens=settings.document_chat_max_output_tokens,
            google_api_key=settings.google_api_key,
        )

    def create_pending_session(
        self,
        filename: str,
        doc_format: str,
    ) -> DocumentSession:
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        expires_at = compute_expires_at(
            created_at=now,
            last_accessed_at=now,
            idle_ttl_minutes=settings.session_idle_ttl_minutes,
            max_ttl_minutes=settings.session_max_ttl_minutes,
        )
        session = DocumentSession(
            session_id=session_id,
            format=doc_format,  # type: ignore[arg-type]
            filename=filename,
            created_at=now,
            expires_at=expires_at,
            last_accessed_at=now,
            page_count=None,
            chapter_count=None,
            word_count=0,
            chunk_count=0,
            chunks=[],
            status="processing",
            chunks_total=0,
            chunks_embedded=0,
        )
        self._session_store.put(session)
        logger.info("session_pending format=%s session_id=%s", doc_format, session_id)
        return session

    def process_session(
        self,
        session_id: str,
        data: bytes,
        doc_format: str,
    ) -> None:
        session = self._session_store.get(session_id)
        if session is None:
            logger.warning("process_session skipped; session gone session_id=%s", session_id)
            return

        try:
            if doc_format == "pdf":
                extraction = self._pdf_extractor.extract(data)
                page_count = extraction.page_count
                chapter_count = None
                segments = extraction.segments
                word_count = extraction.word_count
                extract_truncated = extraction.truncated
            else:
                extraction = self._epub_extractor.extract(data)
                page_count = None
                chapter_count = extraction.chapter_count
                segments = extraction.segments
                word_count = extraction.word_count
                extract_truncated = extraction.truncated

            split_result = self._chunker.split_segments(segments)
            if not split_result.items:
                raise ValueError("Could not extract text from document")

            session = self._session_store.get(session_id)
            if session is None:
                return

            session.page_count = page_count
            session.chapter_count = chapter_count
            session.word_count = word_count
            session.truncated = extract_truncated or split_result.truncated
            session.chunks_total = len(split_result.items)
            session.chunks_embedded = 0
            session.status = "processing"
            self._session_store.put(session)

            texts = [item[0] for item in split_result.items]

            def _on_progress(done: int, total: int) -> None:
                current = self._session_store.get(session_id)
                if current is None:
                    return
                current.chunks_embedded = done
                current.chunks_total = total
                self._session_store.put(current)

            try:
                embeddings = self._embedding_client.embed_documents(
                    texts,
                    on_progress=_on_progress,
                )
            except DailyQuotaExhaustedError:
                raise
            except Exception as error:
                raise RuntimeError("Embedding service is not configured") from error

            chunks = self._chunker.build_chunks(split_result.items, embeddings)
            session = self._session_store.get(session_id)
            if session is None:
                return

            session.chunks = chunks
            session.chunk_count = len(chunks)
            session.chunks_embedded = len(chunks)
            session.chunks_total = len(chunks)
            session.status = "ready"
            session.error_message = None
            self._session_store.touch(session)
            logger.info(
                "session_ready format=%s chunk_count=%d truncated=%s session_id=%s",
                doc_format,
                len(chunks),
                split_result.truncated,
                session_id,
            )
        except Exception as error:
            logger.exception("session_failed session_id=%s", session_id)
            session = self._session_store.get(session_id)
            if session is None:
                return
            session.status = "failed"
            session.error_message = str(error)
            session.chunks = []
            session.chunk_count = 0
            self._session_store.put(session)

    def create_session(
        self,
        data: bytes,
        filename: str,
        doc_format: str,
    ) -> DocumentSession:
        """Synchronous create (extract + embed). Prefer create_pending + process_session."""
        session = self.create_pending_session(filename=filename, doc_format=doc_format)
        self.process_session(session.session_id, data=data, doc_format=doc_format)
        ready = self._session_store.get(session.session_id)
        if ready is None:
            raise RuntimeError("Session disappeared during processing")
        if ready.status == "failed":
            detail = ready.error_message or "Session processing failed"
            if "Could not" in detail or "exceeds" in detail or "extract" in detail.lower():
                raise ValueError(detail)
            if "quota" in detail.lower() or "Embedding" in detail:
                raise RuntimeError(detail)
            raise RuntimeError(detail)
        return ready

    def get_session(self, session_id: str) -> DocumentSession | None:
        return self._session_store.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._session_store.delete(session_id)

    def chat(self, session_id: str, question: str) -> ChatAnswer:
        session = self._session_store.get(session_id)
        if session is None:
            raise LookupError("Session not found or expired")

        if session.status == "processing":
            raise SessionNotReadyError(
                "processing",
                "Session is still processing; poll GET /sessions/{id} until status is ready",
            )
        if session.status == "failed":
            raise SessionNotReadyError(
                "failed",
                session.error_message or "Session processing failed",
            )

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if len(question) > settings.document_max_question_length:
            raise ValueError(
                f"Question exceeds {settings.document_max_question_length} character limit"
            )

        if session.question_count >= settings.document_max_questions_per_session:
            raise OverflowError("Question limit reached for this session")

        standalone_question = self._condense_question(session, question)
        query_embedding = self._embedding_client.embed_query(standalone_question)
        if query_embedding is None:
            raise RuntimeError("Embedding service is not configured")

        retrieved = self._retrieve_chunks(session.chunks, query_embedding)
        context = self._build_context(retrieved)
        answer = self._generate_answer(
            question,
            context,
            filename=session.filename or "document",
        )

        now = datetime.now(timezone.utc)
        session.chat_turns.append(ChatTurn(role="user", content=question, created_at=now))
        session.chat_turns.append(ChatTurn(role="assistant", content=answer, created_at=now))
        session.chat_turns = session.chat_turns[-settings.document_max_chat_turns_memory :]
        session.question_count += 1
        self._session_store.touch(session)

        logger.info(
            "document_chat question_count=%d session_id=%s",
            session.question_count,
            session_id,
        )

        return ChatAnswer(
            answer=answer,
            sources=[self._format_source(chunk) for chunk in retrieved],
            session_expires_at=session.expires_at,
            questions_remaining=settings.document_max_questions_per_session
            - session.question_count,
        )

    def _condense_question(self, session: DocumentSession, question: str) -> str:
        if not session.chat_turns:
            return question

        history_lines = []
        for turn in session.chat_turns:
            label = "Human" if turn.role == "user" else "Assistant"
            history_lines.append(f"{label}: {turn.content}")
        chat_history = "\n".join(history_lines)

        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=chat_history,
            question=question,
        )
        try:
            response = self._chat_llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            standalone = str(content).strip()
            return standalone or question
        except Exception as error:
            logger.warning("Condense question failed, using original: %s", error)
            return question

    def _retrieve_chunks(
        self,
        chunks: list[DocumentChunk],
        query_embedding: list[float],
    ) -> list[DocumentChunk]:
        scored = [
            (self._cosine_similarity(query_embedding, chunk.embedding), chunk)
            for chunk in chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = scored[: settings.document_retrieval_top_k]
        return [chunk for _, chunk in top_k]

    def _build_context(self, chunks: list[DocumentChunk]) -> str:
        parts: list[str] = []
        total_chars = 0
        for chunk in chunks:
            piece = chunk.content.strip()
            if not piece:
                continue
            if total_chars + len(piece) > settings.document_max_context_chars:
                remaining = settings.document_max_context_chars - total_chars
                if remaining <= 0:
                    break
                piece = piece[:remaining]
            parts.append(piece)
            total_chars += len(piece)
            if total_chars >= settings.document_max_context_chars:
                break
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str, filename: str) -> str:
        system = ANSWER_SYSTEM_PROMPT.format(filename=filename)
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Document filename: {filename}\n\n"
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {question}"
                )
            ),
        ]
        response = self._chat_llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()

    def _format_source(self, chunk: DocumentChunk) -> dict:
        excerpt = chunk.content.strip()
        if len(excerpt) > EXCERPT_MAX_CHARS:
            excerpt = excerpt[:EXCERPT_MAX_CHARS].rstrip() + "..."
        sentences = re.split(r"(?<=[.!?])\s+", excerpt)
        if len(sentences) > 2:
            excerpt = " ".join(sentences[:2])
        return {
            "chunkIndex": chunk.index,
            "excerpt": excerpt,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
