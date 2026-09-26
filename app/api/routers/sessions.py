import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.data.session_store import get_session_store
from app.domain.document_chat_service import DocumentChatService, SessionNotReadyError
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
    CreateSessionResponse,
    SessionLimitsResponse,
    SessionStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["document-chat"])
_bearer = HTTPBearer(auto_error=False)
_chat_service = DocumentChatService()
_session_times: dict[str, list[datetime]] = defaultdict(list)

# The Supabase edge function (the only caller in production) authenticates the
# end user and forwards the verified id here. Every request therefore arrives
# from that function's egress IP, so an IP-keyed limit would be one bucket shared
# by all users. Trust is anchored in the API key: only the edge function has it.
USER_ID_HEADER = "x-user-id"
_USER_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

PDF_MAGIC = b"%PDF"
EPUB_MIME_TYPES = {
    "application/epub+zip",
    "application/x-epub+zip",
    "application/octet-stream",
}
PDF_MIME_TYPES = {"application/pdf", "application/octet-stream"}


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _detect_format(filename: str, content_type: str | None, data: bytes) -> str | None:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf" or (content_type in PDF_MIME_TYPES and data.startswith(PDF_MAGIC)):
        if data.startswith(PDF_MAGIC):
            return "pdf"
    if extension == "epub" or content_type in EPUB_MIME_TYPES:
        if data[:2] == b"PK":
            return "epub"
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data[:2] == b"PK" and extension == "epub":
        return "epub"
    return None


def _rate_limit_subject(request: Request) -> tuple[str, int, str]:
    """Who a new session is counted against: (bucket key, hourly limit, label).

    Prefers the account id forwarded by the edge function; falls back to the
    caller's IP for requests without a (well-formed) id, e.g. direct calls.
    """
    user_id = (request.headers.get(USER_ID_HEADER) or "").strip()
    if _USER_ID_RE.match(user_id):
        return (
            f"user:{user_id.lower()}",
            settings.document_max_sessions_per_user_hour,
            "account",
        )
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}", settings.document_max_sessions_per_ip_hour, "IP address"


def _check_rate_limit(key: str, limit: int, label: str) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    recent = [timestamp for timestamp in _session_times.get(key, []) if timestamp > cutoff]
    if recent:
        _session_times[key] = recent
    else:
        # Drop idle buckets so the dict doesn't grow with every caller ever seen.
        _session_times.pop(key, None)
    if len(recent) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Too many sessions created from this {label}",
        )


def _record_session(key: str) -> None:
    _session_times[key].append(datetime.now(timezone.utc))


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    _: None = Depends(_verify_api_key),
) -> CreateSessionResponse:
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Embedding service is not configured")

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Reject on the spooled size first: read() would load the whole upload into
    # memory as bytes just to then find it too large.
    if file.size is not None and file.size > settings.document_max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.document_max_file_size_mb} MB limit",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="No file provided")

    if len(data) > settings.document_max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.document_max_file_size_mb} MB limit",
        )

    doc_format = _detect_format(file.filename, file.content_type, data)
    if doc_format is None:
        raise HTTPException(
            status_code=415,
            detail="Unsupported format; use PDF or EPUB",
        )

    session_store = get_session_store()
    if session_store.count() >= settings.document_max_concurrent_sessions:
        raise HTTPException(status_code=503, detail="Server session capacity reached")

    limit_key, limit, limit_label = _rate_limit_subject(request)
    _check_rate_limit(limit_key, limit, limit_label)

    session = _chat_service.create_pending_session(
        filename=file.filename,
        doc_format=doc_format,
    )
    background_tasks.add_task(
        _chat_service.process_session,
        session.session_id,
        data,
        doc_format,
    )
    _record_session(limit_key)

    return CreateSessionResponse(
        sessionId=session.session_id,
        format=session.format,
        filename=session.filename,
        expiresAt=session.expires_at,
        status=session.status,
        pageCount=session.page_count,
        chapterCount=session.chapter_count,
        wordCount=session.word_count,
        chunkCount=session.chunk_count,
        truncated=session.truncated,
        limits=SessionLimitsResponse(
            maxQuestionsRemaining=settings.document_max_questions_per_session
            - session.question_count,
        ),
    )


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat_with_session(
    session_id: str,
    body: ChatRequest,
    _: None = Depends(_verify_api_key),
) -> ChatResponse:
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Embedding service is not configured")

    try:
        result = _chat_service.chat(session_id=session_id, question=body.question)
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Session not found or expired") from error
    except SessionNotReadyError as error:
        status_code = (
            status.HTTP_409_CONFLICT
            if error.status == "processing"
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except OverflowError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document chat failed")
        raise HTTPException(status_code=503, detail="Embedding service is not configured") from error

    return ChatResponse(
        answer=result.answer,
        sources=[ChatSourceResponse(**source) for source in result.sources],
        sessionExpiresAt=result.session_expires_at,
        questionsRemaining=result.questions_remaining,
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session_status(
    session_id: str,
    _: None = Depends(_verify_api_key),
) -> SessionStatusResponse:
    session = _chat_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    get_session_store().touch(session)

    return SessionStatusResponse(
        sessionId=session.session_id,
        format=session.format,
        expiresAt=session.expires_at,
        chunkCount=session.chunk_count,
        questionCount=session.question_count,
        questionsRemaining=settings.document_max_questions_per_session
        - session.question_count,
        status=session.status,
        errorMessage=session.error_message,
        chunksEmbedded=session.chunks_embedded,
        chunksTotal=session.chunks_total,
        pageCount=session.page_count,
        chapterCount=session.chapter_count,
        wordCount=session.word_count,
        truncated=session.truncated,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    _: None = Depends(_verify_api_key),
) -> None:
    if not _chat_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
