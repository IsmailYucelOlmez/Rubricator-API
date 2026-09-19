import logging

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.data.repositories.search_log_repository import SearchLogRepository
from app.domain.search_service import SemanticSearchService
from app.models.schemas import (
    HealthResponse,
    SemanticSearchMeta,
    SemanticSearchRequest,
    SemanticSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["semantic"])
_bearer = HTTPBearer(auto_error=False)
_search_service = SemanticSearchService()
_search_logger = SearchLogRepository()


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/semantic/search", response_model=SemanticSearchResponse)
def semantic_search(
    body: SemanticSearchRequest,
    _: None = Depends(_verify_api_key),
) -> SemanticSearchResponse:
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Embedding service is not configured")

    if body.mode == "advanced" and body.language != "tr" and not settings.google_books_api_key:
        logger.warning("Advanced mode without GOOGLE_BOOKS_API_KEY; using anonymous quota")

    try:
        results, rewritten = _search_service.search(
            query=body.query,
            mode=body.mode,
            category=body.category,
            tone=body.tone,
            limit=body.limit,
            language=body.language,
            relevant_isbns=body.feedback.relevant if body.feedback else None,
            irrelevant_isbns=body.feedback.irrelevant if body.feedback else None,
        )
    except RuntimeError as error:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=503, detail=str(error)) from error

    _search_logger.log_anonymous(
        query=body.query,
        mode=body.mode,
        category=body.category,
        tone=body.tone,
        result_count=len(results),
    )

    return SemanticSearchResponse(
        results=results,
        meta=SemanticSearchMeta(
            mode=body.mode,
            queryRewritten=rewritten,
            resultCount=len(results),
        ),
    )
