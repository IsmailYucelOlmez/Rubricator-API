import logging

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.domain.description_generator import TrbookDescriptionGenerator
from app.models.schemas import TrbookDescriptionRequest, TrbookDescriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["trbooks"])
_bearer = HTTPBearer(auto_error=False)
_description_generator = TrbookDescriptionGenerator()


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/trbooks/generate-description", response_model=TrbookDescriptionResponse)
def generate_description(
    body: TrbookDescriptionRequest,
    _: None = Depends(_verify_api_key),
) -> TrbookDescriptionResponse:
    """Used by the Flutter "add a Turkish book" form's optional AI-description
    button (bookapp's trbooks feature) — title/author/ISBN in, a short
    spoiler-free Turkish description out. Never persists anything; the
    Flutter client stores the result in `trbooks.description` itself via the
    `submit_user_trbook` Supabase RPC.
    """
    if not settings.google_api_key:
        raise HTTPException(
            status_code=503,
            detail="Description generation service is not configured",
        )

    try:
        description = _description_generator.generate(
            title=body.title,
            author=body.author,
            isbn=body.isbn,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        logger.warning(
            "Description generation failed for %r by %r: %s",
            body.title,
            body.author,
            error,
        )
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Description generation crashed")
        raise HTTPException(status_code=503, detail="Description generation failed") from error

    return TrbookDescriptionResponse(description=description)
