import logging

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.data.datasources.kitapyurdu_relay import DisallowedHostError, fetch
from app.models.schemas import RelayFetchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["scrape-relay"])
_bearer = HTTPBearer(auto_error=False)


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/scrape-relay/fetch", response_model=RelayFetchResponse)
def fetch_relay(
    url: str = Query(..., min_length=1),
    _: None = Depends(_verify_api_key),
) -> RelayFetchResponse:
    """Fetches an allowlisted bookstore URL server-side and returns its raw
    HTML. Used by Supabase's scrape-tr-books edge function, whose own
    (Deno Deploy) egress IP range is blocked by Kitapyurdu's WAF — this
    service's IP isn't, so it fetches on that function's behalf while all
    HTML parsing stays in the already-tested Deno code.
    """
    try:
        status_code, html = fetch(url)
    except DisallowedHostError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except requests.Timeout as error:
        raise HTTPException(status_code=504, detail="Upstream request timed out") from error
    except requests.RequestException as error:
        logger.warning("Kitapyurdu relay fetch failed for %s: %s", url, error)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {error}") from error

    return RelayFetchResponse(status=status_code, html=html)
