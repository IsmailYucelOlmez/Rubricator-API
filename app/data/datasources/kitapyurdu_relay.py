import logging
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Kitapyurdu's WAF blocks Supabase Edge Functions' (Deno Deploy) datacenter
# IP range outright (confirmed: identical requests get 403 from there but
# 200 from this service's own IP). This datasource lets scrape-tr-books
# relay its fetches through here instead. Locked to this one host — an
# authenticated arbitrary-URL fetch would otherwise be an SSRF/open-proxy
# risk if the shared API key ever leaked.
ALLOWED_HOSTS = {"www.kitapyurdu.com", "kitapyurdu.com"}
TIMEOUT_SECONDS = 15

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.kitapyurdu.com/",
}


class DisallowedHostError(ValueError):
    """Raised when the requested URL's host isn't on the allowlist."""


def is_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def fetch(url: str) -> tuple[int, str]:
    """Fetches [url] with browser-like headers and returns (status_code, body).

    Never raises for a non-2xx upstream response — callers (the Deno
    scrape-tr-books function) decide what a given status code means, mirroring
    how it already handles a direct fetch. Raises [DisallowedHostError] for a
    URL outside [ALLOWED_HOSTS], and lets network-level exceptions
    (timeout, connection error) propagate as-is.
    """
    if not is_allowed(url):
        raise DisallowedHostError(f"Host not allowed for relay: {url}")

    response = requests.get(url, headers=_HEADERS, timeout=TIMEOUT_SECONDS)
    return response.status_code, response.text
