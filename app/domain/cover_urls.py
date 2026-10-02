"""Cover image URL helpers shared by search results and catalog backfills."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_GOOGLE_BOOKS_HOSTS = {"books.google.com", "books.googleusercontent.com"}
_KITAPYURDU_HOST = "img.kitapyurdu.com"
_KITAPYURDU_WIDTH = re.compile(r"/wi:\d+/")

# Kitapyurdu serves any width; the scraped URLs ask for 100 px, which is blurry
# in a two-column grid on a phone.
KITAPYURDU_COVER_WIDTH = 400


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def normalize_cover_url(url: str | None) -> str | None:
    """Cleans a stored cover URL: https, and a usable width for Kitapyurdu."""
    if not url or not url.strip():
        return None
    url = url.strip()
    if url.lower().startswith("http://"):
        url = "https://" + url[len("http://") :]
    if _host(url) == _KITAPYURDU_HOST:
        url = _KITAPYURDU_WIDTH.sub(f"/wi:{KITAPYURDU_COVER_WIDTH}/", url, count=1)
    return url


def cover_for_display(thumbnail: str | None) -> str | None:
    """URL returned to clients. Only Google Books understands the `fife` size
    hint, and only as a query parameter; other hosts get the URL unchanged."""
    if not thumbnail:
        return None
    if _host(thumbnail) in _GOOGLE_BOOKS_HOSTS:
        separator = "&" if "?" in thumbnail else "?"
        return f"{thumbnail}{separator}fife=w800"
    return thumbnail
