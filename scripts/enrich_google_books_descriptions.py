"""
Fill in missing descriptions using the Google Books API, keyed by ISBN13.

Usage:
  python scripts/enrich_google_books_descriptions.py \
      --csv "books.csv" --isbn-column isbn \
      --existing-csv data/tr_books_cleaned.csv --existing-csv data/kitapyurdu_descriptions.csv \
      --existing-csv data/openlibrary_descriptions.csv \
      --out data/google_books_descriptions.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from app.core.config import settings  # noqa: E402
from app.domain.book_normalizer import clean_description  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_FIELDS = [
    "isbn13",
    "title",
    "authors",
    "description",
    "simple_categories",
    "published_year",
    "source",
    "language",
]
MIN_DESCRIPTION_LENGTH = 30
API_URL = "https://www.googleapis.com/books/v1/volumes"


class QuotaExceeded(Exception):
    pass


def load_isbn_set(csv_path: Path, column: str) -> set[str]:
    if not csv_path.exists():
        return set()
    isbns: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            isbn = (row.get(column) or "").strip()
            if isbn:
                isbns.add(isbn)
    return isbns


def load_pending_rows(csv_path: Path, isbn_column: str, skip: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            isbn = (row.get(isbn_column) or "").strip()
            if not isbn.isdigit() or len(isbn) != 13:
                continue
            if isbn in skip or isbn in seen:
                continue
            seen.add(isbn)
            rows.append(row)
    return rows


def parse_year(published_date: str | None) -> int | None:
    if not published_date:
        return None
    digits = published_date[:4]
    if digits.isdigit():
        year = int(digits)
        if 1500 <= year <= 2100:
            return year
    return None


def fetch_one(session: requests.Session, api_key: str, isbn: str, timeout: float) -> dict[str, Any]:
    params = {"q": f"isbn:{isbn}", "key": api_key}
    response = session.get(API_URL, params=params, timeout=timeout)

    if response.status_code == 403:
        body = response.text.lower()
        if "quota" in body or "rate" in body:
            raise QuotaExceeded(response.text[:300])

    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        return {"found": False}

    volume_info = items[0].get("volumeInfo", {})
    description = clean_description(volume_info.get("description") or "")
    return {"found": True, "volume_info": volume_info, "description": description}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich missing descriptions via the Google Books API")
    parser.add_argument("--csv", required=True, type=Path, help="Source CSV to read title/author/isbn from")
    parser.add_argument("--isbn-column", default="isbn13")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--author-column", default="authors")
    parser.add_argument("--existing-csv", action="append", default=[], type=Path, help="Skip ISBNs already present (repeatable)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between submissions (seconds)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--breaker-window", type=int, default=100, help="Stop if this many consecutive results all error out (not just 'no match')")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    api_key = settings.google_books_api_key
    if not api_key:
        raise SystemExit("GOOGLE_BOOKS_API_KEY is required (set it in .env)")

    skip: set[str] = set()
    for existing in args.existing_csv:
        found = load_isbn_set(existing, "isbn13")
        skip |= found
        logger.info("Skipping %d ISBNs already present in %s", len(found), existing)
    if args.resume:
        already_done = load_isbn_set(args.out, "isbn13")
        skip |= already_done
        logger.info("Resume: skipping %d ISBNs already written to %s", len(already_done), args.out)

    rows = load_pending_rows(args.csv, args.isbn_column, skip)
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    logger.info("Looking up %d ISBNs via Google Books API", total)

    if total == 0:
        logger.info("Nothing to do.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_is_new = not args.out.exists() or args.out.stat().st_size == 0
    out_f = args.out.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
    if out_is_new:
        writer.writeheader()

    session = requests.Session()
    done = 0
    ok_count = 0
    error_streak = deque(maxlen=args.breaker_window)
    start = time.monotonic()
    quota_hit = False

    chunk_size = max(args.workers * 20, args.workers)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for chunk_start in range(0, total, chunk_size):
                if quota_hit:
                    break

                chunk = rows[chunk_start : chunk_start + chunk_size]
                futures = {}
                for row in chunk:
                    isbn = row[args.isbn_column].strip()
                    futures[executor.submit(fetch_one, session, api_key, isbn, args.timeout)] = row
                    if args.delay:
                        time.sleep(args.delay)

                for future in as_completed(futures):
                    row = futures[future]
                    isbn = row[args.isbn_column].strip()
                    done += 1

                    try:
                        result = future.result()
                        error_streak.append(False)
                    except QuotaExceeded as error:
                        logger.error("Google Books quota exceeded: %s", error)
                        quota_hit = True
                        break
                    except requests.RequestException as error:
                        error_streak.append(True)
                        logger.warning("Request failed for %s: %s", isbn, error)
                        result = {"found": False}

                    if result.get("found") and len(result.get("description", "")) >= MIN_DESCRIPTION_LENGTH:
                        ok_count += 1
                        vi = result["volume_info"]
                        writer.writerow(
                            {
                                "isbn13": isbn,
                                "title": (row.get(args.title_column) or vi.get("title") or "Unknown").strip() or "Unknown",
                                "authors": (row.get(args.author_column) or ";".join(vi.get("authors", [])) or "Unknown").strip() or "Unknown",
                                "description": result["description"],
                                "simple_categories": "Unknown",
                                "published_year": parse_year(vi.get("publishedDate")),
                                "source": "google_books_api",
                                "language": "other",
                            }
                        )

                    if done % 100 == 0 or done == total:
                        out_f.flush()
                        elapsed = time.monotonic() - start
                        rate = done / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Progress: %d/%d (ok=%d, rate=%.1f/s, eta=%.0fmin)",
                            done,
                            total,
                            ok_count,
                            rate,
                            (total - done) / rate / 60 if rate > 0 else 0,
                        )

                    if len(error_streak) == error_streak.maxlen and all(error_streak):
                        logger.error(
                            "Circuit breaker: last %d requests all errored — stopping early.",
                            error_streak.maxlen,
                        )
                        quota_hit = True
                        break
    finally:
        out_f.close()

    if quota_hit:
        logger.info("Stopped early after %d/%d (ok=%d). Re-run with --resume later (e.g. tomorrow, if it's a daily quota).", done, total, ok_count)
    else:
        logger.info("Done. %d/%d found a usable description.", ok_count, total)


if __name__ == "__main__":
    main()
