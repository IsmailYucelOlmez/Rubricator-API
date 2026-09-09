"""
Fill in missing descriptions using Open Library's batch bibkeys API
(official, sanctioned API - not a scrape), keyed by ISBN13.

Usage:
  python scripts/enrich_openlibrary_descriptions.py \
      --csv "books.csv" --isbn-column isbn \
      --existing-csv data/tr_books_cleaned.csv --existing-csv data/kitapyurdu_descriptions.csv \
      --out data/openlibrary_descriptions.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.append(str(_root))

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
BATCH_SIZE = 50
API_URL = "https://openlibrary.org/api/books"


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


def extract_description(details: dict[str, Any]) -> str:
    desc = details.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value", "")
    if not isinstance(desc, str):
        return ""
    return clean_description(desc)


def fetch_batch(session: requests.Session, isbns: list[str], timeout: float) -> dict[str, Any]:
    bibkeys = ",".join(f"ISBN:{isbn}" for isbn in isbns)
    params = {"bibkeys": bibkeys, "jscmd": "details", "format": "json"}
    try:
        response = session.get(API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        logger.warning("Batch request failed (%d isbns): %s", len(isbns), error)
        return {}


def parse_year(publish_date: str | None) -> int | None:
    if not publish_date:
        return None
    digits = "".join(ch for ch in publish_date if ch.isdigit())
    for length in (4,):
        if len(digits) >= length:
            try:
                year = int(digits[:4])
                if 1500 <= year <= 2100:
                    return year
            except ValueError:
                pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich missing descriptions via Open Library bibkeys API")
    parser.add_argument("--csv", required=True, type=Path, help="Source CSV to read title/author/isbn from")
    parser.add_argument("--isbn-column", default="isbn13")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--author-column", default="authors")
    parser.add_argument("--existing-csv", action="append", default=[], type=Path, help="Skip ISBNs already present (repeatable)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between batches (seconds)")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

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
    by_isbn = {row[args.isbn_column].strip(): row for row in rows}
    isbns = list(by_isbn.keys())
    total = len(isbns)
    logger.info("Looking up %d ISBNs via Open Library (batches of %d)", total, args.batch_size)

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
    session.headers.update({"User-Agent": "bookapp-api description enrichment (contact: repo owner)"})

    done = 0
    ok_count = 0
    start = time.monotonic()

    try:
        for i in range(0, total, args.batch_size):
            batch_isbns = isbns[i : i + args.batch_size]
            data = fetch_batch(session, batch_isbns, args.timeout)

            for isbn in batch_isbns:
                done += 1
                entry = data.get(f"ISBN:{isbn}")
                details = (entry or {}).get("details", {}) if entry else {}
                description = extract_description(details)
                if len(description) < MIN_DESCRIPTION_LENGTH:
                    continue

                row = by_isbn[isbn]
                ok_count += 1
                writer.writerow(
                    {
                        "isbn13": isbn,
                        "title": (row.get(args.title_column) or "Unknown").strip() or "Unknown",
                        "authors": (row.get(args.author_column) or "Unknown").strip() or "Unknown",
                        "description": description,
                        "simple_categories": "Unknown",
                        "published_year": parse_year(details.get("publish_date")),
                        "source": "openlibrary",
                        "language": "other",
                    }
                )

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

            if i + args.batch_size < total and args.delay > 0:
                time.sleep(args.delay)
    finally:
        out_f.close()

    logger.info("Done. %d/%d found a description.", ok_count, total)


if __name__ == "__main__":
    main()
