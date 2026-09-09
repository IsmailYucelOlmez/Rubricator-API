"""
Scrape book descriptions from Kitapyurdu product pages for a raw Kitapyurdu
export CSV (title,author,publisher,page,language,discount_rate,
discounted_price,price,rating,reviews,cover,paper,isbn,date,link,image).

Each row already carries the exact product page URL, so this fetches that
page directly instead of trying to match against an external dataset by
title/author/ISBN.

Usage:
  python scripts/scrape_kitapyurdu_descriptions.py --csv "books.csv" --out data/kitapyurdu_descriptions.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

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
FAILED_FIELDS = ["isbn13", "title", "link", "reason"]
MIN_DESCRIPTION_LENGTH = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def build_session(pool_size: int, fresh_connection: bool = False) -> requests.Session:
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9"}
    if fresh_connection:
        # Mimic a bare `curl` invocation per request instead of a pooled
        # keep-alive session — kitapyurdu's WAF appears to fingerprint
        # persistent/concurrent connections more aggressively than isolated ones.
        headers["Connection"] = "close"
    session.headers.update(headers)
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def extract_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("#description_text") or soup.select_one(".pr_description")
    if not node:
        return ""
    return clean_description(str(node))


def parse_published_year(date_str: str) -> int | None:
    date_str = (date_str or "").strip()
    if len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None


def normalize_language(raw: str) -> str:
    return "tr" if (raw or "").strip().upper() == "TÜRKÇE" else "other"


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


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_isbns: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            isbn = (row.get("isbn") or "").strip()
            if not isbn.isdigit() or len(isbn) != 13:
                continue
            if isbn in seen_isbns:
                continue
            if not (row.get("link") or "").strip():
                continue
            seen_isbns.add(isbn)
            rows.append(row)
    return rows


def fetch_via_curl(url: str, timeout: float) -> str:
    """Shell out to a standalone curl process per request.

    kitapyurdu's WAF appears to key on the requesting client's TLS/HTTP
    fingerprint rather than request volume: isolated `curl` invocations kept
    succeeding in manual testing while every attempt made through Python's
    `requests`/urllib3 (any worker count, with or without keep-alive) was
    silently served an empty page. Spawning real curl processes sidesteps
    that specific fingerprint instead of guessing at more urllib3 tuning.
    """
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-A",
            USER_AGENT,
            "-H",
            "Accept-Language: tr-TR,tr;q=0.9",
            "--max-time",
            str(timeout),
            "-w",
            "\n%{http_code}",
            url,
        ],
        capture_output=True,
        timeout=timeout + 5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exited {result.returncode}: {result.stderr.decode(errors='replace')[:200]}")

    text = result.stdout.decode("utf-8", errors="replace")
    body, _, status_code = text.rpartition("\n")
    if not status_code.isdigit():
        raise RuntimeError(f"unexpected curl output tail: {text[-50:]!r}")
    if int(status_code) >= 400:
        raise RuntimeError(f"HTTP {status_code}")
    return body


def scrape_one(session: requests.Session, row: dict[str, str], timeout: float, use_curl: bool = False) -> dict[str, Any]:
    isbn = row["isbn"].strip()
    link = row["link"].strip()
    title = (row.get("title") or "Unknown").strip()

    try:
        if use_curl:
            html = fetch_via_curl(link, timeout)
        else:
            response = session.get(link, timeout=timeout)
            response.raise_for_status()
            html = response.text
    except (requests.RequestException, RuntimeError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "isbn13": isbn, "title": title, "link": link, "reason": str(error)[:200]}

    description = extract_description(html)
    if len(description) < MIN_DESCRIPTION_LENGTH:
        return {
            "ok": False,
            "isbn13": isbn,
            "title": title,
            "link": link,
            "reason": "no description found on page",
        }

    return {
        "ok": True,
        "isbn13": isbn,
        "title": title,
        "authors": (row.get("author") or "Unknown").strip() or "Unknown",
        "description": description,
        "simple_categories": "Unknown",
        "published_year": parse_published_year(row.get("date", "")),
        "source": "kitapyurdu_scrape",
        "language": normalize_language(row.get("language", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape descriptions from Kitapyurdu product pages")
    parser.add_argument("--csv", required=True, type=Path, help="Raw Kitapyurdu export CSV")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV (appended to if it exists)")
    parser.add_argument("--failed-out", type=Path, default=None, help="Where to log skipped/failed rows")
    parser.add_argument("--existing-csv", type=Path, default=None, help="Skip ISBNs already present here (isbn13 col)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=0.0, help="Extra per-request delay (seconds) to self-throttle")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of rows processed (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip ISBNs already written to --out")
    parser.add_argument(
        "--fresh-connection",
        action="store_true",
        help="Send 'Connection: close' to avoid keep-alive/pooled connections (mimics isolated curl requests)",
    )
    parser.add_argument(
        "--use-curl",
        action="store_true",
        help="Fetch each page via a standalone curl subprocess instead of Python's requests library",
    )
    parser.add_argument(
        "--breaker-window",
        type=int,
        default=150,
        help="Stop early if this many consecutive results all fail (likely a block)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    failed_out = args.failed_out or args.out.with_name(args.out.stem + "_failed.csv")

    rows = load_rows(args.csv)
    logger.info("Loaded %d unique candidate rows from %s", len(rows), args.csv)

    skip: set[str] = set()
    if args.existing_csv:
        skip |= load_isbn_set(args.existing_csv, "isbn13")
        logger.info("Skipping %d ISBNs already present in %s", len(skip), args.existing_csv)
    if args.resume:
        already_done = load_isbn_set(args.out, "isbn13")
        skip |= already_done
        logger.info("Resume: skipping %d ISBNs already written to %s", len(already_done), args.out)

    pending = [row for row in rows if row["isbn"].strip() not in skip]
    if args.limit:
        pending = pending[: args.limit]
    total = len(pending)
    logger.info("Scraping %d pages with %d workers", total, args.workers)

    if total == 0:
        logger.info("Nothing to do.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_is_new = not args.out.exists() or args.out.stat().st_size == 0
    failed_is_new = not failed_out.exists() or failed_out.stat().st_size == 0

    out_f = args.out.open("a", encoding="utf-8", newline="")
    failed_f = failed_out.open("a", encoding="utf-8", newline="")
    out_writer = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
    failed_writer = csv.DictWriter(failed_f, fieldnames=FAILED_FIELDS)
    if out_is_new:
        out_writer.writeheader()
    if failed_is_new:
        failed_writer.writeheader()

    session = build_session(args.workers, fresh_connection=args.fresh_connection)
    done = 0
    ok_count = 0
    start = time.monotonic()
    recent = deque(maxlen=args.breaker_window)
    tripped = False
    chunk_size = max(args.workers * 10, args.workers)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for chunk_start in range(0, total, chunk_size):
                if tripped:
                    break

                chunk = pending[chunk_start : chunk_start + chunk_size]
                futures = {}
                for row in chunk:
                    futures[executor.submit(scrape_one, session, row, args.timeout, args.use_curl)] = row
                    if args.delay:
                        time.sleep(args.delay)

                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    recent.append(result["ok"])
                    if result["ok"]:
                        ok_count += 1
                        out_writer.writerow({k: result[k] for k in OUT_FIELDS})
                    else:
                        failed_writer.writerow({k: result[k] for k in FAILED_FIELDS})

                    if done % 200 == 0 or done == total:
                        out_f.flush()
                        failed_f.flush()
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

                    if len(recent) == recent.maxlen and not any(recent):
                        logger.error(
                            "Circuit breaker: last %d results all failed — site is likely "
                            "throttling/blocking us again. Stopping early to avoid wasting "
                            "the remaining backlog.",
                            recent.maxlen,
                        )
                        tripped = True
                        break
    finally:
        out_f.close()
        failed_f.close()

    if tripped:
        logger.info("Stopped early after %d/%d (ok=%d). Re-run with --resume once traffic looks clear again.", done, total, ok_count)
    else:
        logger.info("Done. %d/%d succeeded. Failures logged to %s", ok_count, total, failed_out)


if __name__ == "__main__":
    main()
