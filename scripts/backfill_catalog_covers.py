"""
Fill empty `thumbnail_url` payloads in the Qdrant catalog from bookapp's
`trbooks.image_url`, matched by ISBN-13.

The Turkish catalog was imported from CSVs without covers; the covers were
scraped into `trbooks` later. Only the payload field is written: vectors and
every other field stay as they are, and points that already have a cover are
never touched. Re-running only picks up what is still empty, so run it again
after any catalog import.

Usage:
  python scripts/backfill_catalog_covers.py --dry-run
  python scripts/backfill_catalog_covers.py --limit 200
  python scripts/backfill_catalog_covers.py
  python scripts/backfill_catalog_covers.py --undo logs/cover_backfill_<stamp>.jsonl

Every write is appended to a JSONL log first; --undo clears exactly those
points' thumbnail_url again.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from dotenv import load_dotenv
from qdrant_client.http import models as qmodels

load_dotenv()

from app.core.config import settings  # noqa: E402
from app.data.datasources.qdrant import get_qdrant_client  # noqa: E402
from app.data.datasources.supabase import get_supabase_client  # noqa: E402
from app.domain.cover_urls import normalize_cover_url  # noqa: E402

TRBOOKS_PAGE_SIZE = 1000
SCROLL_PAGE_SIZE = 2000
DEFAULT_BATCH_SIZE = 500


def pick_covers(rows: list[dict]) -> dict[str, str]:
    """ISBN -> cover URL. When one ISBN has several rows, the most recently
    scraped (then created) row wins, so the choice is stable across runs."""

    def recency(row: dict) -> tuple[str, str]:
        return (row.get("scraped_at") or "", row.get("created_at") or "")

    best: dict[str, dict] = {}
    for row in rows:
        isbn = (row.get("isbn") or "").strip().upper()
        url = normalize_cover_url(row.get("image_url"))
        if not isbn or not url:
            continue
        current = best.get(isbn)
        if current is None or recency(row) > recency(current):
            best[isbn] = {**row, "image_url": url}
    return {isbn: row["image_url"] for isbn, row in best.items()}


def _execute_with_retry(query, attempts: int = 4):
    """Supabase occasionally answers a page with a 522/timeout; retry it."""
    for attempt in range(1, attempts + 1):
        try:
            return query.execute()
        except Exception as error:  # postgrest.APIError, httpx errors
            if attempt == attempts:
                raise
            wait = 2 ** attempt
            print(f"trbooks page failed ({str(error)[:80]}...), retrying in {wait}s")
            time.sleep(wait)


def load_trbooks_covers() -> dict[str, str]:
    db = get_supabase_client()
    rows: list[dict] = []
    last_id: str | None = None
    # Keyset pagination on the primary key: deep offsets time out on this table.
    while True:
        query = (
            db.table("trbooks")
            .select("id,isbn,image_url,scraped_at,created_at")
            .not_.is_("image_url", "null")
            .not_.is_("isbn", "null")
            .order("id")
            .limit(TRBOOKS_PAGE_SIZE)
        )
        if last_id is not None:
            query = query.gt("id", last_id)
        page = _execute_with_retry(query).data or []
        rows.extend(page)
        if len(page) < TRBOOKS_PAGE_SIZE:
            break
        last_id = page[-1]["id"]
    covers = pick_covers(rows)
    print(f"trbooks: {len(rows)} rows with a cover -> {len(covers)} distinct ISBNs")
    return covers


def plan_updates(points, covers: dict[str, str]) -> tuple[list[dict], dict[str, int]]:
    """Points whose thumbnail_url is empty and whose ISBN has a cover."""
    updates: list[dict] = []
    stats = {"points": 0, "has_cover": 0, "empty": 0, "matched": 0}
    for point in points:
        stats["points"] += 1
        payload = point.payload or {}
        if payload.get("thumbnail_url"):
            stats["has_cover"] += 1
            continue
        stats["empty"] += 1
        isbn = str(payload.get("isbn13") or "").strip().upper()
        url = covers.get(isbn)
        if url:
            stats["matched"] += 1
            updates.append(
                {
                    "point_id": point.id,
                    "isbn13": isbn,
                    "language": payload.get("language"),
                    "thumbnail_url": url,
                }
            )
    return updates, stats


def scroll_catalog(client):
    offset = None
    while True:
        points, offset = client.scroll(
            settings.qdrant_collection,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["isbn13", "thumbnail_url", "language"],
            with_vectors=False,
        )
        yield from points
        if offset is None:
            return


def apply_updates(client, updates: list[dict], batch_size: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with log_path.open("a", encoding="utf-8") as log:
        for start in range(0, len(updates), batch_size):
            batch = updates[start : start + batch_size]
            # Log before writing so an interrupted batch can still be undone.
            for update in batch:
                log.write(json.dumps(update, ensure_ascii=False) + "\n")
            log.flush()
            client.batch_update_points(
                collection_name=settings.qdrant_collection,
                update_operations=[
                    qmodels.SetPayloadOperation(
                        set_payload=qmodels.SetPayload(
                            payload={"thumbnail_url": update["thumbnail_url"]},
                            points=[update["point_id"]],
                        )
                    )
                    for update in batch
                ],
                wait=True,
            )
            done += len(batch)
            print(f"Progress: {done}/{len(updates)}")


def undo(client, log_path: Path, batch_size: int) -> None:
    ids = [json.loads(line)["point_id"] for line in log_path.read_text("utf-8").splitlines() if line]
    for start in range(0, len(ids), batch_size):
        client.delete_payload(
            collection_name=settings.qdrant_collection,
            keys=["thumbnail_url"],
            points=ids[start : start + batch_size],
            wait=True,
        )
    print(f"Cleared thumbnail_url on {len(ids)} points from {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill catalog covers from trbooks")
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    parser.add_argument("--limit", type=int, default=None, help="Write at most N points (pilot run)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--undo", type=Path, default=None, help="JSONL log of a previous run")
    args = parser.parse_args()

    client = get_qdrant_client()
    if args.undo:
        undo(client, args.undo, args.batch_size)
        return

    covers = load_trbooks_covers()
    started = time.monotonic()
    updates, stats = plan_updates(scroll_catalog(client), covers)
    print(
        f"catalog: {stats['points']} points, {stats['has_cover']} already have a cover, "
        f"{stats['empty']} empty, {stats['matched']} matched in trbooks "
        f"({time.monotonic() - started:.0f}s)"
    )
    by_language: dict[str, int] = {}
    for update in updates:
        key = update["language"] or "?"
        by_language[key] = by_language.get(key, 0) + 1
    print(f"matched by language: {by_language}")

    if args.limit is not None:
        updates = updates[: args.limit]

    for update in random.Random(0).sample(updates, min(5, len(updates))):
        print(f"  {update['isbn13']} -> {update['thumbnail_url']}")

    if args.dry_run:
        print(f"Dry run: would set thumbnail_url on {len(updates)} points")
        return
    if not updates:
        print("Nothing to do")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = _root / "logs" / f"cover_backfill_{stamp}.jsonl"
    apply_updates(client, updates, args.batch_size, log_path)
    print(f"Done. Undo log: {log_path}")


if __name__ == "__main__":
    main()
