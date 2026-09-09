"""
Migrate book_catalog from Supabase (pgvector) to Qdrant Cloud, reusing the
existing embeddings already stored in Supabase (no Gemini re-embedding).

Usage:
  python scripts/migrate_to_qdrant.py --dry-run --limit 50
  python scripts/migrate_to_qdrant.py --limit 500
  python scripts/migrate_to_qdrant.py
"""
from __future__ import annotations

import argparse
import json
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
from app.data.datasources.qdrant import (  # noqa: E402
    ensure_collection,
    get_qdrant_client,
    isbn_to_point_id,
)
from app.data.datasources.supabase import get_supabase_client  # noqa: E402

SELECT_COLUMNS = (
    "isbn13,isbn10,title,authors,description,thumbnail_url,google_volume_id,"
    "simple_category,emotion_scores,embedding,source,language"
)
PAGE_SIZE = 200


def parse_embedding(raw: object) -> list[float]:
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        return [float(value) for value in json.loads(raw)]
    raise TypeError(f"Unexpected embedding type: {type(raw)!r}")


def fetch_page(client, after_isbn13: str | None, page_size: int) -> list[dict]:
    """Keyset pagination on isbn13 (the primary key) — avoids Postgres's O(offset)
    rescanning cost that .range() pays as the offset grows, which is what caused
    statement timeouts past ~29k rows during the first migration attempt."""
    query = client.table("book_catalog").select(SELECT_COLUMNS).order("isbn13").limit(page_size)
    if after_isbn13 is not None:
        query = query.gt("isbn13", after_isbn13)
    result = query.execute()
    return result.data or []


def upsert_with_retry(qdrant, collection_name: str, points, max_retries: int = 5) -> None:
    """Upserts are idempotent (deterministic point ids), so retrying on transient
    write timeouts against the free-tier cluster is always safe."""
    for attempt in range(max_retries):
        try:
            qdrant.upsert(collection_name=collection_name, points=points, wait=True)
            return
        except Exception as error:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"Upsert failed ({error}); retrying in {wait}s ({attempt + 1}/{max_retries})...")
            time.sleep(wait)


def row_to_point(row: dict) -> qmodels.PointStruct:
    isbn13 = str(row["isbn13"])
    vector = parse_embedding(row["embedding"])
    return qmodels.PointStruct(
        id=isbn_to_point_id(isbn13),
        vector=vector,
        payload={
            "isbn13": isbn13,
            "isbn10": row.get("isbn10"),
            "title": row.get("title"),
            "authors": row.get("authors"),
            "description": row.get("description"),
            "thumbnail_url": row.get("thumbnail_url"),
            "google_volume_id": row.get("google_volume_id"),
            "simple_category": row.get("simple_category"),
            "emotion_scores": row.get("emotion_scores") or {},
            "source": row.get("source") or "local",
            "language": row.get("language") or "en",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate book_catalog to Qdrant")
    parser.add_argument("--batch-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows (pilot run)")
    parser.add_argument(
        "--start-after",
        type=str,
        default=None,
        help="Resume from the isbn13 keyset cursor printed by a previous run's last Progress line",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate rows (embedding parsing) without writing to Qdrant",
    )
    args = parser.parse_args()

    supabase = get_supabase_client()
    qdrant = get_qdrant_client()
    if not args.dry_run:
        ensure_collection(qdrant)

    total_result = supabase.table("book_catalog").select("isbn13", count="exact").limit(1).execute()
    total = total_result.count or 0
    print(f"Supabase book_catalog row count: {total}")

    migrated = 0
    cursor = args.start_after
    while True:
        if args.limit is not None and migrated >= args.limit:
            break

        page_size = args.batch_size
        if args.limit is not None:
            page_size = min(page_size, args.limit - migrated)

        rows = fetch_page(supabase, cursor, page_size)
        if not rows:
            break

        points = [row_to_point(row) for row in rows]

        if args.dry_run:
            sample = points[0]
            print(
                f"[dry-run] sample point isbn13={sample.payload['isbn13']!r} "
                f"vector_len={len(sample.vector)}"
            )
        else:
            upsert_with_retry(qdrant, settings.qdrant_collection, points)

        migrated += len(rows)
        cursor = str(rows[-1]["isbn13"])
        print(f"Progress: {migrated}/{total if args.limit is None else min(total, args.limit)} (last isbn13={cursor})")

        if len(rows) < page_size:
            break

    print(f"Done. Migrated {migrated} rows{' (dry-run, nothing written)' if args.dry_run else ''}.")

    if not args.dry_run:
        qdrant_count = qdrant.count(collection_name=settings.qdrant_collection, exact=True).count
        print(f"Self-check: Supabase count={total}, Qdrant count={qdrant_count}")
        if args.limit is None and qdrant_count != total:
            print("WARNING: counts do not match.")


if __name__ == "__main__":
    main()
