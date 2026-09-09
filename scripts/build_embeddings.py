"""
Import books_with_emotions.csv into Supabase book_catalog with Gemini embeddings.

Usage:
  python scripts/build_embeddings.py --csv path/to/books_with_emotions.csv --batch-size 50 --resume
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Append (don't prepend) so local supabase/migrations doesn't shadow supabase-py.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.append(str(_root))

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.data.datasources.gemini import DailyQuotaExhaustedError, embed_with_retry
from app.data.datasources.supabase import fetch_all_catalog_isbns, get_supabase_client

load_dotenv()

EMOTION_COLUMNS = ("joy", "surprise", "anger", "fear", "sadness")


def parse_volume_id_from_thumbnail(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else None


def build_emotion_scores(row: pd.Series) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in EMOTION_COLUMNS:
        value = row.get(column)
        if pd.notna(value):
            try:
                scores[column] = float(value)
            except (TypeError, ValueError):
                pass
    return scores


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"isbn13": str, "isbn10": str})
    df["isbn13"] = df["isbn13"].astype(str).str.strip()
    df = df[df["isbn13"].str.match(r"^\d{13}$", na=False)]
    return df


def existing_isbns(client) -> set[str]:
    return fetch_all_catalog_isbns(client)


def upsert_batch(client, rows: list[dict]) -> None:
    client.table("book_catalog").upsert(rows, on_conflict="isbn13").execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build book_catalog embeddings in Supabase")
    parser.add_argument("--csv", required=True, type=Path, help="Path to books_with_emotions.csv")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip ISBNs already in catalog")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=5.0,
        help=(
            "Fixed pause between batches on top of embed_with_retry's own 429 backoff. "
            "The old default (65s) was sized for the Gemini free tier's per-minute cap; "
            "lower it further (or watch the logs for 429s and raise it) once you're on a paid tier."
        ),
    )
    args = parser.parse_args()

    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is required")

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    client = get_supabase_client()
    df = load_dataframe(args.csv)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=768,
    )

    done = existing_isbns(client) if args.resume else set()
    pending = df[~df["isbn13"].isin(done)].reset_index(drop=True)
    total = len(df)

    if pending.empty:
        print(f"Catalog already complete ({total}/{total}).")
        return

    print(f"Importing {len(pending)} books ({len(done)} already done, {total} total).")

    for start in range(0, len(pending), args.batch_size):
        batch_df = pending.iloc[start : start + args.batch_size]
        texts = [
            str(row.get("tagged_description") or f"{row['isbn13']} {row.get('description', '')}")
            for _, row in batch_df.iterrows()
        ]
        vectors = embed_with_retry(
            embeddings,
            texts,
            max_retries=20,
            max_total_wait=1800,
        )

        rows: list[dict] = []
        for (_, row), vector in zip(batch_df.iterrows(), vectors, strict=True):
            isbn = str(row["isbn13"])
            if not isbn.isdigit() or len(isbn) != 13:
                continue
            thumbnail = row.get("thumbnail")
            thumbnail_str = str(thumbnail) if pd.notna(thumbnail) else None
            published = row.get("published_year")
            published_year = int(published) if pd.notna(published) else None

            rows.append(
                {
                    "isbn13": isbn,
                    "isbn10": str(row["isbn10"]) if pd.notna(row.get("isbn10")) else None,
                    "title": str(row.get("title") or "Unknown"),
                    "authors": str(row.get("authors") or "Unknown"),
                    "description": str(row.get("description") or ""),
                    "thumbnail_url": thumbnail_str,
                    "google_volume_id": parse_volume_id_from_thumbnail(thumbnail_str),
                    "simple_category": str(row.get("simple_categories") or "Unknown"),
                    "published_year": published_year,
                    "emotion_scores": build_emotion_scores(row),
                    "embedding": vector,
                    "source": str(row.get("source") or "local"),
                    "language": str(row.get("language") or "en"),
                }
            )

        upsert_batch(client, rows)
        completed = len(done) + start + len(batch_df)
        print(f"Progress: {completed}/{total}")

        if start + args.batch_size < len(pending) and args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    print("Done. Run: reindex index book_catalog_embedding_idx; analyze book_catalog;")


if __name__ == "__main__":
    try:
        main()
    except DailyQuotaExhaustedError as error:
        raise SystemExit(str(error)) from error
