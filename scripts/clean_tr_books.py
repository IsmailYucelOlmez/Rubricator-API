"""
Clean the Kaggle "Turkish Book Dataset" (ardaakdere16, Turkish_Book_Dataset_Kaggle_V2.csv)
into the CSV shape build_embeddings.py expects (isbn13, title, authors, description,
simple_categories, published_year, source).

Usage:
  python scripts/clean_tr_books.py --csv path/to/Turkish_Book_Dataset_Kaggle_V2.csv --out data/tr_books_cleaned.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ISBN13_RE = re.compile(r"^\d{13}$")
WHITESPACE_RE = re.compile(r"\s+")


def to_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def clean_isbn(value: object) -> str:
    return re.sub(r"[^0-9]", "", to_str(value))


def normalize_description(text: object) -> str:
    return WHITESPACE_RE.sub(" ", to_str(text)).strip()


def truncate_description(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > max_chars * 0.6:
        return cut[: boundary + 1].strip()
    return cut.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Kaggle TR book dataset for embedding ingest")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-description-chars", type=int, default=50)
    parser.add_argument("--max-description-chars", type=int, default=3000)
    args = parser.parse_args()

    # engine="python": the default C parser silently mis-splits rows on this
    # file (produced 3x too many rows in testing) — likely an escaped-quote
    # edge case in book_detail that only the python engine's tokenizer handles.
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, engine="python")
    total = len(df)

    df["isbn13"] = df["book_productcode"].map(clean_isbn)
    df["description"] = df["book_detail"].map(normalize_description)
    df["title_clean"] = df["book_title"].map(normalize_description)

    valid_isbn = df["isbn13"].str.match(ISBN13_RE)
    dropped_isbn = total - int(valid_isbn.sum())
    df = df[valid_isbn]

    long_enough = df["description"].str.len() >= args.min_description_chars
    is_title_echo = df["description"].str.lower() == df["title_clean"].str.lower()
    keep_description = long_enough & ~is_title_echo
    dropped_description = len(df) - int(keep_description.sum())
    df = df[keep_description]

    before_dedup = len(df)
    df["_desc_len"] = df["description"].str.len()
    df = df.sort_values("_desc_len", ascending=False).drop_duplicates("isbn13", keep="first")
    dropped_dupes = before_dedup - len(df)

    df["description"] = df["description"].map(lambda d: truncate_description(d, args.max_description_chars))

    out = pd.DataFrame(
        {
            "isbn13": df["isbn13"],
            "title": df["title_clean"].replace("", "Unknown"),
            "authors": df["book_author"].map(normalize_description).replace("", "Unknown"),
            "description": df["description"],
            "simple_categories": df["book_category_name"].map(normalize_description).replace("", "Unknown"),
            "published_year": pd.to_numeric(df["book_released_year"], errors="coerce"),
            "source": "kaggle_tr_books",
            "language": "tr",
        }
    )
    out = out.sort_values("title").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Girdi satir: {total}")
    print(f"Gecersiz ISBN13 nedeniyle atildi: {dropped_isbn}")
    print(f"Aciklama eksik/kisa/baslik-yankisi nedeniyle atildi: {dropped_description}")
    print(f"ISBN tekrari nedeniyle atildi (uzun aciklama tutuldu): {dropped_dupes}")
    print(f"Cikti satir: {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
