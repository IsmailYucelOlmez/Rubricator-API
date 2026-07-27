"""Parse google_volume_id from thumbnail URLs in books_with_emotions.csv."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def parse_volume_id(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"isbn13": str})
    df["google_volume_id"] = df["thumbnail"].apply(parse_volume_id)
    out = args.output or args.csv.with_name("books_with_volume_ids.csv")
    df.to_csv(out, index=False)
    parsed = df["google_volume_id"].notna().sum()
    print(f"Parsed {parsed}/{len(df)} volume IDs → {out}")


if __name__ == "__main__":
    main()
