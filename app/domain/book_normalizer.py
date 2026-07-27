import html
import re
from typing import Any

from app.domain.category_mapping import infer_simple_category
from app.models.book_record import BookRecord

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
MIN_DESCRIPTION_LENGTH = 50


def clean_description(text: str) -> str:
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_isbn(identifiers: list[dict[str, str]] | None) -> tuple[str | None, str | None]:
    if not identifiers:
        return None, None

    isbn13 = None
    isbn10 = None
    for item in identifiers:
        id_type = item.get("type", "")
        identifier = item.get("identifier", "").replace("-", "")
        if id_type == "ISBN_13" and identifier:
            isbn13 = identifier
        elif id_type == "ISBN_10" and identifier:
            isbn10 = identifier

    if not isbn13 and isbn10:
        isbn13 = isbn10
    return isbn13, isbn10


def has_valid_authors(authors: list[str] | str | None) -> bool:
    if authors is None:
        return False
    if isinstance(authors, str):
        normalized = authors.strip()
        if not normalized or normalized.lower() == "unknown":
            return False
        return bool(normalized)
    return any(author.strip() for author in authors if author and author.strip())


def normalize_volumes(
    volumes: list[dict[str, Any]],
    existing_isbns: set[str],
    max_books: int = 5,
    category: str = "All",
) -> list[BookRecord]:
    results: list[BookRecord] = []
    seen: set[str] = set()

    for volume in volumes:
        if len(results) >= max_books:
            break

        isbn13, isbn10 = extract_isbn(volume.get("industryIdentifiers"))
        if not isbn13 or isbn13 in existing_isbns or isbn13 in seen:
            continue

        raw_description = volume.get("description") or ""
        description = clean_description(raw_description)
        if len(description) < MIN_DESCRIPTION_LENGTH:
            continue

        authors_list = volume.get("authors") or []
        if not has_valid_authors(authors_list):
            continue

        categories_list = volume.get("categories") or []
        simple_categories = infer_simple_category(categories_list)
        if category != "All" and simple_categories != category:
            continue

        authors = ";".join(author.strip() for author in authors_list if author.strip())
        categories = ";".join(categories_list) if categories_list else ""
        thumbnail = (volume.get("imageLinks") or {}).get("thumbnail")
        title = volume.get("title") or "Unknown Title"

        results.append(
            BookRecord(
                isbn13=isbn13,
                isbn10=isbn10,
                title=title,
                authors=authors,
                description=description,
                thumbnail=thumbnail,
                categories=categories,
                tagged_description=f"{isbn13} {description}",
                source="google_books",
                simple_categories=simple_categories,
            )
        )
        seen.add(isbn13)

    return results
