"""Google Books category -> application simple_categories mapping."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SIMPLE_CATEGORIES = (
    "Fiction",
    "Nonfiction",
    "Children's Fiction",
    "Children's Nonfiction",
    "Unknown",
)

_MAPPING_PATH = Path(__file__).with_name("category_mapping.json")

CHILDREN_NONFICTION_PATTERNS = (
    r"juvenile nonfiction",
    r"children's nonfiction",
)

CHILDREN_FICTION_PATTERNS = (
    r"juvenile fiction",
    r"children's stories",
    r"children's plays",
    r"children's poetry",
    r"children's fiction",
)

FICTION_PATTERNS = (
    r"fictitious character",
    r"imaginary place",
    r"\bfiction\b",
    r"stories",
    r"novel",
    r"mystery",
    r"horror",
    r"fantasy",
    r"romance",
    r"thriller",
    r"suspense",
    r"comic",
    r"graphic novel",
    r"adventure fiction",
    r"ghost stories",
    r"science fiction",
    r"detective",
    r"political fiction",
    r"domestic fiction",
    r"experimental fiction",
    r"humorous fiction",
    r"diary fiction",
    r"occult fiction",
    r"christian fiction",
    r"arabic fiction",
    r"chick lit",
    r"dystopias",
    r"espionage",
    r"conspiracies",
    r"murder",
    r"young adult fiction",
    r"allegories",
    r"epic literature",
    r"sea stories",
    r"short stories",
    r"plays",
    r"poetry",
    r"drama",
)

NONFICTION_PATTERNS = (
    r"nonfiction",
    r"biography",
    r"autobiography",
    r"history",
    r"\bscience\b",
    r"religion",
    r"philosophy",
    r"criticism",
    r"reference",
    r"self-help",
    r"business",
    r"health",
    r"cookery",
    r"cooking",
    r"political science",
    r"social science",
    r"psychology",
    r"education",
    r"study aids",
    r"true crime",
    r"\bart\b",
    r"\bmusic\b",
    r"\blaw\b",
    r"medical",
    r"mathematics",
    r"computers",
    r"engineering",
    r"gardening",
    r"travel",
    r"performing arts",
    r"photography",
    r"architecture",
    r"crafts",
    r"antiques",
    r"language arts",
    r"transportation",
    r"family & relationships",
    r"body, mind",
    r"essays",
    r"theology",
    r"spiritual life",
    r"christian life",
    r"bible",
    r"bibles",
)


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, str]:
    if not _MAPPING_PATH.exists():
        return {}
    with _MAPPING_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _lookup_exact(category: str, mapping: dict[str, str]) -> str | None:
    if category in mapping:
        return mapping[category]
    lowered = category.lower()
    for key, value in mapping.items():
        if key.lower() == lowered:
            return value
    return None


def _classify_by_patterns(text: str) -> str | None:
    lowered = text.lower()
    for pattern in CHILDREN_NONFICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Children's Nonfiction"
    for pattern in CHILDREN_FICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Children's Fiction"
    for pattern in FICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Fiction"
    for pattern in NONFICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Nonfiction"
    return None


def _segments(category: str) -> list[str]:
    parts = [part.strip() for part in category.split("/") if part.strip()]
    return parts or [category]


def map_single_category(category: str, mapping: dict[str, str] | None = None) -> str | None:
    mapping = mapping if mapping is not None else _load_mapping()
    exact = _lookup_exact(category, mapping)
    if exact:
        return exact

    for segment in _segments(category):
        exact = _lookup_exact(segment, mapping)
        if exact:
            return exact
        pattern_match = _classify_by_patterns(segment)
        if pattern_match:
            return pattern_match

    return _classify_by_patterns(category)


def infer_simple_category(categories: list[str]) -> str:
    if not categories:
        return "Unknown"

    mapping = _load_mapping()
    resolved: list[str] = []

    for category in categories:
        if not category or not category.strip():
            continue
        simple = map_single_category(category.strip(), mapping)
        if simple:
            resolved.append(simple)

    if not resolved:
        return "Unknown"

    priority = (
        "Children's Fiction",
        "Children's Nonfiction",
        "Fiction",
        "Nonfiction",
    )
    for target in priority:
        if target in resolved:
            return target

    return resolved[0]
