"""Cheap hardening for untrusted text (uploaded documents, filenames) that ends up in LLM prompts.

None of this makes prompt injection impossible; it removes the easy wins (forging
our own delimiters, injecting through the filename) and gives us log signals.
"""

import re
from collections import Counter
from collections.abc import Iterable

_EXCERPT_TAG_RE = re.compile(r"<\s*(/?)\s*(excerpts?)\b", re.IGNORECASE)

_SIGNAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instructions?|prompts?|rules?|directions?)\b",
        re.IGNORECASE,
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|show|print|repeat|output)\b[^.\n]{0,30}\b(system|hidden|initial)\s+"
        r"(prompt|instructions?|message)\b",
        re.IGNORECASE,
    ),
    "chat_markup": re.compile(r"<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>", re.IGNORECASE),
    "role_prefix": re.compile(r"(^|\n)\s*(system|developer)\s*:", re.IGNORECASE),
    "tr_ignore_instructions": re.compile(
        r"(talimat|komut|yönerge)\w*[^.\n]{0,30}(yok say|unut|görmezden)\w*"
        r"|(yok say|unut|görmezden gel)\w*[^.\n]{0,30}(talimat|komut|yönerge)",
        re.IGNORECASE,
    ),
    "excerpt_tag_forgery": _EXCERPT_TAG_RE,
}

_FILENAME_MAX_LENGTH = 100


def escape_excerpt_text(text: str) -> str:
    """Stops document text from closing/opening our <excerpts> delimiters."""
    return _EXCERPT_TAG_RE.sub(r"&lt;\1\2", text)


def safe_filename(name: str | None) -> str:
    """User-supplied filenames go into the system prompt; keep only harmless characters."""
    cleaned = re.sub(r"[^\w .,()\[\]-]", "", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()[:_FILENAME_MAX_LENGTH].strip()
    return cleaned or "document"


def find_injection_signals(text: str) -> list[str]:
    return [name for name, pattern in _SIGNAL_PATTERNS.items() if pattern.search(text)]


def summarize_injection_signals(texts: Iterable[str]) -> tuple[int, Counter[str]]:
    """(number of flagged texts, how many texts matched each signal)."""
    flagged = 0
    counts: Counter[str] = Counter()
    for text in texts:
        signals = find_injection_signals(text)
        if signals:
            flagged += 1
            counts.update(signals)
    return flagged, counts
