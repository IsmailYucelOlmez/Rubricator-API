import re
from dataclasses import dataclass
from io import BytesIO

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from app.core.config import settings


class EpubExtractionError(ValueError):
    """Raised when EPUB text cannot be extracted or limits are exceeded."""


@dataclass
class TextSegment:
    content: str
    metadata: dict


@dataclass
class EpubExtractionResult:
    segments: list[TextSegment]
    chapter_count: int
    word_count: int
    char_count: int
    truncated: bool = False


# Front-matter / ads that poison RAG for "summary" style questions.
_SKIP_NAME_RE = re.compile(
    r"(^|/)(nav|toc|ncx|cover|copyright|titlepage|title-page|"
    r"advert|ads?|catalog|catalogue|also[-_ ]?by|about[-_ ]?author|"
    r"colophon|imprint|bonus|excerpt[-_ ]?other)(|[-_.].*)\.(x?html|htm|xml)$",
    re.IGNORECASE,
)
_SKIP_TEXT_RE = re.compile(
    r"\b(also by|daha fazla kitap|diğer kitaplar|katalog|catalogue|"
    r"you may also (like|enjoy)|coming soon)\b",
    re.IGNORECASE,
)


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return _normalize_whitespace(soup.get_text(separator=" ", strip=True))


def _item_name(item: epub.EpubItem) -> str:
    return getattr(item, "get_name", lambda: "")() or ""


def _chapter_title(item: epub.EpubItem) -> str:
    name = _item_name(item)
    if name:
        return name.rsplit("/", 1)[-1].replace(".xhtml", "").replace(".html", "")
    return ""


def _should_skip_item(item: epub.EpubItem, text: str) -> bool:
    name = _item_name(item)
    if name and _SKIP_NAME_RE.search(name):
        return True
    # Short blurb-heavy promo sections.
    if len(text) < 1200 and _SKIP_TEXT_RE.search(text):
        return True
    return False


def _spine_document_items(book: epub.EpubBook) -> list[epub.EpubItem]:
    """Prefer reading order from the spine over unordered get_items()."""
    id_map: dict[str, epub.EpubItem] = {}
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        item_id = item.get_id()
        if item_id:
            id_map[item_id] = item

    ordered: list[epub.EpubItem] = []
    seen: set[str] = set()
    for entry in book.spine or []:
        item_id = entry[0] if isinstance(entry, (list, tuple)) else entry
        item = id_map.get(item_id)
        if item is None or item_id in seen:
            continue
        ordered.append(item)
        seen.add(item_id)

    if ordered:
        return ordered

    return [
        item
        for item in book.get_items()
        if item.get_type() == ITEM_DOCUMENT
    ]


class EpubExtractor:
    def extract(self, data: bytes) -> EpubExtractionResult:
        try:
            book = epub.read_epub(BytesIO(data))
        except Exception as error:
            raise EpubExtractionError("Could not read EPUB file") from error

        document_items = _spine_document_items(book)

        if not document_items:
            raise EpubExtractionError("Could not extract text from document")

        if len(document_items) > settings.document_max_chapters:
            raise EpubExtractionError(
                f"EPUB exceeds {settings.document_max_chapters} chapter limit"
            )

        segments: list[TextSegment] = []
        total_chars = 0
        total_words = 0
        truncated = False
        chapter_index = 0

        for item in document_items:
            try:
                html = item.get_content().decode("utf-8", errors="ignore")
            except Exception:
                continue

            text = _html_to_text(html)
            if not text:
                continue
            if _should_skip_item(item, text):
                continue

            chapter_index += 1
            words_in_segment = _count_words(text)
            if total_words + words_in_segment > settings.document_max_words:
                remaining_words = settings.document_max_words - total_words
                if remaining_words <= 0:
                    truncated = True
                    break
                text = " ".join(text.split()[:remaining_words])
                words_in_segment = _count_words(text)
                truncated = True

            if total_chars + len(text) > settings.document_max_chars:
                remaining = settings.document_max_chars - total_chars
                if remaining <= 0:
                    truncated = True
                    break
                text = text[:remaining]
                words_in_segment = _count_words(text)
                truncated = True

            segments.append(
                TextSegment(
                    content=text,
                    metadata={
                        "chapter_index": chapter_index,
                        "chapter_title": _chapter_title(item),
                        "format": "epub",
                    },
                )
            )
            total_chars += len(text)
            total_words += words_in_segment

        if not segments:
            raise EpubExtractionError("Could not extract text from document")

        full_text = " ".join(segment.content for segment in segments)

        return EpubExtractionResult(
            segments=segments,
            chapter_count=chapter_index,
            word_count=_count_words(full_text),
            char_count=len(full_text),
            truncated=truncated,
        )
