import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from app.core.config import settings


class PdfExtractionError(ValueError):
    """Raised when PDF text cannot be extracted or limits are exceeded."""


@dataclass
class TextSegment:
    content: str
    metadata: dict


@dataclass
class PdfExtractionResult:
    segments: list[TextSegment]
    page_count: int
    word_count: int
    char_count: int
    truncated: bool = False


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class PdfExtractor:
    def extract(self, data: bytes) -> PdfExtractionResult:
        try:
            # pypdf requires a seekable stream — raw bytes raise AttributeError.
            reader = PdfReader(BytesIO(data))
        except Exception as error:
            raise PdfExtractionError("Could not read PDF file") from error

        if getattr(reader, "is_encrypted", False):
            try:
                # Empty password unlocks many "open" encrypted PDFs.
                reader.decrypt("")
            except Exception as error:
                raise PdfExtractionError(
                    "PDF is encrypted and cannot be read"
                ) from error

        page_count = len(reader.pages)
        if page_count == 0:
            raise PdfExtractionError("Could not extract text from document")

        truncated = False
        pages_to_read = page_count
        if page_count > settings.document_max_pages:
            pages_to_read = settings.document_max_pages
            truncated = True

        segments: list[TextSegment] = []
        total_chars = 0

        for page_index, page in enumerate(reader.pages[:pages_to_read], start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                continue
            text = _normalize_whitespace(raw_text)
            if not text:
                continue

            if total_chars + len(text) > settings.document_max_chars:
                remaining = settings.document_max_chars - total_chars
                if remaining <= 0:
                    truncated = True
                    break
                text = text[:remaining]
                truncated = True

            segments.append(
                TextSegment(
                    content=text,
                    metadata={"page": page_index, "format": "pdf"},
                )
            )
            total_chars += len(text)

        if not segments:
            raise PdfExtractionError(
                "Could not extract text from document "
                "(scanned/image-only PDFs are not supported)"
            )

        full_text = " ".join(segment.content for segment in segments)
        word_count = _count_words(full_text)

        return PdfExtractionResult(
            segments=segments,
            page_count=page_count,
            word_count=word_count,
            char_count=len(full_text),
            truncated=truncated,
        )
