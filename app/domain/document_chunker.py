import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.models.document_session import DocumentChunk


@dataclass
class SplitResult:
    items: list[tuple[str, dict]]
    truncated: bool


@dataclass
class ChunkingResult:
    chunks: list[DocumentChunk]
    truncated: bool


# EPUB chapter titles are derived from file names, so most are "ch03" / "split_005" noise.
_GENERIC_TITLE_TOKENS = frozenset(
    {
        "ch", "chap", "chapter", "chp", "sec", "section", "part", "text", "content",
        "split", "index", "page", "xhtml", "html", "item", "id", "bölüm", "kısım",
    }
)


def meaningful_chapter_title(title: str | None) -> str | None:
    """The title if it carries real words, None if it is just a generated file name."""
    if not title:
        return None
    tokens = [token for token in re.split(r"[-_.\s\d]+", title.lower()) if token]
    if not tokens or all(token in _GENERIC_TITLE_TOKENS for token in tokens):
        return None
    if not any(len(token) >= 3 for token in tokens):
        return None
    return title.strip()


def _even_sample(
    items: list[tuple[str, dict]],
    limit: int,
) -> list[tuple[str, dict]]:
    """Keep coverage across the whole book instead of only the opening pages."""
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]

    last = len(items) - 1
    indices = sorted(
        {
            int(round(i * last / (limit - 1)))
            for i in range(limit)
        }
    )
    # Dedup can shrink the set on tiny corpora; fill forward if needed.
    while len(indices) < limit:
        for idx in range(len(items)):
            if idx not in indices:
                indices.append(idx)
                if len(indices) >= limit:
                    break
        else:
            break
    indices = sorted(indices)[:limit]
    return [items[i] for i in indices]


class DocumentChunker:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap,
            length_function=len,
        )

    def split_segments(self, segments: list) -> SplitResult:
        items: list[tuple[str, dict]] = []

        for segment in segments:
            pieces = self._splitter.split_text(segment.content)
            for piece in pieces:
                if not piece.strip():
                    continue
                items.append((piece, dict(segment.metadata)))

        truncated = False
        if len(items) > settings.document_max_chunks:
            truncated = True
            items = _even_sample(items, settings.document_max_chunks)

        return SplitResult(items=items, truncated=truncated)

    @staticmethod
    def embedding_text(content: str, metadata: dict) -> str:
        """Text to embed: the chunk prefixed with its chapter title when one is meaningful.

        Only the vector sees the prefix; the stored chunk and the excerpts shown
        to users/the LLM stay the original text.
        """
        title = meaningful_chapter_title(metadata.get("chapter_title"))
        return f"{title}\n\n{content}" if title else content

    def build_chunks(
        self,
        items: list[tuple[str, dict]],
        embeddings: list[list[float]],
    ) -> list[DocumentChunk]:
        count = min(len(items), len(embeddings))
        return [
            DocumentChunk(
                index=index,
                content=items[index][0],
                embedding=embeddings[index],
                metadata=items[index][1],
            )
            for index in range(count)
        ]
