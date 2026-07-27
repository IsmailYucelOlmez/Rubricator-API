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
