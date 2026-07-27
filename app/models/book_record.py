from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BookRecord:
    isbn13: str
    isbn10: str | None
    title: str
    authors: str
    description: str
    thumbnail: str | None
    categories: str
    tagged_description: str
    source: str
    simple_categories: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BookRecord":
        return cls(**data)
