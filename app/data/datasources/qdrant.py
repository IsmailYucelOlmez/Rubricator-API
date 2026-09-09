import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_SCROLL_PAGE_SIZE = 1000
_VECTOR_SIZE = 768

# Fixed namespace so isbn13 -> point id stays deterministic across runs.
ISBN_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://bookapp-api/book_catalog")


def isbn_to_point_id(isbn13: str) -> str:
    return str(uuid.uuid5(ISBN_NAMESPACE, isbn13))


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        if not settings.cluster_endpoint or not settings.cluster_api_key:
            raise RuntimeError("CLUSTER_ENDPOINT and CLUSTER_API_KEY are required")
        _client = QdrantClient(
            url=settings.cluster_endpoint, api_key=settings.cluster_api_key, timeout=60
        )
    return _client


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Idempotent create-if-missing. Not called at app startup — invoked explicitly
    by the migration script."""
    db = client or get_qdrant_client()
    collection_name = settings.qdrant_collection

    if not db.collection_exists(collection_name):
        db.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=_VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
                on_disk=True,
            ),
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    always_ram=True,
                ),
            ),
            on_disk_payload=True,
        )
        logger.info("Created Qdrant collection %r", collection_name)

    for field_name in ("isbn13", "simple_category", "language"):
        db.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )


def fetch_all_catalog_isbns(client: QdrantClient | None = None) -> set[str]:
    """Load all ISBNs from the book_catalog collection (paginated scroll)."""
    db = client or get_qdrant_client()
    isbns: set[str] = set()
    offset = None

    while True:
        points, offset = db.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["isbn13"],
            with_vectors=False,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
        )
        for point in points:
            isbn = point.payload.get("isbn13") if point.payload else None
            if isbn:
                isbns.add(str(isbn))
        if offset is None:
            break

    return isbns
