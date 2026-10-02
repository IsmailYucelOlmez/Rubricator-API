import importlib.util
from pathlib import Path
from types import SimpleNamespace

from app.domain.cover_urls import cover_for_display, normalize_cover_url

GOOGLE = "http://books.google.com/books/content?id=abc&printsec=frontcover&img=1&zoom=1"
KITAPYURDU = "https://img.kitapyurdu.com/v1/getImage/fn:11252051/wi:100/wh:true"


def _load_backfill():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_catalog_covers.py"
    spec = importlib.util.spec_from_file_location("backfill_catalog_covers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_upgrades_http_and_kitapyurdu_width():
    assert normalize_cover_url(GOOGLE).startswith("https://books.google.com/")
    assert normalize_cover_url(KITAPYURDU) == (
        "https://img.kitapyurdu.com/v1/getImage/fn:11252051/wi:400/wh:true"
    )
    other = "https://i.dr.com.tr/cache/600x600-0/originals/1.jpg"
    assert normalize_cover_url(other) == other


def test_normalize_drops_blank():
    assert normalize_cover_url(None) is None
    assert normalize_cover_url("  ") is None


def test_size_hint_only_for_google_books():
    assert cover_for_display(GOOGLE) == GOOGLE + "&fife=w800"
    assert cover_for_display("https://books.google.com/books/content") == (
        "https://books.google.com/books/content?fife=w800"
    )
    assert cover_for_display(KITAPYURDU) == KITAPYURDU
    assert cover_for_display(None) is None


def test_pick_covers_prefers_latest_scrape_and_skips_unusable_rows():
    backfill = _load_backfill()
    covers = backfill.pick_covers(
        [
            {"isbn": "9780000000001", "image_url": KITAPYURDU, "scraped_at": "2026-01-01"},
            {"isbn": "9780000000001", "image_url": "https://x.test/new.jpg", "scraped_at": "2026-05-01"},
            {"isbn": "9780000000002", "image_url": ""},
            {"isbn": "", "image_url": "https://x.test/a.jpg"},
            {"isbn": "9780000000003", "image_url": KITAPYURDU, "scraped_at": None},
        ]
    )
    assert covers == {
        "9780000000001": "https://x.test/new.jpg",
        "9780000000003": KITAPYURDU.replace("wi:100", "wi:400"),
    }


def test_plan_updates_only_fills_empty_matched_points():
    backfill = _load_backfill()

    def point(pid, isbn, thumb=None):
        return SimpleNamespace(
            id=pid, payload={"isbn13": isbn, "thumbnail_url": thumb, "language": "tr"}
        )

    updates, stats = backfill.plan_updates(
        [
            point("a", "9780000000001"),
            point("b", "9780000000002", thumb="https://keep.test/x.jpg"),
            point("c", "9780000000003"),
        ],
        {"9780000000001": "https://x.test/1.jpg", "9780000000002": "https://x.test/2.jpg"},
    )
    assert [u["point_id"] for u in updates] == ["a"]
    assert updates[0]["thumbnail_url"] == "https://x.test/1.jpg"
    assert stats == {"points": 3, "has_cover": 1, "empty": 2, "matched": 1}


def test_apply_and_undo_against_in_memory_qdrant(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    backfill = _load_backfill()
    monkeypatch.setattr(backfill.settings, "qdrant_collection", "covers_test")
    client = QdrantClient(":memory:")
    client.create_collection(
        "covers_test",
        vectors_config=qmodels.VectorParams(size=2, distance=qmodels.Distance.COSINE),
    )
    client.upsert(
        "covers_test",
        points=[
            qmodels.PointStruct(id=1, vector=[1, 0], payload={"isbn13": "9780000000001", "title": "A"}),
            qmodels.PointStruct(
                id=2,
                vector=[0, 1],
                payload={"isbn13": "9780000000002", "thumbnail_url": "https://keep.test/x.jpg"},
            ),
        ],
    )
    covers = {"9780000000001": "https://x.test/1.jpg", "9780000000002": "https://x.test/2.jpg"}

    updates, _ = backfill.plan_updates(backfill.scroll_catalog(client), covers)
    log_path = tmp_path / "undo.jsonl"
    backfill.apply_updates(client, updates, batch_size=1, log_path=log_path)

    def payload(point_id):
        return client.retrieve("covers_test", ids=[point_id], with_payload=True)[0].payload

    assert payload(1) == {"isbn13": "9780000000001", "title": "A", "thumbnail_url": "https://x.test/1.jpg"}
    assert payload(2)["thumbnail_url"] == "https://keep.test/x.jpg"

    # A second run finds nothing left to fill.
    assert backfill.plan_updates(backfill.scroll_catalog(client), covers)[0] == []

    backfill.undo(client, log_path, batch_size=10)
    assert payload(1) == {"isbn13": "9780000000001", "title": "A"}
    assert payload(2)["thumbnail_url"] == "https://keep.test/x.jpg"
