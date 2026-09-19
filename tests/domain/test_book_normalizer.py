from app.domain.book_normalizer import (
    clean_description,
    extract_isbn,
    has_valid_authors,
    normalize_volumes,
)


def test_clean_description_strips_html_and_unescapes_entities():
    raw = "<p>A story about &amp; friendship.</p>"
    assert clean_description(raw) == "A story about & friendship."


def test_clean_description_collapses_whitespace():
    raw = "Line one.\n\n   Line   two."
    assert clean_description(raw) == "Line one. Line two."


def test_extract_isbn_prefers_isbn13_and_strips_dashes():
    identifiers = [
        {"type": "ISBN_10", "identifier": "0-13-468599-7"},
        {"type": "ISBN_13", "identifier": "978-0-13-468599-1"},
    ]
    isbn13, isbn10 = extract_isbn(identifiers)
    assert isbn13 == "9780134685991"
    assert isbn10 == "0134685997"


def test_extract_isbn_falls_back_to_isbn10_when_isbn13_missing():
    identifiers = [{"type": "ISBN_10", "identifier": "0134685997"}]
    isbn13, isbn10 = extract_isbn(identifiers)
    assert isbn13 == "0134685997"
    assert isbn10 == "0134685997"


def test_extract_isbn_returns_none_pair_for_empty_input():
    assert extract_isbn(None) == (None, None)
    assert extract_isbn([]) == (None, None)


def test_has_valid_authors_rejects_missing_unknown_and_blank():
    assert has_valid_authors(None) is False
    assert has_valid_authors("Unknown") is False
    assert has_valid_authors("unknown") is False
    assert has_valid_authors("   ") is False
    assert has_valid_authors([]) is False
    assert has_valid_authors(["", "   "]) is False


def test_has_valid_authors_accepts_real_names():
    assert has_valid_authors("Jane Austen") is True
    assert has_valid_authors(["Jane Austen"]) is True


def _volume(isbn13="9780134685991", description=None, authors=None, categories=None):
    return {
        "industryIdentifiers": [{"type": "ISBN_13", "identifier": isbn13}],
        "description": description if description is not None else ("A" * 60),
        "authors": authors if authors is not None else ["Jane Austen"],
        "categories": categories if categories is not None else ["Fiction"],
        "imageLinks": {"thumbnail": "https://example.com/cover.jpg?id=abc"},
        "title": "Some Title",
    }


def test_normalize_volumes_dedupes_repeated_isbn_within_the_same_batch():
    volumes = [_volume(isbn13="1111111111111"), _volume(isbn13="1111111111111")]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5)
    assert [book.isbn13 for book in results] == ["1111111111111"]


def test_normalize_volumes_skips_isbns_already_in_the_catalog():
    volumes = [_volume(isbn13="2222222222222")]
    results = normalize_volumes(volumes, existing_isbns={"2222222222222"}, max_books=5)
    assert results == []


def test_normalize_volumes_skips_descriptions_below_minimum_length():
    volumes = [_volume(isbn13="3333333333333", description="Too short")]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5)
    assert results == []


def test_normalize_volumes_skips_volumes_without_valid_authors():
    volumes = [_volume(isbn13="4444444444444", authors=[])]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5)
    assert results == []


def test_normalize_volumes_stops_at_max_books():
    volumes = [_volume(isbn13=str(i) * 13) for i in range(1, 8)]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=3)
    assert len(results) == 3


def test_normalize_volumes_filters_by_requested_category():
    volumes = [_volume(isbn13="5555555555555", categories=["Fiction"])]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5, category="Nonfiction")
    assert results == []

    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5, category="Fiction")
    assert len(results) == 1


def test_normalize_volumes_tags_description_with_isbn_for_embedding():
    volumes = [_volume(isbn13="6666666666666")]
    results = normalize_volumes(volumes, existing_isbns=set(), max_books=5)
    assert results[0].tagged_description.startswith("6666666666666 ")
