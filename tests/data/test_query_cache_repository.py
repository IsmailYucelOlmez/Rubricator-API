from app.data.repositories.query_cache_repository import make_cache_key


def test_make_cache_key_is_case_and_whitespace_insensitive():
    key_a = make_cache_key("  Harry Potter ", "Advanced", "Fiction", "Happy")
    key_b = make_cache_key("harry potter", "advanced", "fiction", "happy")
    assert key_a == key_b


def test_make_cache_key_differs_when_any_field_differs():
    base = make_cache_key("query", "simple", "All", "All")
    assert base != make_cache_key("other query", "simple", "All", "All")
    assert base != make_cache_key("query", "advanced", "All", "All")
    assert base != make_cache_key("query", "simple", "Fiction", "All")
    assert base != make_cache_key("query", "simple", "All", "Happy")


def test_make_cache_key_is_a_deterministic_sha256_hex_digest():
    key = make_cache_key("query", "simple", "All", "All")
    assert key == make_cache_key("query", "simple", "All", "All")
    assert len(key) == 64
    int(key, 16)  # raises ValueError if not valid hex
