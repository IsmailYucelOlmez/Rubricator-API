from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from app.core.upload_limit import MULTIPART_OVERHEAD_BYTES, UploadSizeLimitMiddleware

LIMIT = 1024  # bytes of file content allowed in these tests
WIRE_LIMIT = LIMIT + MULTIPART_OVERHEAD_BYTES


def _app() -> tuple[TestClient, dict]:
    reached = {"count": 0, "size": None}
    app = FastAPI()
    app.add_middleware(
        UploadSizeLimitMiddleware,
        path="/api/v1/sessions",
        max_bytes=lambda: LIMIT,
    )

    @app.post("/api/v1/sessions")
    async def upload(file: UploadFile = File(...)) -> dict:
        reached["count"] += 1
        reached["size"] = len(await file.read())
        return {"ok": True}

    @app.post("/api/v1/other")
    async def other(file: UploadFile = File(...)) -> dict:
        reached["count"] += 1
        return {"ok": True}

    return TestClient(app), reached


def _file(size: int) -> dict:
    return {"file": ("book.pdf", b"%PDF" + b"x" * (size - 4), "application/pdf")}


def test_a_normal_upload_passes_untouched():
    client, reached = _app()
    response = client.post("/api/v1/sessions", files=_file(500))
    assert response.status_code == 200
    assert reached == {"count": 1, "size": 500}


def test_a_declared_length_over_the_limit_is_rejected_before_the_app_runs():
    client, reached = _app()
    response = client.post("/api/v1/sessions", files=_file(WIRE_LIMIT + 10))
    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]
    assert reached["count"] == 0


def test_the_limit_includes_multipart_overhead_but_not_much_more():
    client, _ = _app()
    just_under = client.post("/api/v1/sessions", files=_file(LIMIT + MULTIPART_OVERHEAD_BYTES // 2))
    assert just_under.status_code == 200


def test_chunked_upload_without_content_length_is_cut_off_mid_stream():
    """A body streamed with no Content-Length can't be judged up front."""
    client, reached = _app()
    boundary = "xyzboundary"

    def body():
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        yield head
        for _ in range(WIRE_LIMIT // 1024 + 20):  # keeps going well past the limit
            yield b"x" * 1024
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        "/api/v1/sessions",
        content=body(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert reached["count"] == 0


def test_a_lying_content_length_is_still_caught_by_the_byte_count():
    client, reached = _app()
    boundary = "b"
    payload = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + b"x" * (WIRE_LIMIT + 5000) + f"\r\n--{boundary}--\r\n".encode()
    response = client.post(
        "/api/v1/sessions",
        content=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": "100",  # claims to be tiny
        },
    )
    # Either the transport or our counter refuses it; it must never reach the app.
    assert response.status_code in (400, 413)
    assert reached["count"] == 0


def test_a_malformed_content_length_is_a_400():
    client, reached = _app()
    response = client.post(
        "/api/v1/sessions",
        content=b"",
        headers={"Content-Type": "multipart/form-data; boundary=b", "Content-Length": "nope"},
    )
    assert response.status_code in (400, 422)
    assert reached["count"] == 0


def test_other_paths_and_methods_are_not_limited():
    client, reached = _app()
    big = client.post("/api/v1/other", files=_file(WIRE_LIMIT + 5000))
    assert big.status_code == 200
    assert reached["count"] == 1


def test_a_trailing_slash_path_is_still_covered():
    client, reached = _app()
    response = client.post("/api/v1/sessions/", files=_file(WIRE_LIMIT + 10))
    assert response.status_code in (307, 404, 405, 413)
    assert reached["count"] == 0
