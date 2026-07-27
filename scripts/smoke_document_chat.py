"""Quick smoke test for document-chat endpoints."""
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ebooklib import epub

BASE = "http://127.0.0.1:8000/api/v1"


def _build_epub_bytes() -> bytes:
    book = epub.EpubBook()
    book.set_identifier("smoke-test")
    book.set_title("Smoke Test")
    book.set_language("tr")
    book.add_author("Test Author")

    chapter = epub.EpubHtml(
        title="Bolum 1",
        file_name="chap_01.xhtml",
        lang="tr",
    )
    chapter.content = (
        "<html xmlns=\"http://www.w3.org/1999/xhtml\">"
        "<head><title>Bolum 1</title></head>"
        "<body><p>Ana karakter Ali, hayallerinin pesinden "
        "gitmek icin sehre gitmeye karar verir.</p></body></html>"
    )
    book.add_item(chapter)
    book.toc = [chapter]
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as handle:
        path = Path(handle.name)
    try:
        epub.write_epub(str(path), book, {})
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _wait_until_ready(session_id: str, timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = _get_json(f"{BASE}/sessions/{session_id}")
        print(
            "STATUS",
            status.get("status"),
            "embedded",
            status.get("chunksEmbedded"),
            "/",
            status.get("chunksTotal"),
        )
        if status.get("status") == "ready":
            return status
        if status.get("status") == "failed":
            raise RuntimeError(status.get("errorMessage") or "Session processing failed")
        time.sleep(1)
    raise TimeoutError(f"Session {session_id} not ready within {timeout_seconds}s")


def main() -> int:
    boundary = "----testboundary"
    epub_bytes = _build_epub_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.epub"\r\n'
        "Content-Type: application/epub+zip\r\n\r\n"
    ).encode() + epub_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE}/sessions",
        data=body,
        method="POST",
    )
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            session = json.loads(resp.read().decode())
            print("CREATE", resp.status, session)
    except urllib.error.HTTPError as error:
        print("CREATE FAIL", error.code, error.read().decode())
        return 1

    session_id = session["sessionId"]
    if session.get("status") != "processing":
        print("CREATE WARN expected status=processing, got", session.get("status"))

    try:
        ready = _wait_until_ready(session_id)
    except Exception as error:
        print("WAIT FAIL", error)
        return 1

    chat_body = json.dumps({"question": "Ana karakterin motivasyonu nedir?"}).encode()
    req_chat = urllib.request.Request(
        f"{BASE}/sessions/{session_id}/chat",
        data=chat_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req_chat, timeout=120) as resp:
            chat = json.loads(resp.read().decode())
            print(
                "CHAT",
                resp.status,
                "answer_len",
                len(chat.get("answer", "")),
                "sources",
                len(chat.get("sources", [])),
            )
            print("ANSWER", chat.get("answer", "")[:200])
    except urllib.error.HTTPError as error:
        print("CHAT FAIL", error.code, error.read().decode())
        return 1

    print("READY META", ready.get("chunkCount"), ready.get("wordCount"))

    req_delete = urllib.request.Request(
        f"{BASE}/sessions/{session_id}",
        method="DELETE",
    )
    with urllib.request.urlopen(req_delete) as resp:
        print("DELETE", resp.status)

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
