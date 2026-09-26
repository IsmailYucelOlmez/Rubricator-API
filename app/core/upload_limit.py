"""Reject oversized document uploads before the body is parsed.

FastAPI/Starlette parse a multipart body (spooling it to memory/disk) *before*
the endpoint runs, so the size check inside ``create_session`` only fires after
the whole upload has already been received and stored. This ASGI middleware
enforces the cap on the wire instead:

* a declared ``Content-Length`` above the limit is answered with 413 immediately,
  without reading the body;
* for bodies without a (truthful) ``Content-Length`` -- chunked uploads -- the
  bytes are counted as they arrive and the request is cut off with 413 once the
  limit is crossed.
"""

import json
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Multipart framing (boundaries, part headers) on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadSizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        path: str,
        max_bytes: Callable[[], int],
    ) -> None:
        self.app = app
        self.path = path.rstrip("/")
        self._max_bytes = max_bytes

    def _applies(self, scope: Scope) -> bool:
        return (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == self.path
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        limit = self._max_bytes() + MULTIPART_OVERHEAD_BYTES
        headers = {name.lower(): value for name, value in scope["headers"]}

        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                await _respond(send, 400, "Invalid Content-Length header")
                return
            if declared_bytes > limit:
                await _respond(send, 413, _too_large_detail(limit))
                return

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request" and not rejected:
                received += len(message.get("body") or b"")
                if received > limit:
                    rejected = True
                    await _respond(send, 413, _too_large_detail(limit))
                    # Tell the app the client is gone so it stops reading.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if rejected:
                return  # our 413 is already on the wire; drop the app's own reply
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected:
                raise
            # The app choked on the truncated body; the client already has a 413.


def _too_large_detail(limit: int) -> str:
    return f"Request body exceeds the upload limit of {limit // (1024 * 1024)} MB"


async def _respond(send: Send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"connection", b"close"),
    ]
    start: dict[str, Any] = {"type": "http.response.start", "status": status_code, "headers": headers}
    await send(start)
    await send({"type": "http.response.body", "body": body})


__all__ = ["UploadSizeLimitMiddleware", "MULTIPART_OVERHEAD_BYTES"]
