"""A cap on request bodies, enforced before a body is read rather than after."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.resume import MAX_UPLOAD_BYTES

#: The largest request body accepted on any route: a resume file at the 2 MB
#: upload limit plus room for the multipart framing around it, which measured
#: 478 bytes with a 255-character file name and a long content type. The upload
#: limit alone was only checked once Starlette had received the whole body, so
#: a 12 MB upload streamed in whole before its 400. JSON bodies fit easily: the
#: largest valid candidate, 100,000 resume characters each escaped as a
#: surrogate pair, is about 1.2 MB.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024


class RequestTooLarge(HTTPException):
    """A body sent without a declared length grew past the cap as it was read.

    An ``HTTPException``, so FastAPI passes it on as the error it is instead of
    reporting a body it failed to parse.
    """

    def __init__(self) -> None:
        super().__init__(status_code=http_status.HTTP_413_CONTENT_TOO_LARGE)


def too_large_response() -> JSONResponse:
    """The 413, closing the connection since the rest of the body goes unread."""
    return JSONResponse(
        status_code=http_status.HTTP_413_CONTENT_TOO_LARGE,
        content={
            "detail": f"Request body is larger than the {MAX_REQUEST_BYTES:,}-byte limit",
            "code": "request_too_large",
        },
        headers={"Connection": "close"},
    )


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            return int(value) if value.isdigit() else None
    return None


class BodySizeLimit:
    """ASGI middleware answering 413 to any request body over ``max_bytes``.

    A declared ``Content-Length`` over the cap is refused before a byte of the
    body is read. A body sent without one (chunked) is counted as it arrives,
    and reading it stops with ``RequestTooLarge`` once it passes the cap.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await too_large_response()(scope, receive, send)
            return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            # Only "http.request" messages carry a body; a disconnect adds 0.
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise RequestTooLarge()
            return message

        await self.app(scope, counted_receive, send)
