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
#: a 12 MB upload streamed in whole before its 400. JSON bodies fit easily: a
#: candidate with every text field at its limit, all but the plain-ASCII email
#: written as surrogate-pair escapes, measured 1,327,845 bytes, 61% of the cap.
#: (Nothing can make every valid JSON body fit, since JSON allows any amount of
#: whitespace.)
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024

#: How much of a refused body is read and thrown away before the 413 is sent.
#: The 413 goes out while the client is still sending, and closing a socket
#: with unread data resets the connection, which discards the answer on the
#: client's side: on 2026-09-28, 12 of 12 over-cap uploads to a live server
#: read no answer at all. Reading the rest first lets the answer through, at
#: the cost of receiving bytes nobody uses. Past this bound the rest is left
#: unread, so a much larger upload can still end in a reset.
MAX_DRAIN_BYTES = 16 * 1024 * 1024


class RequestTooLarge(HTTPException, OSError):
    """A body sent without a declared length grew past the cap as it was read.

    An ``HTTPException``, so FastAPI passes it on as the error it is instead of
    reporting a body it failed to parse. Also an ``OSError``, the one error
    besides its own that Starlette's multipart parser cleans up after: it
    closes the temp file holding the part of an upload read so far, then
    re-raises. Otherwise that file, on disk once past 1 MB, is left for the
    garbage collector to close.
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


def _expects_continue(scope: Scope) -> bool:
    """Whether the client holds its body back until the server says to send it."""
    return any(
        name == b"expect" and value.lower() == b"100-continue" for name, value in scope["headers"]
    )


async def _drain(receive: Receive, budget: int) -> None:
    """Read and discard the rest of a body, stopping once past ``budget`` bytes."""
    drained = 0
    while drained <= budget:
        message = await receive()
        drained += len(message.get("body", b""))
        # The body's last part, or a disconnect: neither says more is coming.
        if not message.get("more_body", False):
            return


class BodySizeLimit:
    """ASGI middleware answering 413 to any request body over ``max_bytes``.

    A declared ``Content-Length`` over the cap is refused before the app sees a
    byte of the body. A body sent without one (chunked) is counted as it
    arrives, and reading it stops with ``RequestTooLarge`` once it passes the
    cap. Either way the rest of the body is read and thrown away first, up to
    ``max_drain`` bytes in all, so the client can read the 413 (see
    :data:`MAX_DRAIN_BYTES`); a client waiting for ``100 Continue`` has sent
    nothing, and reading would ask it to, so it is answered at once.
    """

    def __init__(
        self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES, max_drain: int = MAX_DRAIN_BYTES
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.max_drain = max_drain

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            if declared <= self.max_drain and not _expects_continue(scope):
                await _drain(receive, self.max_drain)
            await too_large_response()(scope, receive, send)
            return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            # Only "http.request" messages carry a body; a disconnect adds 0.
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                if message.get("more_body", False):
                    await _drain(receive, self.max_drain - received)
                raise RequestTooLarge()
            return message

        await self.app(scope, counted_receive, send)
