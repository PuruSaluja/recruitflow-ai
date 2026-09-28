"""A JSON 500 for any exception no handler claimed, sent from inside CORS."""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


def internal_error_response() -> JSONResponse:
    """The 500 every unexpected error is answered with."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "internal_error"},
    )


class UnexpectedErrors:
    """ASGI middleware answering any escaped exception with a JSON 500.

    Starlette's own 500 comes from ``ServerErrorMiddleware``, the outermost
    layer and outside CORS: plain text with no CORS headers, which a browser
    client cannot read and reports as a network error. A handler registered for
    ``Exception`` does not help, since Starlette installs it on that same outer
    layer. This one is added just inside CORS, so its 500 gets CORS headers.

    The error is logged here and not raised on, since nothing further out would
    see it. One raised after the answer has started cannot be replaced by a
    500, and goes on to the server as before.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def tracked_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            if started:
                raise
            logger.error("unexpected error on %s", scope["path"], exc_info=exc)
            await internal_error_response()(scope, receive, send)
