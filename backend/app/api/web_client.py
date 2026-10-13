"""Serve the built web client (``frontend/dist``) from the API's own server.

Mounted at ``/`` after every API route, so ``/api/...``, ``/docs`` and
``/openapi.json`` are matched first. What reaches it:

* a file the build produced (``/assets/index-3f9a1c.js``, ``/favicon.svg``)
  is sent as it is;
* any other ``GET`` of a page address (``/jobs/12``, ``/candidates/3``) gets
  ``index.html``, and the client's router draws the page;
* everything else is the 404 an unknown path always was: a path under the API
  prefix, any method other than ``GET`` and ``HEAD``, and a missing file
  (a path whose last part has an extension), which would otherwise come back
  as HTML where the browser expects a script.
"""

from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import Receive, Scope, Send

#: Vite names every file under ``assets/`` by a hash of its content, so a
#: browser may keep one for good; ``index.html`` names the current ones and
#: must be checked on every load.
ASSET_CACHE = "public, max-age=31536000, immutable"
PAGE_CACHE = "no-cache"


class WebClient:
    """ASGI app answering for the built client's files and page addresses."""

    def __init__(self, dist: Path, api_prefix: str) -> None:
        self.dist = dist.resolve()
        self.index = self.dist / "index.html"
        self.api_prefix = api_prefix.rstrip("/")

    def _file(self, path: str) -> Path | None:
        candidate = (self.dist / path.lstrip("/")).resolve()
        if candidate.is_relative_to(self.dist) and candidate.is_file():
            return candidate
        return None

    def _respond(self, scope: Scope) -> Response:
        path: str = scope["path"]
        not_found = JSONResponse({"detail": "Not Found"}, status_code=404)
        if scope["method"] not in ("GET", "HEAD"):
            return not_found
        if self.api_prefix and (path == self.api_prefix or path.startswith(self.api_prefix + "/")):
            return not_found
        file = self._file(path)
        if file is not None:
            cache = ASSET_CACHE if file.parent.name == "assets" else PAGE_CACHE
            return FileResponse(file, headers={"Cache-Control": cache})
        if "." in path.rsplit("/", 1)[-1]:
            return not_found
        return FileResponse(self.index, headers={"Cache-Control": PAGE_CACHE})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        await self._respond(scope)(scope, receive, send)


def built_client(dist: Path | None) -> Path | None:
    """``dist`` if it holds a build (an ``index.html``), else None: nothing is served."""
    if dist is not None and (dist / "index.html").is_file():
        return dist
    return None
