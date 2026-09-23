"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DataError, IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app import __version__
from app.api.v1 import api_router
from app.config import Settings, get_settings
from app.db.session import get_db, make_session_factory
from app.ml.artifact import REGENERATE_COMMAND, MatcherUnavailable

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    When ``settings`` is passed explicitly the app is bound to that settings
    object's database through a dependency override, so the test suite can
    point an app at the test database without mutating process-wide state.
    Passing nothing uses the shared process-wide engine.
    """
    explicit_settings = settings is not None
    settings = settings or get_settings()

    app = FastAPI(
        title="RecruitFlow AI",
        description=(
            "Recruiting platform API: job requisitions, candidates, application "
            "pipelines, and ML-scored candidate shortlists."
        ),
        version=__version__,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    if explicit_settings:
        # Bind this app instance to its own engine. Without this the app would
        # silently fall through to the process-wide engine and, for the test
        # suite, run against the development database.
        session_factory = make_session_factory(settings.database_url)

        def _scoped_db() -> Iterator[Session]:
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = _scoped_db

    @app.exception_handler(IntegrityError)
    def _handle_integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
        """Translate constraint violations into 409 rather than a 500.

        Unique constraints on (job_id, candidate_id) and candidate email are
        part of the API contract, so a violation is a client error.
        """
        logger.warning("integrity error on %s: %s", request.url.path, exc.orig)
        return JSONResponse(
            status_code=409,
            content={"detail": "Resource conflicts with an existing record", "code": "conflict"},
        )

    @app.exception_handler(MatcherUnavailable)
    def _handle_matcher_unavailable(request: Request, exc: MatcherUnavailable) -> JSONResponse:
        """Answer 503 when the trained matcher is missing or cannot be loaded.

        The log gets the full message, file path and underlying error
        included; the client gets the command that fixes it, but not the
        server's filesystem layout.
        """
        logger.error("matcher unavailable on %s: %s", request.url.path, exc, exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "The trained matcher is missing or cannot be loaded. "
                    f"Regenerate it with: {REGENERATE_COMMAND}"
                ),
                "code": "matcher_unavailable",
            },
        )

    @app.exception_handler(DataError)
    def _handle_data_error(request: Request, exc: DataError) -> JSONResponse:
        """Answer a value PostgreSQL will not store as a bad request, not an outage.

        ``DataError`` is PostgreSQL's "data exception" class: text holding a NUL
        character, which no PostgreSQL text column can store, JSON it cannot
        parse (a lone surrogate in a skill list), a number too large for its
        column. Each such value arrived with the request, so it is a 422 like
        any other invalid input; the log keeps the database's own reason.
        """
        logger.warning("value refused by the database on %s: %s", request.url.path, exc.orig)
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "A value in this request cannot be stored, such as text containing "
                    "the NUL character (U+0000)"
                ),
                "code": "invalid_value",
            },
        )

    @app.exception_handler(SQLAlchemyError)
    def _handle_db_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        # exc_info=exc rather than logger.exception: a handler runs outside the
        # except block, so there is no active exception for it to pick up and
        # the traceback logged only ever read "NoneType: None".
        logger.error("database error on %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Database unavailable", "code": "db_unavailable"},
        )

    return app


app = create_app()
