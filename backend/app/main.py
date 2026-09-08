"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app import __version__
from app.api.v1 import api_router
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Accepting an optional Settings object keeps the factory testable: the test
    suite can construct an app bound to the test database without mutating
    process-wide state.
    """
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

    @app.exception_handler(SQLAlchemyError)
    def _handle_db_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("database error on %s", request.url.path)
        return JSONResponse(
            status_code=503,
            content={"detail": "Database unavailable", "code": "db_unavailable"},
        )

    return app


app = create_app()
