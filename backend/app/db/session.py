"""Engine and session management.

The engine is created lazily so that tests can point the application at a
different database before the first connection is opened.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(database_url: str) -> Engine:
    """Create an engine that fails fast when PostgreSQL is unreachable.

    ``connect_timeout`` is the important one: without it a dead database makes
    every request hang on the TCP connect instead of returning an error, so
    /health never gets to report 503 and an orchestrator has nothing to act on.

    Every session runs in UTC, so every timestamp the API returns is written
    the same way. PostgreSQL renders a ``timestamptz`` in the session's zone,
    which otherwise follows the server's configuration (``-04:00`` on the
    development machine), while a scoring response's ``scored_at`` comes from
    Python in UTC: the same instant, but different text.
    """
    settings = get_settings()
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_timeout=settings.db_pool_timeout,
        connect_args={"connect_timeout": settings.db_connect_timeout, "options": "-c timezone=UTC"},
        future=True,
    )


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = _build_engine(get_settings().database_url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _session_factory


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    """Build an independent session factory bound to ``database_url``.

    Unlike :func:`configure_engine` this touches no module-level state, so an
    app instance can be bound to its own database without affecting anything
    else in the process.
    """
    return sessionmaker(
        bind=_build_engine(database_url), autoflush=False, expire_on_commit=False
    )


def configure_engine(database_url: str) -> Engine:
    """Rebind the engine to ``database_url``. Used by the test fixtures."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = _build_engine(database_url)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
