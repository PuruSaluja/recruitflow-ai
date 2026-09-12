"""Health endpoint, and proof that the suite is wired to the test database."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.config import Settings, get_settings
from app.db.session import get_db, get_engine
from app.main import create_app


def _unreachable_settings() -> Settings:
    # Port 1 on loopback refuses immediately; nothing is listening there.
    return Settings(
        database_url="postgresql+psycopg://nobody:nothing@127.0.0.1:1/unreachable",
        db_connect_timeout=1,
    )


def test_health_reports_ok_with_a_real_database_round_trip(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "database": "ok"}


def test_health_returns_503_when_the_database_is_unreachable() -> None:
    settings = _unreachable_settings()
    with TestClient(create_app(settings)) as unreachable:
        response = unreachable.get(f"{settings.api_prefix}/health")

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
    assert response.json()["status"] == "degraded"


def test_an_unreachable_database_is_a_503_on_an_ordinary_endpoint() -> None:
    """/health handles its own failure; every other route goes to the handler."""
    settings = _unreachable_settings()
    with TestClient(create_app(settings)) as unreachable:
        response = unreachable.get(f"{settings.api_prefix}/jobs")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable", "code": "db_unavailable"}


def test_the_503_log_carries_the_real_cause(caplog: pytest.LogCaptureFixture) -> None:
    """The handler runs outside the except block, so it has to pass exc_info.

    Without it logging has no active exception to pick up and writes the
    traceback as "NoneType: None", losing which database error actually fired.
    """
    settings = _unreachable_settings()
    with (
        caplog.at_level(logging.ERROR, logger="app.main"),
        TestClient(create_app(settings)) as unreachable,
    ):
        unreachable.get(f"{settings.api_prefix}/jobs")

    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors, "the handler logged nothing"
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[0] is not None
    assert issubclass(errors[0].exc_info[0], SQLAlchemyError)
    assert "NoneType: None" not in logging.Formatter().format(errors[0])


def test_app_sessions_are_bound_to_the_test_database(app: FastAPI) -> None:
    sessions = app.dependency_overrides[get_db]()
    session = next(sessions)
    try:
        database = session.execute(text("SELECT current_database()")).scalar_one()
    finally:
        sessions.close()

    assert database.endswith("_test")


def test_process_wide_engine_is_also_bound_to_the_test_database() -> None:
    """The second guard: code bypassing the app's session still cannot reach dev data."""
    assert get_settings().database_url == Settings().test_database_url

    with get_engine().connect() as conn:
        database = conn.execute(text("SELECT current_database()")).scalar_one()

    assert database.endswith("_test")
