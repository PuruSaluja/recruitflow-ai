"""Health endpoint, and proof that the suite is wired to the test database."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from app import __version__
from app.config import Settings, get_settings
from app.db.session import get_db, get_engine
from app.main import create_app


def test_health_reports_ok_with_a_real_database_round_trip(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "database": "ok"}


def test_health_returns_503_when_the_database_is_unreachable() -> None:
    # Port 1 on loopback refuses immediately; nothing is listening there.
    settings = Settings(
        database_url="postgresql+psycopg://nobody:nothing@127.0.0.1:1/unreachable",
        db_connect_timeout=1,
    )
    with TestClient(create_app(settings)) as unreachable:
        response = unreachable.get(f"{settings.api_prefix}/health")

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
    assert response.json()["status"] == "degraded"


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
