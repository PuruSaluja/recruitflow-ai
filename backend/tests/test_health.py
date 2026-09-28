"""Health endpoint, and proof that the suite is wired to the test database."""

from __future__ import annotations

import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import (
    InterfaceError,
    InternalError,
    InvalidRequestError,
    NoResultFound,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

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


# --- which database errors are an outage ------------------------------------
# Was: every SQLAlchemy error but IntegrityError answered 503 "Database
# unavailable", so a bug in a query told the client, and whoever read the
# logs, that the database was down. Now only the errors that mean it could not
# be reached or could not serve the request do; any other is a 500.

ORIGIN = "http://localhost:5173"
INTERNAL_ERROR = {"detail": "Internal server error", "code": "internal_error"}


def _app_raising(error: Exception) -> FastAPI:
    settings = _unreachable_settings().model_copy(update={"cors_origins": ORIGIN})
    app = create_app(settings)

    def boom() -> None:
        raise error

    app.add_api_route("/boom", boom)
    return app


def _app_running(statement: str) -> FastAPI:
    """An app on the test database with one route that runs ``statement``."""
    app = create_app(Settings(database_url=Settings().test_database_url))

    def run(db: Session = Depends(get_db)) -> None:
        db.execute(text(statement))

    app.add_api_route("/run", run)
    return app


UNAVAILABLE = [
    OperationalError("SELECT 1", {}, Exception("server closed the connection")),
    InterfaceError("SELECT 1", {}, Exception("connection already closed")),
    PoolTimeoutError("QueuePool limit of size 5 overflow 10 reached"),
]
FAULTS = [
    ProgrammingError("SELECT * FROM nowhere", {}, Exception('relation "nowhere"')),
    InternalError("SELECT 1", {}, Exception("current transaction is aborted")),
    InvalidRequestError("This session is in 'prepared' state"),
    StaleDataError("UPDATE statement on table 'jobs' expected to update 1 row(s)"),
    NoResultFound("No row was found when one was required"),
]


def _name(error: Exception) -> str:
    return type(error).__name__


@pytest.mark.parametrize("error", UNAVAILABLE, ids=_name)
def test_an_error_meaning_the_database_cannot_serve_is_a_503(error: Exception) -> None:
    with TestClient(_app_raising(error)) as client:
        response = client.get("/boom")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable", "code": "db_unavailable"}


@pytest.mark.parametrize("error", FAULTS, ids=_name)
def test_any_other_database_error_is_a_500_not_an_outage(error: Exception) -> None:
    """Answered by the app's own handler: one that escaped would fail the test (conftest)."""
    with TestClient(_app_raising(error)) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == INTERNAL_ERROR


@pytest.mark.parametrize("error", [UNAVAILABLE[0], FAULTS[0]], ids=_name)
def test_a_browser_client_can_read_either_answer(error: Exception) -> None:
    """Found in the 2026-09-25 recheck: the narrowed handler first left the 500 to Starlette.

    Starlette sends its default 500 from outside the CORS middleware, as plain
    text with no CORS headers, so a browser client saw a network error where
    the 503 before it had been readable.
    """
    with TestClient(_app_raising(error)) as client:
        response = client.get("/boom", headers={"Origin": ORIGIN})

    assert response.headers.get("access-control-allow-origin") == ORIGIN
    assert response.headers["content-type"] == "application/json"


def test_the_500_log_carries_the_real_cause(caplog: pytest.LogCaptureFixture) -> None:
    """Handled, so nothing else logs it: the handler has to, traceback included."""
    with (
        caplog.at_level(logging.ERROR, logger="app.main"),
        TestClient(_app_raising(FAULTS[0])) as client,
    ):
        client.get("/boom")

    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[1] is FAULTS[0]
    assert "/boom" in errors[0].getMessage()


def test_a_real_query_bug_is_a_500() -> None:
    """PostgreSQL's own ProgrammingError, not a constructed one."""
    with TestClient(_app_running("SELECT * FROM no_such_table")) as client:
        response = client.get("/run")

    assert response.status_code == 500
    assert response.json() == INTERNAL_ERROR


def test_a_real_statement_timeout_is_a_503(caplog: pytest.LogCaptureFixture) -> None:
    """PostgreSQL cancelling a statement files it under OperationalError: busy, not broken."""
    statement = "SET LOCAL statement_timeout = 50; SELECT pg_sleep(2)"
    with (
        caplog.at_level(logging.ERROR, logger="app.main"),
        TestClient(_app_running(statement)) as client,
    ):
        response = client.get("/run")

    assert response.status_code == 503
    assert response.json()["code"] == "db_unavailable"
    logged = [record.exc_info[1] for record in caplog.records if record.exc_info]
    assert len(logged) == 1
    assert "canceling statement due to statement timeout" in str(logged[0])
