"""Shared fixtures.

Database tests run against ``RECRUITFLOW_TEST_DATABASE_URL`` and truncate every
table before each test, so pointing them at a real database would destroy it.
Two guards stand in the way of that:

1. The run is refused unless the test database name ends in ``_test`` and is
   not the development database.
2. The whole process is repointed at the test database *before* the app is
   imported, so even code that reaches for the process-wide engine -- rather
   than the app's bound session -- cannot land on development data.

Tests that need no database (the resume normaliser, schema validation) do not
request these fixtures and run without PostgreSQL.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from app.config import REPO_ROOT, Settings, get_settings

_configured = Settings()


def _unsafe_reason(test_url: str, dev_url: str) -> str | None:
    test, dev = make_url(test_url), make_url(dev_url)
    if not (test.database or "").endswith("_test"):
        return (
            f"Refusing to run: test database {test.database!r} does not end in '_test'. "
            "Every test truncates its tables, so set RECRUITFLOW_TEST_DATABASE_URL to a "
            "disposable database."
        )
    if (test.host, test.port, test.database) == (dev.host, dev.port, dev.database):
        return "Refusing to run: the test database URL points at the development database."
    return None


_UNSAFE = _unsafe_reason(_configured.test_database_url, _configured.database_url)

if _UNSAFE is None:
    os.environ["RECRUITFLOW_DATABASE_URL"] = _configured.test_database_url
    get_settings.cache_clear()


def pytest_configure(config: pytest.Config) -> None:
    if _UNSAFE is not None:
        raise pytest.UsageError(_UNSAFE)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    """Test-database engine, migrated to head once per run.

    Migrating with Alembic rather than ``metadata.create_all`` means the suite
    exercises the schema that production actually gets.
    """
    from alembic.config import Config

    from alembic import command

    # No ini file: env.py would otherwise run logging.fileConfig, which disables
    # every logger created before it -- including the app's.
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    command.upgrade(alembic_cfg, "head")

    test_engine = create_engine(get_settings().database_url)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def clean_db(engine: Engine) -> Engine:
    """Empty every table *before* the test, so a failure leaves its rows to inspect."""
    import app.models  # noqa: F401  (registers every table on Base.metadata)
    from app.db.base import Base

    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    return engine


@pytest.fixture(scope="session")
def app() -> FastAPI:
    """One app for the run: each instance owns a connection pool."""
    from app.main import create_app

    return create_app(get_settings())


@pytest.fixture
def client(app: FastAPI, clean_db: Engine) -> Iterator[TestClient]:
    """Client whose relative paths resolve under the API prefix: ``client.get("/jobs")``."""
    base_url = f"http://testserver{get_settings().api_prefix}"
    with TestClient(app, base_url=base_url) as test_client:
        yield test_client
