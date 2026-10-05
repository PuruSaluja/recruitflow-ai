"""The migration chain has to be reversible, not just runnable forwards.

`downgrade base` used to leave the five enum types behind, because
`create_table` creates them implicitly and autogenerate does not drop them.
The database looked empty -- no tables, no version row -- but the next
`upgrade head` died on 'type "seniority_level" already exists', so resetting a
database this way left it unusable.

This runs against a scratch database of its own, created and dropped here, so
the round trip cannot disturb the suite's own test database. `alembic/env.py`
takes its URL from settings, so pointing it elsewhere means repointing those
the same way conftest does.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from app.config import REPO_ROOT, Settings, get_settings

SCRATCH_DATABASE = "recruitflow_roundtrip_test"


@pytest.fixture
def scratch_database() -> Iterator[Engine]:
    configured = Settings()
    admin_url = make_url(configured.database_url).set(database="postgres")
    scratch_url = make_url(configured.database_url).set(database=SCRATCH_DATABASE)

    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DATABASE}"'))
        conn.execute(text(f'CREATE DATABASE "{SCRATCH_DATABASE}"'))

    rendered = scratch_url.render_as_string(hide_password=False)
    previous = os.environ.get("RECRUITFLOW_DATABASE_URL")
    os.environ["RECRUITFLOW_DATABASE_URL"] = rendered
    get_settings.cache_clear()

    engine = create_engine(rendered)
    try:
        yield engine
    finally:
        engine.dispose()
        if previous is None:
            os.environ.pop("RECRUITFLOW_DATABASE_URL", None)
        else:
            os.environ["RECRUITFLOW_DATABASE_URL"] = previous
        get_settings.cache_clear()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH_DATABASE}"'))
        admin.dispose()


def _alembic_config():  # type: ignore[no-untyped-def]
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    return config


def _enum_type_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM pg_type WHERE typtype = 'e'")).scalar_one()


def _table_names(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
        )


def test_head_builds_the_schema_the_models_describe(engine: Engine) -> None:
    """What `alembic check` reports, as a test: head and the models agree.

    On 2026-09-26 a foreign key's deferral changed in the model and in two
    migrations the same day, and only running `alembic check` by hand showed
    the two sides still matched. Compared the way `alembic/env.py` compares.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    import app.models  # noqa: F401  (registers every table on Base.metadata)
    from app.db.base import Base

    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_upgrade_downgrade_upgrade_on_one_database(scratch_database: Engine) -> None:
    from alembic import command

    config = _alembic_config()

    command.upgrade(config, "head")
    assert {"applications", "candidate_embeddings", "job_embeddings"} <= _table_names(
        scratch_database
    )
    assert _enum_type_count(scratch_database) == 5

    command.downgrade(config, "base")
    assert _table_names(scratch_database) <= {"alembic_version"}
    # The bug: these survived, and then blocked the upgrade below.
    assert _enum_type_count(scratch_database) == 0

    command.upgrade(config, "head")
    assert "applications" in _table_names(scratch_database)
    assert _enum_type_count(scratch_database) == 5
