"""Tests for scripts/spike_vector_storage.py, the embedding-storage spike.

They run against the test database, but the script only ever creates a
temporary table, so nothing is left behind (checked below).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import psycopg
import pytest
from sqlalchemy import Engine, text

from app.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "spike_vector_storage.py"
SMALL = ["--sizes", "30", "7", "--dim", "8", "--rounds", "3", "--top-k", "5"]


def _load() -> ModuleType:
    """Not on the import path; load it from its file, as the other script tests do."""
    spec = importlib.util.spec_from_file_location("spike_vector_storage", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


@pytest.fixture
def url(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=False)


# --- a normal run ------------------------------------------------------


def test_a_run_reports_every_size_and_read_form(url: str, tmp_path: Path) -> None:
    out = tmp_path / "spike.json"

    assert script.main([*SMALL, "--url", url, "--out", str(out)]) == 0

    results = json.loads(out.read_text(encoding="utf-8"))
    assert [size["vectors"] for size in results["sizes"]] == [30, 7]
    for size in results["sizes"]:
        assert set(size["read_ms_median"]) == set(script.READ_FORMS)
        assert size["search_ms_median"] >= 0
        # 30 or 7 float32 vectors of 8 dims, plus each value's own header.
        assert size["stored_bytes"]["bytea"] > size["vectors"] * 8 * 4
    assert (results["dim"], results["rounds"], results["top_k"]) == (8, 3, 5)


def test_it_reports_the_vector_extension_as_the_server_lists_it(
    url: str, engine: Engine, tmp_path: Path
) -> None:
    out = tmp_path / "spike.json"
    assert script.main([*SMALL, "--url", url, "--out", str(out)]) == 0

    with engine.connect() as conn:
        listed = conn.execute(
            text("select default_version from pg_available_extensions where name = 'vector'")
        ).scalar_one_or_none()

    reported = json.loads(out.read_text(encoding="utf-8"))["vector_extension"]
    assert reported["available"] is (listed is not None)
    assert reported["default_version"] == listed


def test_it_leaves_no_table_behind(url: str, engine: Engine) -> None:
    assert script.main([*SMALL, "--url", url]) == 0

    with engine.connect() as conn:
        assert (
            conn.execute(text("select to_regclass(:name)"), {"name": script.TABLE}).scalar() is None
        )


def test_it_runs_as_a_command(url: str) -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *SMALL, "--url", url],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stderr
    assert "pgvector" in done.stdout


# --- every read form returns what was stored ---------------------------


@pytest.mark.parametrize("form", ["real_array_text", "real_array_binary", "bytea"])
def test_each_read_form_returns_the_stored_float32_values_exactly(url: str, form: str) -> None:
    vectors = script.unit_vectors(np.random.default_rng(3), 25, 16)
    # Values a decimal round trip could plausibly lose.
    vectors[0, :4] = np.array(
        [1e-38, -3.4e38, np.float32(1 / 3), np.nextafter(np.float32(1), np.float32(2))], np.float32
    )

    with psycopg.connect(script._libpq_url(url), autocommit=True) as conn:
        script.load(conn, vectors)
        got = script.READERS[form](conn, 16)

    assert got.dtype == np.float32
    assert np.array_equal(got, vectors)


def test_a_read_that_differs_from_what_was_stored_fails_the_run(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def off_by_one_ulp(conn: psycopg.Connection, dim: int) -> np.ndarray:
        got = script.read_bytea(conn, dim).copy()
        got[0, 0] = np.nextafter(got[0, 0], np.float32(2))
        return got

    monkeypatch.setitem(script.READERS, "bytea", off_by_one_ulp)

    assert script.main([*SMALL, "--url", url]) == 1
    assert "bytea read of 30 vectors differs" in capsys.readouterr().err


# --- the search ----------------------------------------------------------


def test_top_k_matches_a_full_sort_by_cosine() -> None:
    # Large on purpose: numpy 2.4's argpartition happens to return a small
    # top-k already sorted, which would hide a missing sort; at 500 of 4,000
    # it does not.
    rng = np.random.default_rng(11)
    stored = rng.standard_normal((4_000, 12)).astype(np.float32)
    query = rng.standard_normal(12).astype(np.float32)

    cosine = (stored @ query) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(query))
    expected = np.argsort(-cosine, kind="stable")[:500]

    assert np.array_equal(script.top_k_cosine(stored, query, 500), expected)


def test_top_k_does_not_assume_unit_length() -> None:
    # By dot product the long vector wins; by cosine the aligned one does.
    stored = np.array([[10.0, 10.0], [1.0, 0.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert script.top_k_cosine(stored, query, 1).tolist() == [1]


def test_top_k_larger_than_the_pool_returns_the_whole_pool_in_order() -> None:
    stored = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert script.top_k_cosine(stored, query, 10).tolist() == [1, 2, 0]


# --- refusals ----------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["--dim", "0"], ["--rounds", "0"], ["--top-k", "0"], ["--sizes", "10", "0"]],
)
def test_it_refuses_sizes_below_one(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert script.main(args) == 2
    assert "must" in capsys.readouterr().err


def test_an_unreachable_database_is_reported_not_raised(
    capsys: pytest.CaptureFixture[str],
) -> None:
    unreachable = "postgresql+psycopg://postgres:pw-not-for-stderr@127.0.0.1:1/recruitflow_test"

    assert script.main([*SMALL, "--url", unreachable]) == 2
    err = capsys.readouterr().err
    assert "cannot reach the database" in err
    assert "pw-not-for-stderr" not in err


def test_an_unparseable_url_is_refused_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    assert script.main([*SMALL, "--url", "not-a-url"]) == 2
    assert "cannot parse --url" in capsys.readouterr().err


def test_a_url_naming_another_kind_of_database_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Without the check it would be rewritten into a PostgreSQL URL and tried.
    assert script.main([*SMALL, "--url", "sqlite:///spike.db"]) == 2
    assert "must name a PostgreSQL database, not 'sqlite'" in capsys.readouterr().err


def test_any_other_database_error_is_reported_not_raised(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refused(conn: psycopg.Connection, vectors: np.ndarray) -> None:
        raise psycopg.errors.InsufficientPrivilege("permission denied to create temporary tables")

    monkeypatch.setattr(script, "load", refused)

    assert script.main([*SMALL, "--url", url]) == 2
    assert "permission denied to create temporary tables" in capsys.readouterr().err
