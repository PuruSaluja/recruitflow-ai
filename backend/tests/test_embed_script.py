"""Tests for scripts/embed.py, with the stand-in encoder from test_embedding_service."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fake_encoder import FakeEncoder
from sqlalchemy import Engine, text

from app.config import REPO_ROOT
from app.ml.encoder import EncoderUnavailable

SCRIPT = REPO_ROOT / "scripts" / "embed.py"
Factory = Callable[..., dict[str, Any]]


def _load() -> ModuleType:
    """Not on the import path; load it from its file, as the other script tests do."""
    spec = importlib.util.spec_from_file_location("embed_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


@pytest.fixture
def url(clean_db: Engine) -> str:
    return clean_db.url.render_as_string(hide_password=False)


@pytest.fixture
def encoders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    devices: list[str] = []

    def fake(device: str) -> FakeEncoder:
        devices.append(device)
        return FakeEncoder()

    monkeypatch.setattr(script, "make_encoder", fake)
    return devices


def _rows(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.scalar(text(f"SELECT count(*) FROM {table}"))


def test_a_run_embeds_candidates_and_jobs_then_a_rerun_encodes_nothing(
    url: str,
    clean_db: Engine,
    encoders: list[str],
    make_candidate: Factory,
    make_job: Factory,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for _ in range(3):
        make_candidate()
    make_job()
    out = tmp_path / "report.json"

    assert script.main(["--url", url, "--out", str(out), "--batch-size", "2"]) == 0
    first = json.loads(out.read_text(encoding="utf-8"))
    assert first["encoder"] == "fake-encoder@1"
    assert (first["dimensions"], first["batch_size"], first["device"]) == (8, 2, "cpu")
    assert first["kinds"]["candidates"] | {"seconds": 0, "encode_seconds": 0} == {
        "encoded": 3,
        "current": 0,
        "vanished": 0,
        "seconds": 0,
        "encode_seconds": 0,
    }
    assert first["kinds"]["jobs"]["encoded"] == 1
    candidates = first["kinds"]["candidates"]
    assert 0 <= candidates["encode_seconds"] <= candidates["seconds"]
    assert (_rows(clean_db, "candidate_embeddings"), _rows(clean_db, "job_embeddings")) == (3, 1)
    printed = capsys.readouterr().out
    assert "candidates: 3 encoded, 0 already current, 0 deleted meanwhile" in printed

    assert script.main(["--url", url, "--out", str(out)]) == 0
    second = json.loads(out.read_text(encoding="utf-8"))
    assert second["kinds"]["candidates"]["encoded"] == 0
    assert second["kinds"]["candidates"]["current"] == 3
    assert second["kinds"]["jobs"]["current"] == 1
    assert encoders == ["cpu", "cpu"]


def test_kind_and_device_are_passed_through(
    url: str, clean_db: Engine, encoders: list[str], make_candidate: Factory, make_job: Factory
) -> None:
    make_candidate()
    make_job()
    assert script.main(["--url", url, "--kind", "jobs", "--device", "cuda:1"]) == 0
    assert encoders == ["cuda:1"]
    assert (_rows(clean_db, "candidate_embeddings"), _rows(clean_db, "job_embeddings")) == (0, 1)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--batch-size", "0"], "--batch-size must be at least 1, got 0"),
        (["--url", "not a url"], "cannot parse --url"),
        (["--url", "sqlite:///x.db"], "must name a PostgreSQL database, not 'sqlite'"),
    ],
    ids=["batch-size", "unparseable-url", "not-postgres"],
)
def test_bad_arguments_exit_2_with_the_reason(
    argv: list[str], message: str, encoders: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(argv) == 2
    assert message in capsys.readouterr().err
    assert encoders == []


def test_an_unreachable_database_exits_2_before_loading_the_model(
    encoders: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    url = "postgresql+psycopg://postgres:wrong@127.0.0.1:1/nowhere?connect_timeout=2"
    assert script.main(["--url", url]) == 2
    err = capsys.readouterr().err
    assert "cannot reach the database" in err
    assert "wrong" not in err
    # The driver's own message, not SQLAlchemy's wrapper and its help link.
    assert "sqlalche.me" not in err
    assert encoders == []


def test_a_missing_encoder_exits_2_and_names_the_install(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(device: str) -> None:
        raise EncoderUnavailable("sentence-transformers is not installed; install the extra")

    monkeypatch.setattr(script, "make_encoder", unavailable)
    assert script.main(["--url", url]) == 2
    assert "sentence-transformers is not installed" in capsys.readouterr().err


def test_an_unmigrated_database_exits_2_and_says_so(
    url: str, clean_db: Engine, encoders: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with clean_db.begin() as conn:
        conn.execute(text("ALTER TABLE candidate_embeddings RENAME TO candidate_embeddings_away"))
    try:
        assert script.main(["--url", url, "--kind", "candidates"]) == 2
    finally:
        with clean_db.begin() as conn:
            conn.execute(
                text("ALTER TABLE candidate_embeddings_away RENAME TO candidate_embeddings")
            )
    assert "is it migrated? run: alembic upgrade head" in capsys.readouterr().err


def test_an_unwritable_out_exits_2(
    url: str, encoders: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert script.main(["--url", url, "--out", str(blocker / "report.json")]) == 2
    assert "cannot write" in capsys.readouterr().err


def test_help_runs_without_importing_torch() -> None:
    """The command-line help must not cost a torch import, which takes seconds."""
    probe = textwrap.dedent(
        f"""
        import runpy, sys
        sys.argv = ['embed.py', '--help']
        try:
            runpy.run_path({str(SCRIPT)!r}, run_name='__main__')
        except SystemExit as exit:
            assert exit.code == 0, exit.code
        assert 'torch' not in sys.modules, 'torch was imported'
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Encode every candidate and job" in result.stdout


def test_the_batch_size_reaches_the_refresh(
    url: str, monkeypatch: pytest.MonkeyPatch, make_candidate: Factory
) -> None:
    for _ in range(3):
        make_candidate()
    made: list[FakeEncoder] = []
    monkeypatch.setattr(
        script, "make_encoder", lambda device: made.append(FakeEncoder()) or made[-1]
    )

    assert script.main(["--url", url, "--kind", "candidates", "--batch-size", "2"]) == 0
    assert [len(call) for call in made[0].calls] == [2, 1]


def test_the_device_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(script, "SentenceTransformerEncoder", lambda **kwargs: seen.append(kwargs))
    script.make_encoder("cuda:0")
    assert seen == [{"device": "cuda:0"}]


def test_time_in_the_model_is_counted() -> None:
    class Slow(FakeEncoder):
        def encode(self, texts):  # type: ignore[no-untyped-def]
            time.sleep(0.05)
            return super().encode(texts)

    timed = script.TimedEncoder(Slow())
    timed.encode(["a"])
    timed.encode(["b", "c"])
    assert 0.1 <= timed.seconds < 5
    assert (timed.name, timed.dimensions) == ("fake-encoder@1", 8)


def test_time_is_counted_even_when_the_model_fails() -> None:
    class Failing(FakeEncoder):
        def encode(self, texts):  # type: ignore[no-untyped-def]
            time.sleep(0.05)
            raise RuntimeError("model crashed")

    timed = script.TimedEncoder(Failing())
    with pytest.raises(RuntimeError, match="model crashed"):
        timed.encode(["a"])
    assert timed.seconds >= 0.05
