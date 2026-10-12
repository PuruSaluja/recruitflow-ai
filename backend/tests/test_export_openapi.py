"""Tests for scripts/export_openapi.py, which keeps the web client's schema current."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from app.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "export_openapi.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_openapi", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_openapi = _load()


def test_the_committed_schema_matches_the_api() -> None:
    """The web client's types are generated from this file, so it must not lag the API."""
    assert export_openapi.main(["--check"]) == 0


def test_writing_then_checking_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "openapi.json"

    assert export_openapi.main(["--out", str(out)]) == 0
    assert export_openapi.main(["--out", str(out), "--check"]) == 0

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"}\n")
    schema = json.loads(raw)
    assert "/api/v1/jobs/{job_id}/matches" in schema["paths"]


@pytest.mark.parametrize("content", [None, "{}\n"], ids=["missing", "stale"])
def test_check_fails_on_a_missing_or_stale_file(
    tmp_path: Path, content: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "openapi.json"
    if content is not None:
        out.write_text(content, encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 1
    assert "is out of date" in capsys.readouterr().err
    if content is None:
        assert not out.exists()
    else:
        assert out.read_text(encoding="utf-8") == content


def test_a_crlf_copy_counts_as_current(tmp_path: Path) -> None:
    """Line endings are a checkout's business; the content is what has to match."""
    out = tmp_path / "openapi.json"
    out.write_bytes(export_openapi.render_schema().replace("\n", "\r\n").encode())

    assert export_openapi.main(["--out", str(out), "--check"]) == 0


def test_a_one_character_change_counts_as_stale(tmp_path: Path) -> None:
    out = tmp_path / "openapi.json"
    out.write_text(export_openapi.render_schema().replace("Fetch", "Fetsh", 1), encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 1
