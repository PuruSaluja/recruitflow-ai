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


def test_a_changed_type_counts_as_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    schema = json.loads(export_openapi.render_schema())
    schema["components"]["schemas"]["JobRead"]["properties"]["min_years_experience"]["type"] = (
        "string"
    )
    out = tmp_path / "openapi.json"
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 1
    assert "is out of date" in capsys.readouterr().err


def test_a_removed_property_counts_as_stale(tmp_path: Path) -> None:
    schema = json.loads(export_openapi.render_schema())
    del schema["components"]["schemas"]["JobRead"]["properties"]["title"]
    out = tmp_path / "openapi.json"
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 1


def test_reworded_descriptions_and_limits_only_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A newer FastAPI or pydantic rewording these must not fail CI: the types are unchanged."""
    schema = json.loads(export_openapi.render_schema())
    job = schema["components"]["schemas"]["JobRead"]
    job["description"] = "Reworded."
    job["properties"]["title"]["maxLength"] = 999
    schema["paths"]["/api/v1/jobs"]["get"]["summary"] = "Reworded too"
    # Parameters are a list, so the keywords inside lists count too.
    schema["paths"]["/api/v1/jobs"]["get"]["parameters"][0]["description"] = "And this"
    out = tmp_path / "openapi.json"
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 0
    assert "differs from the API only in descriptions" in capsys.readouterr().err


def test_a_committed_file_that_is_not_json_counts_as_stale(tmp_path: Path) -> None:
    out = tmp_path / "openapi.json"
    out.write_text("<<<<<<< HEAD\n", encoding="utf-8")

    assert export_openapi.main(["--out", str(out), "--check"]) == 1


def test_the_type_view_keeps_properties_named_like_keywords() -> None:
    """A job's ``title`` and ``description`` are properties; a schema's are documentation."""
    schema = {
        "title": "JobRead",
        "description": "A job.",
        "properties": {
            "title": {"type": "string", "title": "Title", "maxLength": 200},
            "description": {"type": "string", "description": "What the job is."},
        },
        "required": ["title", "description"],
    }

    assert export_openapi.type_view(schema) == {
        "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
        "required": ["title", "description"],
    }


def test_validation_patterns_are_left_out_and_a_property_named_pattern_is_kept() -> None:
    """Patterns differ between pydantic versions; the client's types never use them."""
    schema = {
        "Job": {
            "properties": {
                "salary_min": {"type": "string", "pattern": "^[0-9]+$"},
                "pattern": {"type": "string", "pattern": "^x$"},
            },
            "anyOf": [{"type": "string", "pattern": "^y$"}, {"type": "null"}],
        }
    }

    assert export_openapi.without_patterns(schema) == {
        "Job": {
            "properties": {
                "salary_min": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }
    }


def test_the_committed_schema_holds_no_patterns() -> None:
    """Pydantic 2.13 gives a decimal field a pattern that 2.14 leaves out."""
    assert '"pattern":' not in export_openapi.DEFAULT_OUT.read_text(encoding="utf-8")
