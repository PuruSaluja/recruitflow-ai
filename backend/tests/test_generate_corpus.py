"""Unit tests for data/generate_corpus.py, the corpus writer. No database needed."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, corpus_summary, generate_corpus

SCRIPT = REPO_ROOT / "data" / "generate_corpus.py"
BASE = ["--seed", "7", "--jobs", "5"]
#: Small enough to stay fast, big enough to fill a slate from every family.
ARGS = [*BASE, "--candidates", "300", "--slate-size", "10"]
CONFIG = CorpusConfig(seed=7, n_jobs=5, n_candidates=300, slate_size=10)


def _load() -> ModuleType:
    """The script is not on the import path; load it from its file."""
    spec = importlib.util.spec_from_file_location("generate_corpus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    assert script.main([*ARGS, "--out", str(out)]) == 0
    return out


def test_it_writes_the_four_files(written: Path) -> None:
    assert sorted(path.name for path in written.iterdir()) == [
        "candidates.jsonl",
        "jobs.jsonl",
        "manifest.json",
        "pairs.jsonl",
    ]


def test_it_creates_an_output_directory_that_does_not_exist(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "corpus"

    assert script.main([*ARGS, "--out", str(out)]) == 0
    assert (out / "manifest.json").is_file()


def test_every_row_is_the_dataclass_it_came_from(written: Path) -> None:
    corpus = generate_corpus(CONFIG)
    for name, items in (
        ("jobs", corpus.jobs),
        ("candidates", corpus.candidates),
        ("pairs", corpus.pairs),
    ):
        rows = _rows(written / f"{name}.jsonl")
        assert len(rows) == len(items), name
        for row, item in zip(rows, items, strict=True):
            # Through JSON, so a tuple field is compared as the list it becomes.
            assert row == json.loads(json.dumps(script.record(item))), (name, row.get("ref", row))


def test_a_candidate_row_keeps_what_the_api_payload_withholds(written: Path) -> None:
    """The fairness audit reads these; the matcher must never be handed them."""
    rows = _rows(written / "candidates.jsonl")

    for row in rows:
        assert {"gender", "age_band", "true_skills", "family", "ref"} <= row.keys()
    assert {row["gender"] for row in rows} == {"woman", "man", "nonbinary"}
    assert {row["age_band"] for row in rows} == {"under_40", "40_and_over"}


def test_a_frozenset_is_written_in_a_stable_order(written: Path) -> None:
    """Set order varies between processes; the file must not."""
    for row in _rows(written / "candidates.jsonl"):
        assert row["true_skills"] == sorted(row["true_skills"])


def test_the_manifest_records_the_config_and_the_interpreter(written: Path) -> None:
    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["synthetic"] is True
    assert manifest["config"] == dataclasses.asdict(CONFIG)
    assert manifest["rows"] == {"jobs.jsonl": 5, "candidates.jsonl": 300, "pairs.jsonl": 50}
    assert manifest["python"] == ".".join(str(part) for part in sys.version_info[:3])
    assert manifest["implementation"] == platform.python_implementation()


def test_the_manifest_summary_describes_the_corpus_it_wrote(written: Path) -> None:
    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))
    expected = dataclasses.asdict(corpus_summary(generate_corpus(CONFIG)))

    # JSON has no integer keys; relevance_counts comes back keyed by string.
    assert manifest["summary"]["relevance_counts"] == {
        str(grade): count for grade, count in expected.pop("relevance_counts").items()
    }
    del manifest["summary"]["relevance_counts"]
    assert manifest["summary"] == expected


def test_writing_the_same_corpus_twice_gives_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"

    assert script.main([*ARGS, "--out", str(first)]) == 0
    assert script.main([*ARGS, "--out", str(second)]) == 0

    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_the_bias_flags_reach_the_config_and_clear_both_plants(tmp_path: Path) -> None:
    assert (
        script.main([*ARGS, "--out", str(tmp_path), "--historical-bias", "0", "--proxy-share", "0"])
        == 0
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["config"]["historical_bias"] == 0.0
    assert manifest["config"]["proxy_share"] == 0.0
    assert manifest["summary"]["proxy_rate_women"] == 0.0


def test_a_corpus_that_cannot_exist_is_refused_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config the generator rejects is the caller's mistake, not a crash."""
    code = script.main(["--out", str(tmp_path), "--candidates", "30", "--slate-size", "50"])

    assert code == 2
    assert "slate_size cannot be larger" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_a_slate_the_pool_cannot_fill_is_refused_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every candidate of the slate from one family, from a pool too small to hold ten of any.
    args = [*BASE, "--candidates", "100", "--slate-size", "10", "--same-family-share", "1.0"]

    code = script.main([*args, "--out", str(tmp_path)])

    assert code == 2
    assert "cannot fill a slate" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_it_runs_as_a_command(tmp_path: Path) -> None:
    """Imported and invoked are different things; this is the documented way to run it."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *ARGS, "--out", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "candidates.jsonl: 300 rows" in result.stdout
    assert (tmp_path / "manifest.json").is_file()
