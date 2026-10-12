"""The scripts' shared check that an output can be written before a long run."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import check_writable


def test_a_new_file_in_new_folders_passes_and_leaves_only_the_folders(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "report.json"

    check_writable(out)

    assert out.parent.is_dir()
    assert not out.exists()


def test_an_existing_file_passes_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.write_bytes(b'{"kept": true}\n')

    check_writable(out)

    assert out.read_bytes() == b'{"kept": true}\n'


def test_a_directory_is_refused_by_name(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    out.mkdir()

    with pytest.raises(IsADirectoryError, match="is a directory"):
        check_writable(out)


def test_a_file_where_a_folder_should_be_is_refused(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(OSError):
        check_writable(blocker / "report.json")
