"""Helpers the command-line scripts in ``scripts/`` share."""

from __future__ import annotations

from pathlib import Path


def check_writable(path: Path) -> None:
    """Make sure ``path`` can be written before a long run produces what goes there.

    Creates the parent folders, refuses a directory, and opens the file for
    appending, which an existing file survives unchanged; a file this check
    created is removed again. Raises ``OSError`` naming the problem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory")
    existed = path.exists()
    with path.open("ab"):
        pass
    if not existed:
        path.unlink()
