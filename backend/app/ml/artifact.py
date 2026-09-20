"""Persisting the trained matcher, and failing clearly when it is missing.

Model binaries do not belong in git -- ``.gitignore`` excludes the whole
``artifacts`` directory -- so the matcher is written by
``scripts/train_matcher.py`` and read back from the same path here. Anything
that wants a trained matcher calls :func:`load_matcher`, which raises
:class:`MatcherUnavailable` naming the regeneration command rather than
letting a missing file surface as a raw ``FileNotFoundError`` two layers down.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV

from app.config import get_settings

REGENERATE_COMMAND = "python scripts/train_matcher.py"


class MatcherUnavailable(RuntimeError):
    """Raised when a trained matcher (or its manifest) was asked for but never persisted."""


def default_matcher_path() -> Path:
    """Where the trained matcher lives, per the app's settings."""
    return get_settings().resolved_model_path


def manifest_path_for(matcher_path: Path) -> Path:
    """The manifest that sits beside a given matcher path."""
    return matcher_path.with_name(f"{matcher_path.stem}.manifest.json")


def save_matcher(
    matcher: CalibratedClassifierCV,
    manifest: dict[str, Any],
    *,
    path: Path | None = None,
) -> Path:
    """Write the fitted matcher and a manifest describing how it was trained.

    ``manifest`` carries the training config and measured metrics; the
    interpreter is added here, the way the corpus manifest records it,
    because the split and the classifier both depend on ``random``'s draws.
    Returns the path the matcher was written to.
    """
    matcher_path = path if path is not None else default_matcher_path()
    matcher_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matcher, matcher_path)

    full_manifest = {
        **manifest,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    manifest_path_for(matcher_path).write_text(
        json.dumps(full_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return matcher_path


def load_matcher(path: Path | None = None) -> CalibratedClassifierCV:
    """Load the persisted matcher, or fail with a message that says how to make one."""
    matcher_path = path if path is not None else default_matcher_path()
    if not matcher_path.exists():
        raise MatcherUnavailable(
            f"No trained matcher at {matcher_path}. Regenerate it with: {REGENERATE_COMMAND}"
        )
    return joblib.load(matcher_path)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the manifest written beside a persisted matcher."""
    matcher_path = path if path is not None else default_matcher_path()
    manifest_path = manifest_path_for(matcher_path)
    if not manifest_path.exists():
        raise MatcherUnavailable(
            f"No matcher manifest at {manifest_path}. Regenerate it with: {REGENERATE_COMMAND}"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))
