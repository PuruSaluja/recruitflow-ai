"""Persisting the trained matcher, and failing clearly when it is missing or damaged.

Model binaries do not belong in git -- ``.gitignore`` excludes the whole
``artifacts`` directory -- so the matcher is written by
``scripts/train_matcher.py`` and read back from the same path here. Anything
that wants a trained matcher calls :func:`load_matcher` (or
:func:`load_versioned_matcher`, when it also records which matcher scored),
which raises :class:`MatcherUnavailable` naming the regeneration command
rather than letting a missing file surface as a raw ``FileNotFoundError``, or
a damaged one as an unpickling error, two layers down.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV

from app.config import get_settings

REGENERATE_COMMAND = "python scripts/train_matcher.py"

#: Hex digest length for a matcher's version. 16 characters (64 bits) is far
#: more collision-resistant than this project will ever need, and short
#: enough to read at a glance in a match_scores row or a log line.
VERSION_DIGEST_LENGTH = 16


class MatcherUnavailable(RuntimeError):
    """Raised when a trained matcher (or its manifest) is missing, or cannot be loaded."""


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
    return load_versioned_matcher(path)[0]


def load_versioned_matcher(path: Path | None = None) -> tuple[CalibratedClassifierCV, str]:
    """Load the persisted matcher together with its :func:`matcher_version`.

    Both come from one read of the file. Hashing it and loading it as two
    reads would let a retrain that lands in between tag one model's scores
    with the other model's version.

    A file that exists but cannot be used raises :class:`MatcherUnavailable`
    as well, since the fix is the same: truncated by an interrupted write,
    not a pickle at all, pickled against code that has since moved, or
    holding something other than a matcher. Unpickling fails in too many
    ways to list -- ``EOFError``, ``UnpicklingError``, ``KeyError`` and
    ``ModuleNotFoundError`` were all measured -- so any failure is caught
    and chained.
    """
    matcher_path = path if path is not None else default_matcher_path()
    data = _read_matcher(matcher_path)
    try:
        matcher = joblib.load(io.BytesIO(data))
    except Exception as exc:
        raise MatcherUnavailable(
            f"The matcher at {matcher_path} could not be loaded ({exc!r}). "
            f"Regenerate it with: {REGENERATE_COMMAND}"
        ) from exc
    if not isinstance(matcher, CalibratedClassifierCV):
        raise MatcherUnavailable(
            f"The file at {matcher_path} holds a {type(matcher).__name__}, not a trained "
            f"matcher. Regenerate it with: {REGENERATE_COMMAND}"
        )
    return matcher, _version_of(data)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the manifest written beside a persisted matcher.

    Missing, unreadable, not JSON, or JSON that is not an object: each raises
    :class:`MatcherUnavailable` naming the regeneration command.
    """
    matcher_path = path if path is not None else default_matcher_path()
    manifest_path = manifest_path_for(matcher_path)
    if not manifest_path.exists():
        raise MatcherUnavailable(
            f"No matcher manifest at {manifest_path}. Regenerate it with: {REGENERATE_COMMAND}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # ValueError covers both invalid JSON and bytes that are not UTF-8.
        raise MatcherUnavailable(
            f"The matcher manifest at {manifest_path} could not be read ({exc!r}). "
            f"Regenerate it with: {REGENERATE_COMMAND}"
        ) from exc
    if not isinstance(manifest, dict):
        raise MatcherUnavailable(
            f"The file at {manifest_path} holds a JSON {type(manifest).__name__}, not a "
            f"matcher manifest. Regenerate it with: {REGENERATE_COMMAND}"
        )
    return manifest


def matcher_version(path: Path | None = None) -> str:
    """A short, content-derived identifier for the matcher file at ``path``.

    This is a hash of the file's bytes rather than a counter or a manifest
    field: two matcher files with identical bytes are the same trained model,
    so scoring against an unchanged artifact reuses the same version and the
    same cached rows, while retraining -- which always rewrites the file,
    even from the same config on a new corpus draw -- earns a fresh one on
    its own. Raises :class:`MatcherUnavailable` when the file is missing,
    naming the same fix as :func:`load_matcher`. It only hashes the bytes,
    so a file that cannot be unpickled still gets a version; anything that
    needs a usable matcher goes through :func:`load_versioned_matcher`.
    """
    matcher_path = path if path is not None else default_matcher_path()
    return _version_of(_read_matcher(matcher_path))


def _read_matcher(matcher_path: Path) -> bytes:
    """The matcher file's bytes, or :class:`MatcherUnavailable` naming the fix.

    A path that exists but cannot be read -- a directory, or a file this
    process may not open -- is unavailable too, rather than an ``OSError``
    surfacing as a 500.
    """
    if not matcher_path.exists():
        raise MatcherUnavailable(
            f"No trained matcher at {matcher_path}. Regenerate it with: {REGENERATE_COMMAND}"
        )
    try:
        return matcher_path.read_bytes()
    except OSError as exc:
        raise MatcherUnavailable(
            f"The matcher at {matcher_path} could not be read ({exc!r}). "
            f"Regenerate it with: {REGENERATE_COMMAND}"
        ) from exc


def _version_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:VERSION_DIGEST_LENGTH]
