"""Unit tests for persisting the matcher: the round trip and the missing-file errors.

No database needed.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.ml.artifact import (
    VERSION_DIGEST_LENGTH,
    MatcherUnavailable,
    default_matcher_path,
    load_manifest,
    load_matcher,
    manifest_path_for,
    matcher_version,
    save_matcher,
)
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.train import split_by_job, train_matcher, training_data


@pytest.fixture(scope="module")
def fitted_and_scores():
    """A small fitted matcher, and the scores it gives its own held-out set."""
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    train_set, test_set = split_by_job(training_data(corpus))
    matcher = train_matcher(train_set)
    scores = matcher.predict_proba(test_set.pairs)[:, 1]
    return matcher, test_set, scores


# --- default path -------------------------------------------------------


def test_the_default_path_is_the_configured_model_path() -> None:
    assert default_matcher_path() == get_settings().resolved_model_path


def test_the_manifest_sits_beside_the_matcher_with_its_own_name(tmp_path: Path) -> None:
    assert manifest_path_for(tmp_path / "matcher.joblib") == tmp_path / "matcher.manifest.json"
    assert manifest_path_for(tmp_path / "foo.bar.joblib") == tmp_path / "foo.bar.manifest.json"


# --- save / load round trip ----------------------------------------------


def test_a_saved_matcher_loads_back_to_identical_predictions(
    fitted_and_scores, tmp_path: Path
) -> None:
    matcher, test_set, scores = fitted_and_scores
    path = save_matcher(matcher, manifest={}, path=tmp_path / "matcher.joblib")

    loaded = load_matcher(path)

    assert np.array_equal(loaded.predict_proba(test_set.pairs)[:, 1], scores)


def test_save_creates_directories_that_do_not_exist_yet(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "nested" / "dir" / "matcher.joblib"

    written = save_matcher(matcher, manifest={}, path=path)

    assert written == path
    assert path.is_file()


def test_save_returns_the_path_it_wrote_to(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    assert save_matcher(matcher, manifest={}, path=path) == path


# --- the manifest ----------------------------------------------------------


def test_the_manifest_carries_what_was_passed_plus_the_interpreter(
    fitted_and_scores, tmp_path: Path
) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    save_matcher(matcher, manifest={"training_label": "advanced", "test_share": 0.25}, path=path)
    manifest = load_manifest(path)

    assert manifest["training_label"] == "advanced"
    assert manifest["test_share"] == 0.25
    assert manifest["python"] == platform.python_version()
    assert manifest["implementation"] == platform.python_implementation()


def test_the_real_interpreter_wins_over_a_caller_supplied_one(
    fitted_and_scores, tmp_path: Path
) -> None:
    """The interpreter is truth about the run, not a value a caller can shadow."""
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    save_matcher(matcher, manifest={"python": "bogus", "implementation": "bogus"}, path=path)
    manifest = load_manifest(path)

    assert manifest["python"] == platform.python_version()
    assert manifest["implementation"] == platform.python_implementation()


def test_the_manifest_file_is_the_one_beside_the_matcher(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    save_matcher(matcher, manifest={}, path=path)

    assert manifest_path_for(path).is_file()


def test_the_manifest_is_written_with_line_feeds_only(fitted_and_scores, tmp_path: Path) -> None:
    """Written on Windows too, and read back on whatever regenerates it next."""
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    save_matcher(matcher, manifest={"note": "line one\nline two"}, path=path)

    assert b"\r" not in manifest_path_for(path).read_bytes()


# --- missing files -----------------------------------------------------


def test_loading_a_missing_matcher_names_the_path_and_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "nope.joblib"

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py") as excinfo:
        load_matcher(path)
    assert str(path) in str(excinfo.value)


def test_loading_a_missing_manifest_names_the_path_and_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "nope.joblib"

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py") as excinfo:
        load_manifest(path)
    assert str(manifest_path_for(path)) in str(excinfo.value)


def test_a_matcher_without_its_manifest_still_loads(fitted_and_scores, tmp_path: Path) -> None:
    """The two files are independent: a stray matcher without a manifest is still usable."""
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"
    save_matcher(matcher, manifest={}, path=path)
    manifest_path_for(path).unlink()

    load_matcher(path)  # does not raise
    with pytest.raises(MatcherUnavailable):
        load_manifest(path)


# --- version -------------------------------------------------------------


def test_the_version_is_a_hex_digest_of_the_configured_length(
    fitted_and_scores, tmp_path: Path
) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"
    save_matcher(matcher, manifest={}, path=path)

    version = matcher_version(path)

    assert len(version) == VERSION_DIGEST_LENGTH
    assert all(char in "0123456789abcdef" for char in version)


def test_the_version_is_the_sha256_of_the_files_bytes(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"
    save_matcher(matcher, manifest={}, path=path)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()[:VERSION_DIGEST_LENGTH]
    assert matcher_version(path) == expected


def test_identical_bytes_give_the_same_version(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    save_matcher(matcher, manifest={}, path=first)
    second.write_bytes(first.read_bytes())

    assert matcher_version(first) == matcher_version(second)


def test_different_bytes_give_a_different_version(tmp_path: Path) -> None:
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    assert matcher_version(first) != matcher_version(second)


def test_the_version_is_unaffected_by_the_manifest(fitted_and_scores, tmp_path: Path) -> None:
    """Only the matcher file's own bytes matter -- the manifest lives in a separate file."""
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"
    save_matcher(matcher, manifest={"note": "first"}, path=path)
    before = matcher_version(path)

    save_matcher(matcher, manifest={"note": "second, and much longer"}, path=path)

    assert matcher_version(path) == before


def test_asking_for_a_missing_matchers_version_names_the_path_and_the_fix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nope.joblib"

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py") as excinfo:
        matcher_version(path)
    assert str(path) in str(excinfo.value)
