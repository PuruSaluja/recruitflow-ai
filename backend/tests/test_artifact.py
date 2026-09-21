"""Unit tests for persisting the matcher: the round trip and the missing-file errors.

No database needed.
"""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.ml.artifact import (
    MatcherUnavailable,
    default_matcher_path,
    load_manifest,
    load_matcher,
    manifest_path_for,
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


def test_the_manifest_file_is_the_one_beside_the_matcher(fitted_and_scores, tmp_path: Path) -> None:
    matcher, _, _ = fitted_and_scores
    path = tmp_path / "matcher.joblib"

    save_matcher(matcher, manifest={}, path=path)

    assert manifest_path_for(path).is_file()


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
