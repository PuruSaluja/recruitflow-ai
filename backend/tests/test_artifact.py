"""Unit tests for persisting the matcher: the round trip, and a missing or unusable file.

No database needed.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import platform
import sys
import types
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pytest

from app.config import get_settings
from app.ml.artifact import (
    VERSION_DIGEST_LENGTH,
    MatcherUnavailable,
    default_matcher_path,
    load_manifest,
    load_matcher,
    load_versioned_matcher,
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


# --- loading the matcher with its version ----------------------------------


def test_the_versioned_load_gives_the_saved_matcher_and_the_files_version(
    fitted_and_scores, tmp_path: Path
) -> None:
    matcher, test_set, scores = fitted_and_scores
    path = save_matcher(matcher, manifest={}, path=tmp_path / "matcher.joblib")

    loaded, version = load_versioned_matcher(path)

    assert np.array_equal(loaded.predict_proba(test_set.pairs)[:, 1], scores)
    assert version == matcher_version(path)


def test_the_matcher_and_its_version_come_from_one_read_of_the_file(
    fitted_and_scores, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reads would let a retrain landing in between pair one model with another's version."""
    matcher, _, _ = fitted_and_scores
    path = save_matcher(matcher, manifest={}, path=tmp_path / "matcher.joblib")
    reads: list[bytes] = []
    unpickled_from: list[object] = []
    real_read_bytes, real_load = Path.read_bytes, joblib.load

    def counting_read_bytes(self: Path) -> bytes:
        data = real_read_bytes(self)
        reads.append(data)
        return data

    def recording_load(source: object, *args: object, **kwargs: object) -> object:
        unpickled_from.append(source)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    monkeypatch.setattr(joblib, "load", recording_load)

    _, version = load_versioned_matcher(path)

    assert len(reads) == 1
    [source] = unpickled_from
    assert isinstance(source, io.BytesIO)
    assert source.getvalue() == reads[0]
    assert version == hashlib.sha256(reads[0]).hexdigest()[:VERSION_DIGEST_LENGTH]


def test_a_missing_matcher_is_unavailable_to_the_versioned_load_too(tmp_path: Path) -> None:
    path = tmp_path / "nope.joblib"

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py") as excinfo:
        load_versioned_matcher(path)
    assert str(path) in str(excinfo.value)


# --- a matcher file that exists but cannot be used -------------------------


def _pickle_of_a_class_that_has_since_moved(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """What an artifact looks like after the code it was pickled against is renamed."""
    module = types.ModuleType("vanished_module")

    class Thing:
        pass

    Thing.__module__ = module.__name__
    Thing.__qualname__ = "Thing"
    module.Thing = Thing
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data = pickle.dumps(Thing())
    monkeypatch.delitem(sys.modules, module.__name__)
    return data


_DAMAGE: dict[str, Callable[[bytes, pytest.MonkeyPatch], bytes]] = {
    "empty": lambda data, _: b"",
    "truncated to half": lambda data, _: data[: len(data) // 2],
    "last byte missing": lambda data, _: data[:-1],
    "not a pickle": lambda data, _: b"not a pickle at all\n" * 10,
    "pickled against moved code": lambda _, monkeypatch: _pickle_of_a_class_that_has_since_moved(
        monkeypatch
    ),
}


@pytest.mark.parametrize("damage", list(_DAMAGE))
def test_an_unreadable_matcher_is_unavailable_and_names_the_path_and_the_fix(
    fitted_and_scores, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    matcher, _, _ = fitted_and_scores
    path = save_matcher(matcher, manifest={}, path=tmp_path / "matcher.joblib")
    path.write_bytes(_DAMAGE[damage](path.read_bytes(), monkeypatch))

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py") as excinfo:
        load_versioned_matcher(path)
    assert str(path) in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_a_file_holding_something_other_than_a_matcher_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "matcher.joblib"
    joblib.dump({"not": "a matcher"}, path)

    with pytest.raises(MatcherUnavailable, match="holds a dict") as excinfo:
        load_versioned_matcher(path)
    assert str(path) in str(excinfo.value)
    assert "scripts/train_matcher.py" in str(excinfo.value)


_LOADERS: dict[str, Callable[[Path], object]] = {
    "load_matcher": load_matcher,
    "load_versioned_matcher": load_versioned_matcher,
    "matcher_version": matcher_version,
}


@pytest.mark.parametrize("loader", list(_LOADERS))
def test_a_matcher_path_that_cannot_be_read_is_unavailable(tmp_path: Path, loader: str) -> None:
    """A directory exists but cannot be read as a file; that used to escape as an OSError."""
    path = tmp_path / "matcher.joblib"
    path.mkdir()

    with pytest.raises(MatcherUnavailable, match="could not be read") as excinfo:
        _LOADERS[loader](path)
    assert str(path) in str(excinfo.value)
    assert "scripts/train_matcher.py" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_load_matcher_turns_an_unreadable_file_into_unavailable_as_well(tmp_path: Path) -> None:
    path = tmp_path / "matcher.joblib"
    path.write_bytes(b"not a pickle at all\n")

    with pytest.raises(MatcherUnavailable, match="scripts/train_matcher.py"):
        load_matcher(path)
