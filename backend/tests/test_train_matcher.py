"""Unit tests for scripts/train_matcher.py, the training CLI. No database needed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from app.config import REPO_ROOT
from app.ml.artifact import load_manifest, load_matcher, manifest_path_for
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.train import TrainingData, split_by_job, train_matcher, training_data

SCRIPT = REPO_ROOT / "scripts" / "train_matcher.py"
#: Small enough to stay fast; big enough to fill a slate and hold both labels
#: past the calibration-fold floor the script checks for.
BASE = ["--seed", "5", "--jobs", "40", "--candidates", "1200"]


def _load() -> ModuleType:
    """Not on the import path; load it from its file, as generate_corpus.py's test does."""
    spec = importlib.util.spec_from_file_location("train_matcher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    assert script.main([*BASE, "--out", str(out)]) == 0
    return out


# --- a normal run ------------------------------------------------------


def test_it_writes_the_matcher_and_its_manifest(written: Path) -> None:
    assert written.is_file()
    assert manifest_path_for(written).is_file()


def test_the_written_matcher_scores_like_training_the_module_directly(written: Path) -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    train_set, test_set = split_by_job(training_data(corpus))
    expected = train_matcher(train_set).predict_proba(test_set.pairs)[:, 1]

    loaded = load_matcher(written)

    assert np.array_equal(loaded.predict_proba(test_set.pairs)[:, 1], expected)


def test_the_manifest_records_the_corpus_config_the_split_and_the_label(written: Path) -> None:
    manifest = load_manifest(written)

    assert manifest["corpus_config"] == {
        "seed": 5,
        "n_jobs": 40,
        "n_candidates": 1200,
        "slate_size": 50,
        "same_family_share": 0.4,
        "historical_bias": 0.25,
        "proxy_share": 0.6,
    }
    assert manifest["training_label"] == "advanced"
    assert manifest["calibration_method"] == "sigmoid"
    assert manifest["test_share"] == 0.25
    assert manifest["split_seed"] == 0


def test_the_manifest_declares_the_corpus_synthetic(written: Path) -> None:
    """Same disclosure the corpus's own manifest carries; honesty rule 2."""
    assert load_manifest(written)["synthetic"] is True


def test_the_manifest_row_counts_are_exact_not_just_the_right_shape(written: Path) -> None:
    """Measured directly: this corpus/split gives 1,500 training pairs across 30 jobs."""
    assert load_manifest(written)["rows"] == {
        "train_pairs": 1500,
        "test_pairs": 500,
        "train_jobs": 30,
        "test_jobs": 10,
    }


def test_the_manifest_metrics_are_measured_not_guessed(written: Path) -> None:
    manifest = load_manifest(written)
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    train_set, test_set = split_by_job(training_data(corpus))
    scores = load_matcher(written).predict_proba(test_set.pairs)[:, 1]

    metrics = manifest["metrics"]
    assert metrics["mean_predicted_probability"] == pytest.approx(float(scores.mean()))
    assert metrics["base_rate"] == pytest.approx(float(train_set.advanced.mean()))
    assert 0.0 <= metrics["roc_auc_advanced"] <= 1.0


# --- the auc helper ---------------------------------------------------


def test_safe_auc_is_none_when_the_held_out_set_has_only_one_class() -> None:
    assert script._safe_auc([True, True, True], [0.1, 0.4, 0.9]) is None


def test_safe_auc_scores_normally_with_both_classes_present() -> None:
    assert script._safe_auc([True, False], [0.9, 0.1]) == pytest.approx(1.0)


# --- the untrainable-reason helper, exactly at its boundaries ---------


def _fake_training_data(n_pos: int, n_neg: int) -> TrainingData:
    """A ``TrainingData`` with the given class counts; the pairs and job_refs are dummies.

    ``_untrainable_reason`` reads only ``advanced`` and the row count, so this
    pins its two thresholds exactly, without depending on a corpus/seed search
    landing on the right shape by luck.
    """
    n = n_pos + n_neg
    return TrainingData(
        pairs=tuple((None, None) for _ in range(n)),
        advanced=np.array([True] * n_pos + [False] * n_neg),
        relevance=np.zeros(n, dtype=int),
        job_refs=tuple(f"J{i}" for i in range(n)),
    )


def test_untrainable_reason_is_none_for_a_well_balanced_split() -> None:
    assert script._untrainable_reason(_fake_training_data(50, 50)) is None


def test_untrainable_reason_names_whichever_label_never_occurs() -> None:
    assert "nobody in the training split was ever advanced" in script._untrainable_reason(
        _fake_training_data(0, 10)
    )
    assert "nobody in the training split was ever rejected" in script._untrainable_reason(
        _fake_training_data(10, 0)
    )


def test_untrainable_reason_is_none_right_at_the_calibration_floor() -> None:
    """5 matches CalibratedClassifierCV's default fold count (pinned in test_train.py)."""
    assert script._untrainable_reason(_fake_training_data(5, 50)) is None
    assert script._untrainable_reason(_fake_training_data(50, 5)) is None


def test_untrainable_reason_fires_just_under_the_calibration_floor() -> None:
    assert script._untrainable_reason(_fake_training_data(4, 50)) is not None
    assert script._untrainable_reason(_fake_training_data(50, 4)) is not None


def test_rerunning_with_the_same_arguments_gives_the_same_matcher(tmp_path: Path) -> None:
    first, second = tmp_path / "a.joblib", tmp_path / "b.joblib"

    assert script.main([*BASE, "--out", str(first)]) == 0
    assert script.main([*BASE, "--out", str(second)]) == 0

    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    _, test_set = split_by_job(training_data(corpus))
    scores_a = load_matcher(first).predict_proba(test_set.pairs)[:, 1]
    scores_b = load_matcher(second).predict_proba(test_set.pairs)[:, 1]
    assert np.array_equal(scores_a, scores_b)


# --- refusing rather than shipping a bad matcher ----------------------


def test_a_split_where_nobody_was_ever_advanced_is_refused_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measured: seed 1 with this shape has an all-unadvanced training split."""
    args = ["--seed", "1", "--jobs", "4", "--candidates", "200", "--slate-size", "6"]

    code = script.main([*args, "--out", str(tmp_path / "matcher.joblib")])

    assert code == 2
    err = capsys.readouterr().err
    assert "nobody in the training split was ever advanced" in err
    assert not (tmp_path / "matcher.joblib").exists()


def test_a_minority_class_too_small_to_calibrate_is_refused_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measured: seed 1 with this shape trains but calibrates on a 2-example minority class."""
    args = ["--seed", "1", "--jobs", "4", "--candidates", "200", "--slate-size", "10"]

    code = script.main([*args, "--out", str(tmp_path / "matcher.joblib")])

    assert code == 2
    err = capsys.readouterr().err
    assert "fewer than the 5 calibration folds need" in err
    assert not (tmp_path / "matcher.joblib").exists()


@pytest.mark.parametrize("test_share", ["nan", "0.0", "1.0", "-0.2", "1.2"])
def test_an_invalid_test_share_is_refused_by_its_own_check(
    test_share: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Previously this reached ``round()``'s "cannot convert float NaN to integer" instead."""
    code = script.main(
        [*BASE, "--test-share", test_share, "--out", str(tmp_path / "matcher.joblib")]
    )

    assert code == 2
    assert "--test-share must be a number between 0 and 1" in capsys.readouterr().err


def test_a_corpus_that_cannot_exist_is_refused_with_the_generators_own_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = script.main(
        ["--candidates", "30", "--slate-size", "50", "--out", str(tmp_path / "matcher.joblib")]
    )

    assert code == 2
    assert "slate_size cannot be larger" in capsys.readouterr().err


def test_a_test_share_too_extreme_for_the_job_count_is_refused_with_the_splits_own_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [*BASE, "--test-share", "0.001", "--out", str(tmp_path / "matcher.joblib")]

    code = script.main(args)

    assert code == 2
    assert "at least one job on each side" in capsys.readouterr().err


def test_an_out_path_it_cannot_write_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory where the matcher should go used to end in a PermissionError traceback."""
    out = tmp_path / "matcher.joblib"
    out.mkdir()

    assert script.main([*BASE, "--out", str(out)]) == 2
    assert f"cannot write {out}" in capsys.readouterr().err


def test_it_runs_as_a_command(tmp_path: Path) -> None:
    """Imported and invoked are different things; this is the documented way to run it."""
    out = tmp_path / "matcher.joblib"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *BASE, "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "held-out ROC-AUC vs advanced:" in result.stdout
    assert out.is_file()
    assert (
        json.loads(manifest_path_for(out).read_text(encoding="utf-8"))["training_label"]
        == "advanced"
    )
