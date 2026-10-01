"""Unit tests for scripts/evaluate_matcher.py, the evaluation CLI. No database needed."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from app.config import REPO_ROOT
from app.ml.artifact import load_manifest, load_matcher, manifest_path_for, matcher_version
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.evaluate import bootstrap_interval, evaluate_scores
from app.ml.train import TrainingData, split_by_job, train_matcher, training_data

SCRIPT = REPO_ROOT / "scripts" / "evaluate_matcher.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_matcher.py"
#: The same small corpus test_train_matcher.py trains on.
CORPUS = ["--seed", "5", "--jobs", "40", "--candidates", "1200"]
#: Not the default 0, so a split seed taken from anywhere but the manifest shows.
SPLIT_SEED = 2
#: One extra split and few resamples, to stay fast.
FAST = ["--split-seeds", "1", "--resamples", "200"]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load(SCRIPT, "evaluate_matcher")


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    train = _load(TRAIN_SCRIPT, "train_matcher")
    assert train.main([*CORPUS, "--split-seed", str(SPLIT_SEED), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def results(model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("evaluation") / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _copy(model: Path, directory: Path) -> Path:
    copied = directory / model.name
    shutil.copy(model, copied)
    shutil.copy(manifest_path_for(model), manifest_path_for(copied))
    return copied


# --- a normal run ------------------------------------------------------


def test_it_measures_every_scorer_on_the_held_out_jobs(results: dict) -> None:
    assert set(results["scorers"]) == {
        "matcher",
        "random",
        "required_skill_coverage",
        "fitted_on_relevance",
    }
    assert results["held_out"]["jobs"] == 10
    assert results["held_out"]["pairs"] == 500
    assert results["training_label"] == "advanced"
    assert results["synthetic"] is True


def test_it_names_the_matcher_it_measured(model: Path, results: dict) -> None:
    assert results["matcher_version"] == matcher_version(model)


def test_the_matchers_auc_is_the_one_its_training_run_recorded(model: Path, results: dict) -> None:
    recorded = load_manifest(model)["metrics"]["roc_auc_advanced"]
    assert results["scorers"]["matcher"]["roc_auc_advanced"] == pytest.approx(recorded, abs=1e-12)


def _held_out() -> TrainingData:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    return split_by_job(training_data(corpus), seed=SPLIT_SEED)[1]


def test_random_precision_is_each_jobs_relevant_share_averaged(results: dict) -> None:
    test_set = _held_out()
    refs = np.array(test_set.job_refs)
    shares = [(test_set.relevance[refs == ref] >= 1).mean() for ref in set(test_set.job_refs)]

    random = results["scorers"]["random"]
    assert random["precision_at_5"] == pytest.approx(np.mean(shares))
    assert random["roc_auc_advanced"] == 0.5


def test_each_interval_brackets_its_mean(results: dict) -> None:
    for summary in results["scorers"].values():
        for metric in ("ndcg_at_10", "precision_at_5"):
            low, high = summary[f"{metric}_95ci"]
            assert low <= summary[metric] <= high


def test_the_intervals_resample_each_metrics_own_per_job_values(model: Path, results: dict) -> None:
    test_set = _held_out()
    evaluation = evaluate_scores(test_set, load_matcher(model).predict_proba(test_set.pairs)[:, 1])
    ndcgs = [slate.ndcg for slate in evaluation.slates if slate.ndcg is not None]
    precisions = [slate.precision for slate in evaluation.slates]

    matcher = results["scorers"]["matcher"]
    assert matcher["ndcg_at_10_95ci"] == list(bootstrap_interval(ndcgs, resamples=200))
    assert matcher["precision_at_5_95ci"] == list(bootstrap_interval(precisions, resamples=200))


def test_the_spread_covers_the_matchers_own_split_and_the_extra_ones(results: dict) -> None:
    assert [split["split_seed"] for split in results["splits"]] == [SPLIT_SEED, 1]
    assert results["splits"][0]["ndcg_at_10"] == results["scorers"]["matcher"]["ndcg_at_10"]
    spread = results["split_spread"]["ndcg_at_10"]
    values = [split["ndcg_at_10"] for split in results["splits"]]
    assert spread["mean"] == pytest.approx(np.mean(values))
    assert spread["min"] == min(values)
    assert spread["max"] == max(values)


def test_an_extra_split_is_retrained_on_its_own_jobs(results: dict) -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    train_set, test_set = split_by_job(training_data(corpus), seed=1)
    scores = train_matcher(train_set).predict_proba(test_set.pairs)[:, 1]

    assert results["splits"][1]["ndcg_at_10"] == evaluate_scores(test_set, scores).ndcg_at_10


def test_two_runs_write_the_same_file(model: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert script.main(["--model", str(model), "--out", str(first), *FAST]) == 0
    assert script.main(["--model", str(model), "--out", str(second), *FAST]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_no_extra_splits_leaves_one_split_and_no_spread(model: Path, tmp_path: Path) -> None:
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), "--split-seeds"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written["splits"]) == 1
    assert written["split_spread"]["ndcg_at_10"]["std"] == 0.0


# --- refusals ----------------------------------------------------------


def test_a_matcher_that_does_not_match_its_manifest_is_refused(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copied = _copy(model, tmp_path)
    manifest = json.loads(manifest_path_for(copied).read_text(encoding="utf-8"))
    manifest["split_seed"] = SPLIT_SEED + 1  # a different split from the one it was trained on
    manifest_path_for(copied).write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "evaluation.json"

    assert script.main(["--model", str(copied), "--out", str(out), *FAST]) == 2
    assert "was not trained on the split its manifest describes" in capsys.readouterr().err
    assert not out.exists()


def test_a_manifest_naming_another_label_is_refused(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copied = _copy(model, tmp_path)
    manifest = json.loads(manifest_path_for(copied).read_text(encoding="utf-8"))
    manifest["training_label"] = "relevance"
    manifest_path_for(copied).write_text(json.dumps(manifest), encoding="utf-8")

    assert script.main(["--model", str(copied), "--out", str(tmp_path / "e.json"), *FAST]) == 2
    assert "fitted on 'relevance'" in capsys.readouterr().err


def _edited(model: Path, directory: Path, edit: Callable[[dict], None]) -> Path:
    copied = _copy(model, directory)
    manifest = json.loads(manifest_path_for(copied).read_text(encoding="utf-8"))
    edit(manifest)
    manifest_path_for(copied).write_text(json.dumps(manifest), encoding="utf-8")
    return copied


def _drop(key: str) -> Callable[[dict], None]:
    return lambda manifest: manifest.pop(key)


def _set(key: str, value: object) -> Callable[[dict], None]:
    return lambda manifest: manifest.__setitem__(key, value)


def _set_corpus(key: str, value: object) -> Callable[[dict], None]:
    return lambda manifest: manifest["corpus_config"].__setitem__(key, value)


_UNUSABLE_MANIFESTS: dict[str, tuple[Callable[[dict], None], str]] = {
    "no corpus_config": (_drop("corpus_config"), "has no corpus_config object"),
    "corpus_config a list": (_set("corpus_config", []), "has no corpus_config object"),
    "unknown corpus key": (_set_corpus("colour", 1), "corpus_config this code cannot use"),
    "invalid corpus value": (_set_corpus("n_jobs", -3), "corpus_config this code cannot use"),
    "no test_share": (_drop("test_share"), "has no numeric test_share"),
    "test_share a string": (_set("test_share", "0.25"), "has no numeric test_share"),
    "no split_seed": (_drop("split_seed"), "has no integer split_seed"),
    "split_seed a float": (_set("split_seed", 2.0), "has no integer split_seed"),
    "split_seed a bool": (_set("split_seed", True), "has no integer split_seed"),
    "no metrics": (_drop("metrics"), "records no held-out ROC-AUC"),
    "recorded AUC null": (
        lambda manifest: manifest["metrics"].__setitem__("roc_auc_advanced", None),
        "records no held-out ROC-AUC",
    ),
    "recorded AUC a string": (
        lambda manifest: manifest["metrics"].__setitem__("roc_auc_advanced", "0.7"),
        "records no held-out ROC-AUC",
    ),
    "a split that holds out no job": (_set("test_share", 0.001), "cannot be built"),
}


@pytest.mark.parametrize("case", list(_UNUSABLE_MANIFESTS))
def test_a_manifest_it_cannot_use_is_refused_by_name(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    edit, message = _UNUSABLE_MANIFESTS[case]
    copied = _edited(model, tmp_path, edit)
    out = tmp_path / "evaluation.json"

    assert script.main(["--model", str(copied), "--out", str(out), *FAST]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_a_damaged_manifest_is_refused_naming_the_fix(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copied = _copy(model, tmp_path)
    manifest_path_for(copied).write_text("{not json", encoding="utf-8")

    assert script.main(["--model", str(copied), "--out", str(tmp_path / "e.json"), *FAST]) == 2
    assert "python scripts/train_matcher.py" in capsys.readouterr().err


@pytest.mark.parametrize(
    "seeds", [[str(SPLIT_SEED)], ["1", "1"], ["1", str(SPLIT_SEED)]], ids=["own", "twice", "both"]
)
def test_split_seeds_that_would_count_one_split_twice_are_refused(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], seeds: list[str]
) -> None:
    out = tmp_path / "evaluation.json"

    assert script.main(["--model", str(model), "--out", str(out), "--split-seeds", *seeds]) == 2
    assert "--split-seeds must be distinct" in capsys.readouterr().err
    assert not out.exists()


def test_an_out_path_it_cannot_write_is_refused(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "evaluation.json"
    out.mkdir()

    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 2
    assert f"cannot write {out}" in capsys.readouterr().err


def test_a_missing_matcher_names_the_command_that_makes_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nowhere.joblib"

    assert script.main(["--model", str(missing), "--out", str(tmp_path / "e.json")]) == 2
    assert "python scripts/train_matcher.py" in capsys.readouterr().err


def test_zero_resamples_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert script.main(["--out", str(tmp_path / "e.json"), "--resamples", "0"]) == 2
    assert "--resamples must be at least 1" in capsys.readouterr().err
