"""Unit tests for scripts/audit_fairness.py, the fairness audit CLI. No database needed."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, corpus_summary, generate_corpus
from app.ml.train import TrainingData, train_matcher, training_data

SCRIPT = REPO_ROOT / "scripts" / "audit_fairness.py"
#: A small corpus, as the other script tests use, and few resamples, to stay fast.
CORPUS = ["--seed", "5", "--jobs", "40", "--candidates", "1200"]
CONFIG = CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)
FAST = ["--resamples", "50"]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load(SCRIPT, "audit_fairness")


@pytest.fixture(scope="module")
def data() -> TrainingData:
    return training_data(generate_corpus(CONFIG))


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("audit") / "audit.json"
    assert script.main([*CORPUS, *FAST, "--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --- a normal run ------------------------------------------------------


def test_it_audits_every_job_and_says_the_data_is_synthetic(results: dict) -> None:
    assert results["synthetic"] is True
    assert results["jobs"] == 40
    assert results["pairs"] == 2000
    assert results["folds"] == 4
    assert results["training_label"] == "advanced"
    assert set(results["scorers"]) == {
        "matcher",
        "fitted_on_relevance",
        "trained_without_historical_bias",
    }
    for counts in results["group_pairs"].values():
        assert sum(counts.values()) == 2000


def test_every_scorer_is_audited_by_both_attributes_at_every_cut_off(results: dict) -> None:
    for report in results["scorers"].values():
        assert set(report["at_k"]) == {"5", "10"}
        for at_k in report["at_k"].values():
            assert at_k["gender"]["reference"] == "man"
            assert set(at_k["gender"]["groups"]) == {"man", "nonbinary", "woman"}
            assert at_k["age_band"]["reference"] == "under_40"
            assert set(at_k["age_band"]["groups"]) == {"40_and_over", "under_40"}


def test_a_shortlist_of_k_takes_k_from_every_slate(results: dict) -> None:
    # 40 slates of 50: a top 5 shortlists 200 pairs, a top 10 400.
    for report in results["scorers"].values():
        for k in (5, 10):
            groups = report["at_k"][str(k)]["gender"]["groups"].values()
            assert sum(group["rates"]["selected"] for group in groups) == pytest.approx(40 * k)


def test_the_historical_ratio_is_the_one_the_corpus_summary_measures(results: dict) -> None:
    summary = corpus_summary(generate_corpus(CONFIG))
    woman = results["historical_decisions"]["gender"]["groups"]["woman"]

    assert woman["disparate_impact"]["value"] == pytest.approx(summary.women_to_men_advance_ratio)


def test_the_proxy_line_is_the_only_thing_that_moves_a_score(results: dict, data) -> None:
    proxy = results["proxy"]
    planted = sum(
        candidate.gender == "woman" and "women" in candidate.resume_text.lower()
        for _, candidate in data.pairs
    )

    assert proxy["pairs_with_line"] == planted > 0
    assert proxy["max_abs_change_elsewhere"] == 0.0


def test_the_proxy_report_compares_the_same_pairs_scored_both_ways(monkeypatch) -> None:
    # Two slates of four; C1 carries the line and sits on both.
    women = {"C1", "C3", "C6"}
    people = {
        ref: SimpleNamespace(
            ref=ref, gender="woman" if ref in women else "man", age_band="under_40"
        )
        for ref in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
    }
    slates = {"J1": ["C1", "C2", "C3", "C4"], "J2": ["C1", "C5", "C6", "C7"]}
    pairs = [
        (SimpleNamespace(ref=job), people[ref]) for job, refs in slates.items() for ref in refs
    ]
    data = TrainingData(
        pairs=tuple(pairs),
        advanced=np.zeros(8, dtype=bool),
        relevance=np.ones(8, dtype=int),
        job_refs=tuple(job.ref for job, _ in pairs),
    )
    with_line = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.6, 0.1])
    without_line = with_line.copy()
    without_line[[0, 4]] = 0.95
    swapped = np.array([True, False, False, False, True, False, False, False])
    monkeypatch.setattr(script, "CUTOFFS", (1,))

    report = script.proxy_report(data, with_line, without_line, swapped, resamples=10)

    assert report["pairs_with_line"] == 2
    assert report["candidates_with_line"] == 1
    assert report["mean_score_change"] == pytest.approx(-0.75)
    assert report["median_score_change"] == pytest.approx(-0.75)
    assert report["share_of_pairs_scored_lower"] == 1.0
    assert report["max_abs_change_elsewhere"] == 0.0
    top = report["at_k"]["1"]
    # With the line each slate's top is a man; without it, C1 tops both.
    assert top["selection_rate_with_line"] == 0.0
    assert top["selection_rate_without_line"] == 1.0
    assert top["women_selection_rate_with_line"] == 0.0
    assert top["women_selection_rate_without_line"] == pytest.approx(0.5)
    rates = top["gender_without_line"]["groups"]["woman"]["rates"]
    assert rates["selection_rate"] == pytest.approx(0.5)


# --- folds and out-of-fold scoring --------------------------------------


def test_folds_deal_every_job_once_in_near_equal_shares(data) -> None:
    fold_of = script.job_folds(data.job_refs, 4, seed=0)

    assert set(fold_of) == set(data.job_refs)
    sizes = np.bincount(list(fold_of.values()))
    assert sizes.tolist() == [10, 10, 10, 10]


def test_folds_depend_on_the_seed_and_not_on_the_order_pairs_arrive_in(data) -> None:
    refs = list(data.job_refs)

    assert script.job_folds(refs, 4, seed=0) == script.job_folds(refs[::-1], 4, seed=0)
    assert script.job_folds(refs, 4, seed=0) != script.job_folds(refs, 4, seed=1)


@pytest.mark.parametrize("folds", [1, 41])
def test_a_fold_count_that_cannot_hold_out_jobs_is_refused(data, folds: int) -> None:
    with pytest.raises(script.AuditRefused, match="--folds"):
        script.job_folds(data.job_refs, folds, seed=0)


class _SpyModel:
    """Scores each pair with the id of the model, after noting what it was trained on."""

    def __init__(self, ident: int) -> None:
        self.ident = ident

    def predict_proba(self, pairs: list) -> np.ndarray:
        return np.column_stack([np.zeros(len(pairs)), np.full(len(pairs), float(self.ident))])


def test_each_job_is_scored_by_a_model_that_never_trained_on_it(data) -> None:
    fold_of = script.job_folds(data.job_refs, 4, seed=0)
    trained_on: list[set[str]] = []

    def fit(subset: TrainingData) -> _SpyModel:
        trained_on.append(set(subset.job_refs))
        return _SpyModel(len(trained_on) - 1)

    (scores,) = script.out_of_fold_scores(data, fold_of, fit)

    assert len(trained_on) == 4
    for ref, score in zip(data.job_refs, scores, strict=True):
        assert ref not in trained_on[int(score)]
        # Trained on every other fold.
        assert len(trained_on[int(score)]) == 30


def test_other_versions_of_the_pairs_are_scored_by_the_same_models(data) -> None:
    fold_of = script.job_folds(data.job_refs, 4, seed=0)
    count = iter(range(100))

    first, second = script.out_of_fold_scores(
        data, fold_of, lambda _: _SpyModel(next(count)), also_score=[data]
    )

    assert first.tolist() == second.tolist()


class _LengthModel:
    """Scores each pair by its resume's length, so each version scores as itself."""

    def predict_proba(self, pairs: list) -> np.ndarray:
        lengths = np.array([len(candidate.resume_text) for _, candidate in pairs], dtype=float)
        return np.column_stack([np.zeros(len(pairs)), lengths])


def test_each_version_of_the_pairs_is_scored_as_itself(data) -> None:
    fold_of = script.job_folds(data.job_refs, 4, seed=0)
    neutral = training_data(generate_corpus(dataclasses.replace(CONFIG, proxy_share=0.0)))

    first, second = script.out_of_fold_scores(
        data, fold_of, lambda _: _LengthModel(), also_score=[neutral]
    )

    assert first.tolist() == [len(c.resume_text) for _, c in data.pairs]
    assert second.tolist() == [len(c.resume_text) for _, c in neutral.pairs]
    assert first.tolist() != second.tolist()


def test_the_matcher_is_fitted_on_the_past_decisions(data) -> None:
    train, test = data.pairs[:1500], data.pairs[1500:]
    subset = script._subset(data, np.arange(len(data)) < 1500)
    ours = script.fit_on_advanced(subset).predict_proba(test)[:, 1]
    shipped = train_matcher(subset).predict_proba(test)[:, 1]

    assert len(train) == len(subset)
    assert ours.tolist() == shipped.tolist()


def test_the_reference_is_fitted_on_ground_truth(data, monkeypatch) -> None:
    seen: list[np.ndarray] = []
    real_fit = script.make_matcher().__class__.fit

    def spy(self, pairs, labels):
        seen.append(np.asarray(labels))
        return real_fit(self, pairs, labels)

    monkeypatch.setattr(script.make_matcher().__class__, "fit", spy)
    script.fit_on_relevance(data)

    assert seen[0].tolist() == (data.relevance >= 1).tolist()


def test_the_unbiased_reference_learns_from_the_unbiased_record(monkeypatch) -> None:
    # Only what each model is trained on matters here, so no model is fitted.
    labels: list[int] = []

    def spy(subset: TrainingData) -> Any:
        labels.append(int(subset.advanced.sum()))
        return _SpyModel(0)

    monkeypatch.setattr(script, "fit_on_advanced", spy)
    monkeypatch.setattr(script, "fit_on_relevance", lambda _: _SpyModel(0))
    script.audit(CONFIG, folds=2, fold_seed=0, resamples=10)

    biased = training_data(generate_corpus(CONFIG))
    unbiased = training_data(generate_corpus(dataclasses.replace(CONFIG, historical_bias=0.0)))
    fold_of = script.job_folds(biased.job_refs, 2, seed=0)
    folds = np.array([fold_of[ref] for ref in biased.job_refs])
    expected = [int(d.advanced[folds != fold].sum()) for d in (biased, unbiased) for fold in (0, 1)]

    # Two folds for the matcher, then two for the unbiased reference.
    assert labels == expected
    assert sum(labels[2:]) > sum(labels[:2])


# --- refusals ----------------------------------------------------------


def test_a_corpus_whose_counterfactual_changes_a_label_is_refused(monkeypatch) -> None:
    real = script.generate_corpus

    def drifting(config: CorpusConfig):
        corpus = real(config)
        if config.proxy_share == 0.0:
            first = corpus.pairs[0]
            flipped = dataclasses.replace(first, advanced=not first.advanced)
            corpus = dataclasses.replace(corpus, pairs=(flipped, *corpus.pairs[1:]))
        return corpus

    monkeypatch.setattr(script, "generate_corpus", drifting)

    with pytest.raises(script.AuditRefused, match="proxy_share=0"):
        script.audit(CONFIG, folds=2, fold_seed=0, resamples=10)


def test_an_unbiased_corpus_with_other_ground_truth_is_refused(monkeypatch) -> None:
    real = script.generate_corpus

    def drifting(config: CorpusConfig):
        corpus = real(config)
        if config.historical_bias == 0.0:
            first = corpus.pairs[0]
            moved = dataclasses.replace(first, relevance=(first.relevance + 1) % 3)
            corpus = dataclasses.replace(corpus, pairs=(moved, *corpus.pairs[1:]))
        return corpus

    monkeypatch.setattr(script, "generate_corpus", drifting)

    with pytest.raises(script.AuditRefused, match="historical_bias=0"):
        script.audit(CONFIG, folds=2, fold_seed=0, resamples=10)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--resamples", "0"], "--resamples"),
        (["--folds", "1"], "--folds"),
        (["--jobs", "0"], "at least 1"),
    ],
)
def test_bad_arguments_exit_2_with_a_reason(
    arguments: list[str], message: str, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "audit.json"

    assert script.main([*CORPUS, *arguments, "--out", str(out)]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_an_unwritable_output_exits_2_before_the_audit_runs(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(script, "audit", lambda *a, **k: pytest.fail("the audit ran"))

    code = script.main([*CORPUS, *FAST, "--folds", "2", "--out", str(blocker / "audit.json")])

    assert code == 2
    assert "cannot write" in capsys.readouterr().err
