"""Unit tests for scripts/evaluate_matcher.py, the evaluation CLI. No database needed.

Every test but the ``real_model`` ones uses a stand-in encoder. Those load the
pinned model through the script's own ``load_encoder`` and are skipped where it
cannot be loaded, unless ``RECRUITFLOW_REQUIRE_ENCODER`` is set, as in CI's
``embeddings`` job, which runs them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fake_cross_encoder import FakeCrossEncoder, fake_relevance
from fake_encoder import FakeEncoder
from scipy.stats import rankdata, spearmanr

from app.config import REPO_ROOT
from app.ml.artifact import load_manifest, load_matcher, manifest_path_for, matcher_version
from app.ml.corpus import Corpus, CorpusConfig, generate_corpus
from app.ml.cross_encoder import DEFAULT_CROSS_ENCODER, DEFAULT_CROSS_ENCODER_REVISION
from app.ml.encoder import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Encoder,
    EncoderUnavailable,
    candidate_document,
    job_document,
)
from app.ml.evaluate import bootstrap_interval, evaluate_scores
from app.ml.retrieval import BM25Index, fuse_per_group
from app.ml.train import (
    TrainingData,
    make_matcher,
    split_by_job,
    train_matcher,
    training_data,
)

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
#: Taken before any test replaces them with the stand-ins.
REAL_LOAD_ENCODER = script.load_encoder
REAL_LOAD_CROSS_ENCODER = script.load_cross_encoder


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stand-in encoder and cross-encoder, so no test needs the real models."""
    monkeypatch.setattr(script, "load_encoder", FakeEncoder)
    monkeypatch.setattr(script, "load_cross_encoder", FakeCrossEncoder)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    train = _load(TRAIN_SCRIPT, "train_matcher")
    assert train.main([*CORPUS, "--split-seed", str(SPLIT_SEED), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def results(model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("evaluation") / "evaluation.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(script, "load_encoder", FakeEncoder)
        patch.setattr(script, "load_cross_encoder", FakeCrossEncoder)
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
        "bm25",
        "dense",
        "hybrid_rrf",
        "rerank",
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


def test_the_reference_scorer_is_fitted_on_relevance_not_on_advanced(results: dict) -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    train_set, test_set = split_by_job(training_data(corpus), seed=SPLIT_SEED)
    fitted = make_matcher().fit(train_set.pairs, train_set.relevance >= 1)
    expected = evaluate_scores(test_set, fitted.predict_proba(test_set.pairs)[:, 1])

    reference = results["scorers"]["fitted_on_relevance"]
    assert reference["ndcg_at_10"] == expected.ndcg_at_10
    assert reference["roc_auc_relevant"] == expected.roc_auc_relevant
    assert reference["ndcg_at_10"] != results["scorers"]["matcher"]["ndcg_at_10"]


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


# --- retrieval scorers -------------------------------------------------


def _bm25(corpus: Corpus, data: TrainingData) -> np.ndarray:
    index = BM25Index([candidate_document(c) for c in corpus.candidates])
    refs = [c.ref for c in corpus.candidates]
    return np.array(
        [index.scores(job_document(job))[refs.index(cand.ref)] for job, cand in data.pairs]
    )


def _fake_cosine(data: TrainingData) -> np.ndarray:
    encoder = FakeEncoder()
    out = []
    for job, candidate in data.pairs:
        j = encoder.encode([job_document(job)])[0].astype(float)
        c = encoder.encode([candidate_document(candidate)])[0].astype(float)
        out.append(j @ c / np.linalg.norm(j) / np.linalg.norm(c))
    return np.array(out)


def _measured(corpus: Corpus, data: TrainingData) -> dict[str, Any]:
    scores = {"bm25": _bm25(corpus, data), "dense": _fake_cosine(data)}
    scores["hybrid_rrf"] = fuse_per_group(data.job_refs, scores["bm25"], scores["dense"])
    measured: dict[str, Any] = {name: evaluate_scores(data, s) for name, s in scores.items()}
    measured["scores"], measured["job_refs"] = scores, np.array(data.job_refs)
    return measured


@pytest.fixture(scope="module")
def recomputed() -> dict[str, dict[str, Any]]:
    """Every retrieval score recomputed pair by pair, independently of the script."""
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    data = training_data(corpus)
    return {
        "all": _measured(corpus, data),
        "held_out": _measured(corpus, split_by_job(data, seed=SPLIT_SEED)[1]),
    }


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid_rrf"])
def test_each_retrieval_scorer_is_measured_on_the_held_out_jobs(
    results: dict, recomputed: dict, name: str
) -> None:
    expected = recomputed["held_out"][name]
    measured = results["scorers"][name]
    assert measured["ndcg_at_10"] == pytest.approx(expected.ndcg_at_10, abs=1e-12)
    assert measured["precision_at_5"] == pytest.approx(expected.precision_at_5, abs=1e-12)
    assert measured["roc_auc_relevant"] == pytest.approx(expected.roc_auc_relevant, abs=1e-12)


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid_rrf"])
def test_each_retrieval_scorer_is_measured_on_every_job_too(
    results: dict, recomputed: dict, name: str
) -> None:
    every = results["all_jobs"]
    assert (every["jobs"], every["pairs"]) == (40, 2000)
    expected = recomputed["all"][name]
    assert every["scorers"][name]["ndcg_at_10"] == pytest.approx(expected.ndcg_at_10, abs=1e-12)
    assert every["jobs_without_relevant"] == expected.jobs_without_relevant


def test_the_three_retrieval_scorers_rank_differently(results: dict) -> None:
    ndcgs = {results["scorers"][n]["ndcg_at_10"] for n in ("bm25", "dense", "hybrid_rrf")}
    assert len(ndcgs) == 3


@pytest.mark.parametrize("scope", ["held_out", "all"])
@pytest.mark.parametrize("other", ["bm25", "dense"])
def test_fusion_is_compared_with_each_side_job_by_job(
    results: dict, recomputed: dict, scope: str, other: str
) -> None:
    where = results if scope == "held_out" else results["all_jobs"]
    paired = where["comparisons"][f"hybrid_rrf_minus_{other}"]
    fused, alone = recomputed[scope]["hybrid_rrf"], recomputed[scope][other]
    pairs = list(zip(fused.slates, alone.slates, strict=True))
    ndcg = [a.ndcg - b.ndcg for a, b in pairs if a.ndcg is not None]
    precision = [a.precision - b.precision for a, b in pairs]

    assert paired["ndcg_at_10_difference"] == pytest.approx(np.mean(ndcg), abs=1e-12)
    assert paired["precision_at_5_difference"] == pytest.approx(np.mean(precision), abs=1e-12)
    assert paired["ndcg_at_10_difference_95ci"] == pytest.approx(
        bootstrap_interval(ndcg, resamples=200), abs=1e-12
    )
    assert paired["precision_at_5_difference_95ci"] == pytest.approx(
        bootstrap_interval(precision, resamples=200), abs=1e-12
    )
    assert paired["ndcg_at_10_jobs_better"] == sum(d > 0 for d in ndcg)
    assert paired["ndcg_at_10_jobs_tied"] == sum(d == 0 for d in ndcg)
    assert paired["ndcg_at_10_jobs_worse"] == sum(d < 0 for d in ndcg)


@pytest.mark.parametrize("scope", ["held_out", "all"])
def test_agreement_is_spearman_per_slate_and_wins_split_by_the_better_side(
    results: dict, recomputed: dict, scope: str
) -> None:
    agreement = results["agreement"] if scope == "held_out" else results["all_jobs"]["agreement"]
    measured = recomputed[scope]
    scores, refs = measured["scores"], measured["job_refs"]
    rhos = [
        spearmanr(scores["bm25"][refs == ref], scores["dense"][refs == ref]).statistic
        for ref in sorted(set(refs))
    ]
    spread = agreement["bm25_dense_spearman_per_slate"]
    assert spread["mean"] == pytest.approx(np.mean(rhos), abs=1e-12)
    assert (spread["min"], spread["max"]) == pytest.approx((min(rhos), max(rhos)), abs=1e-12)
    assert agreement["slates_without_a_correlation"] == 0
    fused = measured["scores"]["hybrid_rrf"]
    in_ties = sum(
        sum(n for n in Counter(fused[refs == ref].tolist()).values() if n > 1) for ref in set(refs)
    )
    assert agreement["hybrid_rrf_pairs_in_ties"] == in_ties

    level = {"jobs": 0, "hybrid_better_than_dense": 0}
    below = {"jobs": 0, "hybrid_better_than_dense": 0}
    for bm25, dense, hybrid in zip(
        measured["bm25"].slates,
        measured["dense"].slates,
        measured["hybrid_rrf"].slates,
        strict=True,
    ):
        side = level if bm25.ndcg >= dense.ndcg else below
        side["jobs"] += 1
        side["hybrid_better_than_dense"] += hybrid.ndcg > dense.ndcg
    assert agreement["ndcg_at_10_bm25_at_least_dense"] == level
    assert agreement["ndcg_at_10_bm25_below_dense"] == below
    assert level["jobs"] + below["jobs"] == len(rhos)


def test_a_slate_one_side_ranks_all_equal_has_no_correlation() -> None:
    assert script._spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None
    assert script._spearman(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) is None
    assert script._spearman(np.array([3.0, 1.0, 2.0]), np.array([30.0, 10.0, 20.0])) == 1.0


def test_agreement_on_three_hand_made_slates() -> None:
    # J0: BM25 ranks everyone equal, so it has no correlation. J1: both sides
    # put the strong fit first, so BM25 ties dense on NDCG and counts as at
    # least as good. J2: BM25 puts the strong fit second, below dense.
    refs = ("J0",) * 3 + ("J1",) * 3 + ("J2",) * 3
    data = TrainingData(
        pairs=tuple((None, None) for _ in refs),
        advanced=np.zeros(9, dtype=bool),
        relevance=np.array([2, 0, 0, 2, 0, 0, 0, 2, 0]),
        job_refs=refs,
    )
    scores = {
        "bm25": np.array([1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
        "dense": np.array([0.9, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.9, 0.5]),
    }
    scores["hybrid_rrf"] = fuse_per_group(refs, scores["bm25"], scores["dense"])
    results = {name: evaluate_scores(data, s) for name, s in scores.items()}

    agreement = script._agreement(data, scores, results)

    assert agreement["slates_without_a_correlation"] == 1
    j1 = spearmanr([3.0, 2.0, 1.0], [0.9, 0.1, 0.5]).statistic
    j2 = spearmanr([3.0, 2.0, 1.0], [0.1, 0.9, 0.5]).statistic
    assert agreement["bm25_dense_spearman_per_slate"]["mean"] == pytest.approx((j1 + j2) / 2)
    assert agreement["ndcg_at_10_bm25_at_least_dense"] == {
        "jobs": 1,
        "hybrid_better_than_dense": 0,
    }
    assert agreement["ndcg_at_10_bm25_below_dense"] == {"jobs": 2, "hybrid_better_than_dense": 0}
    # J1's second and third candidates rank 2nd and 3rd by BM25, 3rd and 2nd by
    # dense: swapped, so their fused scores tie. No other slate has a swap.
    assert agreement["hybrid_rrf_pairs_in_ties"] == 2


def test_a_slate_with_nobody_relevant_is_left_out_of_the_ndcg_comparison() -> None:
    """NDCG is undefined with nobody relevant to find, so the slate is in neither count."""
    refs = ("J1",) * 3 + ("J3",) * 3
    data = TrainingData(
        pairs=tuple((None, None) for _ in refs),
        advanced=np.zeros(6, dtype=bool),
        relevance=np.array([2, 0, 0, 0, 0, 0]),
        job_refs=refs,
    )
    scores = {
        "bm25": np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0]),
        "dense": np.array([0.9, 0.1, 0.5, 0.5, 0.1, 0.9]),
    }
    scores["hybrid_rrf"] = fuse_per_group(refs, scores["bm25"], scores["dense"])
    results = {name: evaluate_scores(data, s) for name, s in scores.items()}
    assert [slate.ndcg is None for slate in results["dense"].slates] == [False, True]

    agreement = script._agreement(data, scores, results)

    assert agreement["ndcg_at_10_bm25_at_least_dense"]["jobs"] == 1
    assert agreement["ndcg_at_10_bm25_below_dense"]["jobs"] == 0
    # Still correlated: the rankings exist whether or not anyone is relevant.
    assert agreement["slates_without_a_correlation"] == 0


# --- the first stage of a two-stage ranking --------------------------------


def _written_out_recall(scores: np.ndarray, hits: np.ndarray, n: int) -> tuple[float | None, int]:
    """Recall of one slate's top ``n``, ties at the cut kept, and how many it keeps."""
    ordered = sorted(scores.tolist(), reverse=True)
    threshold = ordered[min(n, len(ordered)) - 1]
    kept = [i for i in range(len(scores)) if scores[i] >= threshold]
    total = int(hits.sum())
    return (None if total == 0 else sum(bool(hits[i]) for i in kept) / total), len(kept)


def _expected_recall(
    scores: np.ndarray, refs: np.ndarray, relevance: np.ndarray, jobs: set
) -> dict:
    out: dict[str, dict[str, float]] = {
        "recall_at_n": {},
        "strong_recall_at_n": {},
        "mean_kept": {},
    }
    for n in script.RECALL_CUTOFFS:
        relevant, strong, kept = [], [], []
        for ref in sorted(jobs):
            mask = refs == ref
            r, k = _written_out_recall(scores[mask], relevance[mask] >= 1, n)
            sr, _ = _written_out_recall(scores[mask], relevance[mask] == 2, n)
            kept.append(k)
            relevant += [] if r is None else [r]
            strong += [] if sr is None else [sr]
        out["recall_at_n"][str(n)] = np.mean(relevant)
        out["strong_recall_at_n"][str(n)] = np.mean(strong)
        out["mean_kept"][str(n)] = np.mean(kept)
    return out


@pytest.mark.parametrize("scope", ["training_jobs", "held_out", "all_jobs"])
@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid_rrf"])
def test_first_stage_recall_is_each_slates_kept_share_averaged(
    results: dict, recomputed: dict, scope: str, name: str
) -> None:
    every = recomputed["all"]
    refs = every["job_refs"]
    held_out = set(recomputed["held_out"]["job_refs"].tolist())
    jobs = {
        "training_jobs": set(refs.tolist()) - held_out,
        "held_out": held_out,
        "all_jobs": set(refs.tolist()),
    }[scope]
    data = training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))
    expected = _expected_recall(every["scores"][name], refs, data.relevance, jobs)
    measured = results["first_stage"][scope][name]
    for key, values in expected.items():
        assert measured[key] == pytest.approx(values, abs=1e-12), key


@pytest.mark.parametrize("scope", ["training_jobs", "held_out", "all_jobs"])
def test_same_family_shares_are_each_slates_share_averaged(
    results: dict, recomputed: dict, scope: str
) -> None:
    every = recomputed["all"]
    refs = every["job_refs"]
    held_out = set(recomputed["held_out"]["job_refs"].tolist())
    jobs = {
        "training_jobs": set(refs.tolist()) - held_out,
        "held_out": held_out,
        "all_jobs": set(refs.tolist()),
    }[scope]
    data = training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))
    same = np.array([job.family == candidate.family for job, candidate in data.pairs])
    relevant = data.relevance >= 1
    measured = results["first_stage"]["same_family"][scope]
    slates = [refs == ref for ref in sorted(jobs)]
    assert measured["slate_share"] == pytest.approx(np.mean([same[m].mean() for m in slates]))
    assert measured["relevant_share"] == pytest.approx(
        np.mean([same[m & relevant].sum() / relevant[m].sum() for m in slates if relevant[m].any()])
    )
    for name in ("bm25", "dense", "hybrid_rrf"):
        for n in script.RECALL_CUTOFFS:
            shares = []
            for m in slates:
                scores = every["scores"][name][m]
                threshold = sorted(scores, reverse=True)[n - 1]
                shares.append(same[m][scores >= threshold].mean())
            assert measured["kept_share"][name][str(n)] == pytest.approx(np.mean(shares)), (name, n)


def test_same_family_shares_on_hand_made_slates() -> None:
    # Two slates of four. J0 is "a": BM25 ranks its two "a" candidates 1st and
    # 3rd, and the relevant one is the "b" at 2nd. J1 is "b", nobody relevant.
    a, b = SimpleNamespace(family="a"), SimpleNamespace(family="b")
    refs = ("J0",) * 4 + ("J1",) * 4
    pairs = ((a, a), (a, b), (a, a), (a, b), (b, b), (b, a), (b, a), (b, a))
    data = TrainingData(
        pairs=pairs,
        advanced=np.zeros(8, dtype=bool),
        relevance=np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        job_refs=refs,
    )
    scores = {"bm25": np.array([4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])}
    measured = script._same_family(data, scores)
    assert measured["slate_share"] == pytest.approx((2 / 4 + 1 / 4) / 2)
    # Only J0 has a relevant candidate, and it is out of the family.
    assert measured["relevant_share"] == 0.0
    # A slate of four is kept whole at N = 5, so the cut's share is the slate's.
    assert measured["kept_share"]["bm25"]["5"] == pytest.approx((2 / 4 + 1 / 4) / 2)


def test_same_family_slate_share_is_averaged_per_slate_not_per_pair() -> None:
    # A slate of four with one in the family, and one of two with both: per
    # slate (1/4 + 1) / 2, not 3 of 6 pairs.
    a, b = SimpleNamespace(family="a"), SimpleNamespace(family="b")
    data = TrainingData(
        pairs=((a, a), (a, b), (a, b), (a, b), (b, b), (b, b)),
        advanced=np.zeros(6, dtype=bool),
        relevance=np.zeros(6, dtype=int),
        job_refs=("J0",) * 4 + ("J1",) * 2,
    )
    measured = script._same_family(data, {"bm25": np.arange(6.0)})
    assert measured["slate_share"] == pytest.approx((1 / 4 + 1) / 2)


def test_same_family_cut_shares_at_small_n_on_hand_made_slates() -> None:
    a, b = SimpleNamespace(family="a"), SimpleNamespace(family="b")
    pairs = tuple((a, c) for c in (a, b, a, b, b, b, b, b, b, b))
    data = TrainingData(
        pairs=pairs,
        advanced=np.zeros(10, dtype=bool),
        relevance=np.zeros(10, dtype=int),
        job_refs=("J0",) * 10,
    )
    # The two "a" candidates rank 1st and 3rd, so the top 5 is 2 of 5 family.
    scores = {"bm25": np.arange(10.0, 0.0, -1.0)}
    measured = script._same_family(data, scores)
    assert measured["kept_share"]["bm25"]["5"] == pytest.approx(2 / 5)
    assert measured["kept_share"]["bm25"]["10"] == pytest.approx(2 / 10)
    assert measured["relevant_share"] is None


def test_first_stage_recall_on_hand_made_slates_leaves_out_a_job_with_nobody_to_keep() -> None:
    # 10 candidates a slate. J0: relevant at BM25 ranks 1 and 7, the strong fit
    # at 7. J1: nobody relevant, so it has no recall and does not count as 0.
    # J2: everyone tied, so every cut keeps the whole slate.
    refs = ("J0",) * 10 + ("J1",) * 10 + ("J2",) * 10
    relevance = np.zeros(30, dtype=int)
    relevance[[0, 6]] = [1, 2]
    relevance[[22]] = 1
    data = TrainingData(
        pairs=tuple((None, None) for _ in refs),
        advanced=np.zeros(30, dtype=bool),
        relevance=relevance,
        job_refs=refs,
    )
    scores = np.concatenate([np.arange(10.0, 0.0, -1.0), np.arange(10.0), np.zeros(10)])
    measured = script._first_stage_recall(data, {"bm25": scores})["bm25"]
    assert measured["recall_at_n"]["5"] == pytest.approx((0.5 + 1.0) / 2)
    assert measured["recall_at_n"]["10"] == 1.0
    assert measured["strong_recall_at_n"]["5"] == 0.0
    assert measured["mean_kept"]["5"] == pytest.approx((5 + 5 + 10) / 3)


def test_first_stage_recall_of_a_grade_no_job_has_is_none_not_nan() -> None:
    # No strong fit on any slate, and on the second set nobody relevant at all.
    refs = ("J0",) * 4 + ("J1",) * 4
    scores = {"bm25": np.array([4.0, 3.0, 2.0, 1.0] * 2)}
    for relevance in ([1, 0, 0, 1, 0, 0, 1, 0], [0] * 8):
        data = TrainingData(
            pairs=tuple((None, None) for _ in refs),
            advanced=np.zeros(8, dtype=bool),
            relevance=np.array(relevance),
            job_refs=refs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            measured = script._first_stage_recall(data, scores)["bm25"]
        assert set(measured["strong_recall_at_n"].values()) == {None}
    assert set(measured["recall_at_n"].values()) == {None}
    assert script.choose_first_stage({"bm25": measured, "dense": measured}) == {
        "scorer": None,
        "n": None,
        "target_met": False,
        "recall_on_training_jobs": None,
    }


def test_the_written_json_is_strict(model: Path, results: dict, tmp_path: Path) -> None:
    def refuse(constant: str) -> None:
        raise ValueError(f"{constant} is not JSON")

    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 0
    json.loads(out.read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.mark.parametrize("scope", ["training_jobs", "held_out", "all_jobs"])
def test_the_slates_relevant_counts_and_recall_ceilings_are_recorded(
    results: dict, recomputed: dict, scope: str
) -> None:
    refs = recomputed["all"]["job_refs"]
    held_out = set(recomputed["held_out"]["job_refs"].tolist())
    jobs = {
        "training_jobs": set(refs.tolist()) - held_out,
        "held_out": held_out,
        "all_jobs": set(refs.tolist()),
    }[scope]
    data = training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))
    counts = [int((data.relevance[refs == ref] >= 1).sum()) for ref in sorted(jobs)]
    slates = results["first_stage"]["slates"][scope]
    assert slates["jobs"] == len(jobs)
    assert slates["relevant_per_slate"] == {
        "mean": pytest.approx(np.mean(counts), abs=1e-12),
        "min": min(counts),
        "max": max(counts),
    }
    for n in script.RECALL_CUTOFFS:
        ceiling = np.mean([min(1.0, n / c) for c in counts if c > 0])
        assert slates["recall_ceiling_at_n"][str(n)] == pytest.approx(ceiling, abs=1e-12)


def test_a_set_with_nobody_relevant_has_no_recall_ceiling() -> None:
    data = TrainingData(
        pairs=tuple((None, None) for _ in range(4)),
        advanced=np.zeros(4, dtype=bool),
        relevance=np.zeros(4, dtype=int),
        job_refs=("J0", "J0", "J1", "J1"),
    )
    slates = script._slates(data)
    assert slates["relevant_per_slate"] == {"mean": 0.0, "min": 0, "max": 0}
    assert set(slates["recall_ceiling_at_n"].values()) == {None}


def test_a_job_with_nobody_relevant_counts_in_the_slates_but_not_the_ceiling() -> None:
    # J0 has 10 relevant of 12, J1 none: the counts include J1, the ceiling not.
    relevance = np.zeros(24, dtype=int)
    relevance[:10] = 1
    data = TrainingData(
        pairs=tuple((None, None) for _ in range(24)),
        advanced=np.zeros(24, dtype=bool),
        relevance=relevance,
        job_refs=("J0",) * 12 + ("J1",) * 12,
    )
    slates = script._slates(data)
    assert slates["relevant_per_slate"] == {"mean": 5.0, "min": 0, "max": 10}
    assert slates["recall_ceiling_at_n"]["5"] == 0.5
    assert slates["recall_ceiling_at_n"]["10"] == 1.0


def test_first_stage_recall_never_falls_as_n_grows(results: dict) -> None:
    for scope in ("training_jobs", "held_out", "all_jobs"):
        for measured in results["first_stage"][scope].values():
            recalls = [measured["recall_at_n"][str(n)] for n in script.RECALL_CUTOFFS]
            assert recalls == sorted(recalls)
            assert all(0.0 < r <= 1.0 for r in recalls)


def test_the_first_stage_is_chosen_on_the_training_jobs(results: dict) -> None:
    first = results["first_stage"]
    assert first["chosen_on"] == "training jobs"
    assert first["chosen"] == script.choose_first_stage(first["training_jobs"])
    chosen = first["chosen"]
    measured = first["training_jobs"][chosen["scorer"]]["recall_at_n"][str(chosen["n"])]
    assert chosen["recall_on_training_jobs"] == measured
    assert (first["cutoffs"], first["target_recall"]) == (list(script.RECALL_CUTOFFS), 0.9)


def _recalls(**by_name: list[float]) -> dict:
    return {
        name: {"recall_at_n": dict(zip(map(str, script.RECALL_CUTOFFS), values, strict=True))}
        for name, values in by_name.items()
    }


def test_the_first_stage_is_the_smallest_n_reaching_the_target() -> None:
    chosen = script.choose_first_stage(
        _recalls(
            bm25=[0.5, 0.7, 0.85, 0.92, 0.97, 0.99],
            dense=[0.6, 0.8, 0.91, 0.95, 0.98, 0.99],
            hybrid_rrf=[0.6, 0.8, 0.90, 0.96, 0.98, 1.0],
        )
    )
    assert chosen == {
        "scorer": "dense",
        "n": 15,
        "target_met": True,
        "recall_on_training_jobs": 0.91,
    }


def test_equal_recall_goes_to_the_cheaper_first_stage() -> None:
    values = [0.5, 0.9, 0.95, 1.0, 1.0, 1.0]
    chosen = script.choose_first_stage(_recalls(hybrid_rrf=values, dense=values, bm25=values))
    assert (chosen["scorer"], chosen["n"]) == ("bm25", 10)
    chosen = script.choose_first_stage(_recalls(hybrid_rrf=values, dense=values))
    assert (chosen["scorer"], chosen["n"]) == ("dense", 10)


def test_a_target_nobody_reaches_takes_the_largest_n_and_says_so() -> None:
    chosen = script.choose_first_stage(
        _recalls(bm25=[0.1, 0.2, 0.3, 0.4, 0.5, 0.89], dense=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    )
    assert chosen == {
        "scorer": "bm25",
        "n": 30,
        "target_met": False,
        "recall_on_training_jobs": 0.89,
    }


def test_choosing_a_first_stage_from_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no retrieval scorer"):
        script.choose_first_stage({})


def test_package_versions_are_read_from_the_installed_distribution() -> None:
    assert script._package_version("numpy") == np.__version__
    assert script._package_version("no-such-distribution-anywhere") is None


def test_it_records_how_retrieval_was_configured(results: dict) -> None:
    assert results["retrieval"] == {
        "bm25": {"k1": 1.2, "b": 0.75, "pool": 1200},
        "encoder": "fake-encoder@1",
        "rrf_k": 60,
        "fused_within": "each job's slate",
    }


def test_no_dense_skips_the_encoder_and_what_needs_it(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise AssertionError("the encoder was loaded")

    monkeypatch.setattr(script, "load_encoder", refuse)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST, "--no-dense"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(written["scorers"]) >= {"matcher", "bm25"}
    assert not {"dense", "hybrid_rrf"} & set(written["scorers"])
    # BM25 is the only first stage left, and the re-rank re-orders its top N.
    assert set(written["all_jobs"]["scorers"]) == {"bm25", "rerank"}
    assert set(written["comparisons"]) == {"rerank_minus_bm25"}
    assert set(written["all_jobs"]["comparisons"]) == {"rerank_minus_bm25"}
    assert written["agreement"] == {} == written["all_jobs"]["agreement"]
    assert written["retrieval"]["encoder"] is None
    first = written["first_stage"]
    assert set(first["training_jobs"]) == set(first["held_out"]) == {"bm25"}
    assert first["chosen"]["scorer"] == "bm25"
    assert written["rerank"]["first_stage"] == "bm25"
    assert written["rerank"]["n"] == first["chosen"]["n"]
    # The cross-encoder still ran on them.
    assert written["sentence_transformers"] == script._package_version("sentence-transformers")
    assert written["torch"] == script._package_version("torch")


def test_an_encoder_it_cannot_load_is_refused_naming_no_dense(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable() -> None:
        raise EncoderUnavailable("sentence-transformers is not installed")

    def too_early(*args: object, **kwargs: object) -> None:
        raise AssertionError("work started before the encoder was loaded")

    monkeypatch.setattr(script, "load_encoder", unavailable)
    monkeypatch.setattr(script, "generate_corpus", too_early)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 2
    err = capsys.readouterr().err
    assert "sentence-transformers is not installed" in err and "--no-dense" in err
    assert not out.exists()


def test_it_records_the_encoder_packages_it_ran_with(results: dict) -> None:
    assert results["sentence_transformers"] == script._package_version("sentence-transformers")
    assert results["torch"] == script._package_version("torch")


def test_no_dense_and_no_rerank_record_no_model_packages(model: Path, tmp_path: Path) -> None:
    out = tmp_path / "evaluation.json"
    argv = ["--model", str(model), "--out", str(out), *FAST, "--no-dense", "--no-rerank"]
    assert script.main(argv) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["sentence_transformers"] is None and written["torch"] is None


# --- the re-rank -------------------------------------------------------


def _two_stage(first: np.ndarray, n: int, job: str, documents: list[str]) -> list[float]:
    """One slate's two-stage ranks, computed without the script's helpers.

    Ranks, as the script gives them, since ROC-AUC pools pairs across slates:
    the rest by the first stage from 1, the kept ones by the stand-in's score
    from the slate's size plus 2.
    """
    threshold = sorted(first, reverse=True)[min(n, len(first)) - 1]
    kept = first >= threshold
    out = np.empty(len(first))
    out[~kept] = rankdata(first[~kept])
    second = [fake_relevance(job, d) for d, keep in zip(documents, kept, strict=True) if keep]
    out[kept] = len(first) + 1 + rankdata(second)
    return list(out)


def _reranked(data: TrainingData, first: np.ndarray, n: int) -> np.ndarray:
    refs = np.array(data.job_refs)
    out = np.empty(len(data))
    for ref in sorted(set(data.job_refs)):
        slate = np.flatnonzero(refs == ref)
        job = job_document(data.pairs[slate[0]][0])
        documents = [candidate_document(data.pairs[i][1]) for i in slate]
        out[slate] = _two_stage(first[slate], n, job, documents)
    return out


@pytest.fixture(scope="module")
def recomputed_rerank(results: dict, recomputed: dict) -> dict[str, Any]:
    """The re-rank's evaluations, recomputed from the chosen first stage."""
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    data = training_data(corpus)
    held_out = split_by_job(data, seed=SPLIT_SEED)[1]
    chosen = results["first_stage"]["chosen"]
    out = {}
    for scope, subset in (("all", data), ("held_out", held_out)):
        first = recomputed[scope]["scores"][chosen["scorer"]]
        out[scope] = evaluate_scores(subset, _reranked(subset, first, chosen["n"]))
    return out


@pytest.mark.parametrize("scope", ["held_out", "all"])
def test_rerank_orders_the_first_stages_top_n_by_the_cross_encoder(
    results: dict, recomputed_rerank: dict, scope: str
) -> None:
    measured = (results if scope == "held_out" else results["all_jobs"])["scorers"]["rerank"]
    expected = recomputed_rerank[scope]
    assert measured["ndcg_at_10"] == pytest.approx(expected.ndcg_at_10, abs=1e-12)
    assert measured["precision_at_5"] == pytest.approx(expected.precision_at_5, abs=1e-12)
    assert measured["roc_auc_relevant"] == pytest.approx(expected.roc_auc_relevant, abs=1e-12)
    assert measured["roc_auc_strong"] == pytest.approx(expected.roc_auc_strong, abs=1e-12)


def test_rerank_changes_the_first_stages_order(results: dict) -> None:
    chosen = results["first_stage"]["chosen"]["scorer"]
    assert results["scorers"]["rerank"]["ndcg_at_10"] != results["scorers"][chosen]["ndcg_at_10"]


@pytest.mark.parametrize("scope", ["held_out", "all"])
@pytest.mark.parametrize("other", ["bm25", "dense", "hybrid_rrf"])
def test_rerank_is_compared_with_each_retrieval_scorer_job_by_job(
    results: dict, recomputed: dict, recomputed_rerank: dict, scope: str, other: str
) -> None:
    where = results if scope == "held_out" else results["all_jobs"]
    paired = where["comparisons"][f"rerank_minus_{other}"]
    pairs = list(zip(recomputed_rerank[scope].slates, recomputed[scope][other].slates, strict=True))
    ndcg = [a.ndcg - b.ndcg for a, b in pairs if a.ndcg is not None]
    precision = [a.precision - b.precision for a, b in pairs]
    assert paired["ndcg_at_10_difference"] == pytest.approx(np.mean(ndcg), abs=1e-12)
    assert paired["precision_at_5_difference"] == pytest.approx(np.mean(precision), abs=1e-12)
    assert paired["ndcg_at_10_jobs_better"] == sum(d > 0 for d in ndcg)
    assert paired["ndcg_at_10_jobs_worse"] == sum(d < 0 for d in ndcg)


def test_the_cross_encoder_scores_only_what_the_first_stage_keeps(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recomputed: dict
) -> None:
    made: list[FakeCrossEncoder] = []

    def load() -> FakeCrossEncoder:
        made.append(FakeCrossEncoder())
        return made[-1]

    monkeypatch.setattr(script, "load_cross_encoder", load)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    chosen = written["first_stage"]["chosen"]
    [cross_encoder] = made

    expected_calls, held_out_kept = [], 0
    data = training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))
    held_out_jobs = set(split_by_job(data, seed=SPLIT_SEED)[1].job_refs)
    first, refs = recomputed["all"]["scores"][chosen["scorer"]], recomputed["all"]["job_refs"]
    pairs = data.pairs
    for ref in sorted(set(refs)):
        slate = np.flatnonzero(refs == ref)
        threshold = sorted(first[slate], reverse=True)[chosen["n"] - 1]
        kept = [i for i in slate if first[i] >= threshold]
        expected_calls.append(
            (job_document(pairs[slate[0]][0]), [candidate_document(pairs[i][1]) for i in kept])
        )
        held_out_kept += len(kept) if ref in held_out_jobs else 0

    # One call per slate, holding exactly the kept candidates, in slate order.
    assert cross_encoder.calls == expected_calls
    assert written["rerank"]["pairs_scored"] == sum(len(docs) for _, docs in expected_calls)
    assert written["rerank"]["pairs_scored_held_out"] == held_out_kept
    assert written["rerank"]["pairs_scored"] < written["all_jobs"]["pairs"]


def test_the_cut_follows_the_scores_not_the_slates_order() -> None:
    # A slate lists its own family's candidates first, and on this corpus they
    # are often exactly BM25's top 20, so reverse one to tell the two apart.
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    data = training_data(corpus)
    first_job = sorted(set(data.job_refs))[0]
    rows = [i for i, ref in enumerate(data.job_refs) if ref == first_job][::-1]
    slate = TrainingData(
        pairs=tuple(data.pairs[i] for i in rows),
        advanced=data.advanced[rows],
        relevance=data.relevance[rows],
        job_refs=tuple(data.job_refs[i] for i in rows),
    )
    first = _bm25(corpus, slate)
    cross_encoder = FakeCrossEncoder()

    scores, rescored = script._rerank_scores(corpus, slate, first, 5, cross_encoder)

    kept = first >= np.sort(first)[::-1][4]
    assert not kept[:5].all()
    assert (rescored == kept).all()
    job = job_document(slate.pairs[0][0])
    documents = [candidate_document(slate.pairs[i][1]) for i in np.flatnonzero(kept)]
    assert cross_encoder.calls == [(job, documents)]
    expected = _two_stage(first, 5, job, [candidate_document(c) for _, c in slate.pairs])
    assert scores.tolist() == expected


def test_the_rerank_uses_whichever_first_stage_and_n_were_chosen(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recomputed: dict
) -> None:
    def dense_at_7(training: dict) -> dict:
        return {"scorer": "dense", "n": 7, "target_met": False, "recall_on_training_jobs": 0.5}

    monkeypatch.setattr(script, "choose_first_stage", dense_at_7)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (written["rerank"]["first_stage"], written["rerank"]["n"]) == ("dense", 7)
    data = training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))
    expected = evaluate_scores(data, _reranked(data, recomputed["all"]["scores"]["dense"], 7))
    measured = written["all_jobs"]["scorers"]["rerank"]
    assert measured["ndcg_at_10"] == pytest.approx(expected.ndcg_at_10, abs=1e-12)
    assert measured["roc_auc_relevant"] == pytest.approx(expected.roc_auc_relevant, abs=1e-12)


def test_it_records_how_the_rerank_was_configured(results: dict) -> None:
    chosen = results["first_stage"]["chosen"]
    rerank = results["rerank"]
    assert {k: rerank[k] for k in ("cross_encoder", "first_stage", "n", "within")} == {
        "cross_encoder": "fake-cross-encoder@1",
        "first_stage": chosen["scorer"],
        "n": chosen["n"],
        "within": "each job's slate",
    }


def test_no_rerank_skips_the_cross_encoder_and_what_needs_it(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise AssertionError("the cross-encoder was loaded")

    monkeypatch.setattr(script, "load_cross_encoder", refuse)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST, "--no-rerank"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert "rerank" not in written["scorers"] and "rerank" not in written["all_jobs"]["scorers"]
    assert written["rerank"] is None
    assert set(written["comparisons"]) == {"hybrid_rrf_minus_bm25", "hybrid_rrf_minus_dense"}
    assert set(written["all_jobs"]["comparisons"]) == set(written["comparisons"])
    assert {"dense", "hybrid_rrf"} <= set(written["scorers"])


def test_a_cross_encoder_it_cannot_load_is_refused_before_any_work(
    model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable() -> None:
        raise EncoderUnavailable("cannot load the cross-encoder")

    def too_early(*args: object, **kwargs: object) -> None:
        raise AssertionError("work started before the cross-encoder was loaded")

    monkeypatch.setattr(script, "load_cross_encoder", unavailable)
    monkeypatch.setattr(script, "generate_corpus", too_early)
    out = tmp_path / "evaluation.json"
    assert script.main(["--model", str(model), "--out", str(out), *FAST]) == 2
    err = capsys.readouterr().err
    assert "cannot load the cross-encoder" in err and "--no-rerank" in err
    assert not out.exists()


# --- the real model ----------------------------------------------------


@pytest.fixture(scope="module")
def real_encoder() -> Encoder:
    try:
        return REAL_LOAD_ENCODER()
    except EncoderUnavailable as exc:  # pragma: no cover - depends on the machine
        if os.environ.get("RECRUITFLOW_REQUIRE_ENCODER"):
            raise
        pytest.skip(str(exc))


def test_real_model_is_the_pinned_one(real_encoder: Encoder) -> None:
    assert real_encoder.name == f"{DEFAULT_MODEL}@{DEFAULT_REVISION}+chunks254"


def test_real_model_dense_scores_are_each_pairs_cosine(real_encoder: Encoder) -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    data = training_data(corpus)
    # Two whole slates, so the batch holds two jobs and about a hundred candidates.
    first_two = set(sorted(set(data.job_refs))[:2])
    keep = np.array([ref in first_two for ref in data.job_refs])
    pairs = [pair for pair, kept in zip(data.pairs, keep, strict=True) if kept]
    subset = TrainingData(
        pairs=tuple(pairs),
        advanced=data.advanced[keep],
        relevance=data.relevance[keep],
        job_refs=tuple(ref for ref, kept in zip(data.job_refs, keep, strict=True) if kept),
    )

    scores = script._dense_scores(real_encoder, subset)

    assert scores.shape == (len(subset),) and np.isfinite(scores).all()
    for (job, candidate), score in list(zip(subset.pairs, scores, strict=True))[::7]:
        j, c = real_encoder.encode([job_document(job), candidate_document(candidate)])
        # Batches move a value by up to 1.6e-7 (see app.ml.encoder).
        assert score == pytest.approx(float(j @ c), abs=1e-5)


@pytest.fixture(scope="module")
def real_cross_encoder() -> Any:
    try:
        return REAL_LOAD_CROSS_ENCODER()
    except EncoderUnavailable as exc:  # pragma: no cover - depends on the machine
        if os.environ.get("RECRUITFLOW_REQUIRE_ENCODER"):
            raise
        pytest.skip(str(exc))


def test_real_model_cross_encoder_is_the_pinned_one(real_cross_encoder: Any) -> None:
    assert real_cross_encoder.name == (
        f"{DEFAULT_CROSS_ENCODER}@{DEFAULT_CROSS_ENCODER_REVISION}+maxp128"
    )


def test_real_model_rerank_orders_the_kept_by_the_cross_encoder(real_cross_encoder: Any) -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))
    data = training_data(corpus)
    first_job = sorted(set(data.job_refs))[0]
    keep = np.array([ref == first_job for ref in data.job_refs])
    subset = TrainingData(
        pairs=tuple(pair for pair, kept in zip(data.pairs, keep, strict=True) if kept),
        advanced=data.advanced[keep],
        relevance=data.relevance[keep],
        job_refs=tuple(ref for ref, kept in zip(data.job_refs, keep, strict=True) if kept),
    )
    first = _bm25(corpus, subset)

    scores, rescored = script._rerank_scores(corpus, subset, first, 10, real_cross_encoder)

    kept = first >= np.sort(first)[::-1][9]
    assert (rescored == kept).all() and kept.sum() >= 10
    assert scores[kept].min() > scores[~kept].max()
    job = job_document(subset.pairs[0][0])
    direct = real_cross_encoder.score(
        job, [candidate_document(subset.pairs[i][1]) for i in np.flatnonzero(kept)]
    )
    assert (rankdata(scores[kept]) == rankdata(direct)).all()


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


# --- as a command ------------------------------------------------------


def test_it_runs_as_a_command(model: Path, tmp_path: Path) -> None:
    """Imported and invoked are different things; this is the documented way to run it."""
    out = tmp_path / "evaluation.json"
    command = [sys.executable, str(SCRIPT), "--model", str(model), "--out", str(out)]
    # A subprocess gets the real models, which CI's main job does not install.
    result = subprocess.run(
        [*command, "--split-seeds", "--resamples", "50", "--no-dense", "--no-rerank"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "NDCG@10" in result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["held_out"]["jobs"] == 10


def test_a_refusal_as_a_command_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere.joblib"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(missing), "--out", str(tmp_path / "e.json")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_an_out_path_it_cannot_write_is_refused_before_the_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "evaluation.json"
    out.mkdir()
    monkeypatch.setattr(script, "evaluate", lambda *a, **k: pytest.fail("the evaluation ran"))

    assert script.main(["--out", str(out), *FAST]) == 2
    assert f"cannot write {out}" in capsys.readouterr().err
