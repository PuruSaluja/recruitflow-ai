"""Unit tests for app.ml.evaluate, the held-out ranking metrics. No database needed."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from app.ml.evaluate import (
    bootstrap_interval,
    evaluate_scores,
    gain,
    ndcg_at_k,
    per_slate,
    precision_at_k,
    safe_auc,
)
from app.ml.train import TrainingData


def _log2(position: int) -> float:
    return math.log2(position + 1)


# --- NDCG@k ------------------------------------------------------------


def test_gain_is_two_to_the_grade_minus_one() -> None:
    assert gain(np.array([0, 1, 2])).tolist() == [0.0, 1.0, 3.0]


def test_a_slate_in_ideal_order_scores_one() -> None:
    assert ndcg_at_k([0.9, 0.8, 0.7, 0.1], [2, 1, 1, 0], k=10) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computation() -> None:
    # Ranked order of relevance: 0, 2, 1. Gains 0, 3, 1.
    dcg = 0 / _log2(1) + 3 / _log2(2) + 1 / _log2(3)
    ideal = 3 / _log2(1) + 1 / _log2(2) + 0 / _log2(3)

    assert ndcg_at_k([0.9, 0.5, 0.1], [0, 2, 1], k=10) == pytest.approx(dcg / ideal)


def test_the_cut_off_drops_everything_below_it_from_both_dcgs() -> None:
    # Top 2 holds relevance 0 and 1; the ideal top 2 holds 2 and 1.
    dcg = 0 / _log2(1) + 1 / _log2(2)
    ideal = 3 / _log2(1) + 1 / _log2(2)

    assert ndcg_at_k([0.9, 0.8, 0.1], [0, 1, 2], k=2) == pytest.approx(dcg / ideal)


def test_a_slate_with_nobody_relevant_has_no_ndcg() -> None:
    assert ndcg_at_k([0.3, 0.2, 0.1], [0, 0, 0]) is None


def test_ndcg_does_not_depend_on_the_order_the_slate_arrives_in() -> None:
    scores, relevance = [0.1, 0.9, 0.4, 0.6, 0.3], [2, 0, 1, 0, 1]
    expected = ndcg_at_k(scores, relevance, k=3)
    for order in itertools.permutations(range(5)):
        assert ndcg_at_k([scores[i] for i in order], [relevance[i] for i in order], k=3) == (
            pytest.approx(expected)
        )


@pytest.mark.parametrize("k", [1, 3, 10, 50])
def test_ndcg_matches_scikit_learn_given_the_same_gains(k: int) -> None:
    # scikit-learn takes y_true as the gain itself, and averages over ties as
    # this module does; the coarse rounding makes many ties on purpose.
    rng = np.random.default_rng(7)
    for _ in range(25):
        relevance = rng.integers(0, 3, size=40)
        if not relevance.any():
            continue
        scores = np.round(rng.random(40), 1)
        expected = ndcg_score([gain(relevance)], [scores], k=k)
        assert ndcg_at_k(scores, relevance, k=k) == pytest.approx(expected)


def test_ties_take_the_expected_value_over_their_orders() -> None:
    # All three tie, so each of the 6 orders is equally likely.
    scores, relevance = [0.5, 0.5, 0.5], [2, 0, 1]
    by_order = []
    for order in itertools.permutations(range(3)):
        broken = [3.0 - position for position in range(3)]
        reordered = [0.0] * 3
        for position, index in enumerate(order):
            reordered[index] = broken[position]
        by_order.append(ndcg_at_k(reordered, relevance, k=2))

    assert ndcg_at_k(scores, relevance, k=2) == pytest.approx(np.mean(by_order))


# --- P@k ---------------------------------------------------------------


def test_precision_counts_partial_and_strong_fits_in_the_top_k() -> None:
    assert precision_at_k([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [2, 0, 1, 0, 0, 2], k=5) == 0.4


def test_precision_divides_by_k_even_on_a_shorter_slate() -> None:
    assert precision_at_k([0.9, 0.8], [1, 2], k=5) == 0.4


def test_precision_averages_over_ties_across_the_cut_off() -> None:
    # One certain hit, then four tied for the last spot of a top 2: half are hits.
    assert precision_at_k([0.9, 0.5, 0.5, 0.5, 0.5], [1, 1, 0, 1, 0], k=2) == 0.75


def test_all_scores_equal_gives_the_exact_random_ranking_expectation() -> None:
    relevance = [2, 0, 0, 1, 0, 0, 1, 0]
    scores = [0.0] * len(relevance)

    # P@k of a random order is the slate's relevant share; NDCG is the mean over all orders.
    assert precision_at_k(scores, relevance, k=5) == pytest.approx(3 / 8)
    every_order = [
        ndcg_at_k([-float(p) for p in order], relevance, k=3)
        for order in itertools.permutations(range(len(relevance)))
    ]
    assert ndcg_at_k(scores, relevance, k=3) == pytest.approx(np.mean(every_order))


# --- input checks ------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "relevance", "k", "message"),
    [
        ([0.1, 0.2], [1], 5, "one length"),
        ([], [], 5, "at least one candidate"),
        ([0.1], [1], 0, "k must be at least 1"),
        ([0.1, float("nan")], [1, 0], 5, "finite"),
    ],
)
def test_bad_input_is_refused_with_a_reason(
    scores: list[float], relevance: list[int], k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        precision_at_k(scores, relevance, k=k)
    with pytest.raises(ValueError, match=message):
        ndcg_at_k(scores, relevance, k=k)


# --- over a split ------------------------------------------------------


def _data(job_refs: list[str], relevance: list[int], advanced: list[bool]) -> TrainingData:
    return TrainingData(
        pairs=tuple((None, None) for _ in job_refs),  # type: ignore[misc]
        advanced=np.array(advanced, dtype=bool),
        relevance=np.array(relevance, dtype=int),
        job_refs=tuple(job_refs),
    )


def test_each_job_is_ranked_on_its_own_slate() -> None:
    # Job b's scores are all higher than job a's; pooled, b would fill every top spot.
    data = _data(["a", "b", "a", "b"], [1, 0, 0, 1], [True, False, False, True])
    slates = per_slate(data, [0.2, 0.9, 0.1, 0.8])

    assert [slate.job_ref for slate in slates] == ["a", "b"]
    assert slates[0].ndcg == pytest.approx(1.0)
    assert slates[1].ndcg == pytest.approx(1 / _log2(2))
    assert [slate.size for slate in slates] == [2, 2]


def test_a_job_without_relevant_candidates_is_counted_not_averaged() -> None:
    data = _data(["a", "a", "b", "b"], [2, 0, 0, 0], [True, False, False, True])
    evaluation = evaluate_scores(data, [0.9, 0.1, 0.5, 0.4])

    assert evaluation.ndcg_at_10 == pytest.approx(1.0)
    assert evaluation.jobs_without_relevant == 1
    # P@5 still averages over both jobs: 1 of 5 for a, 0 for b.
    assert evaluation.precision_at_5 == pytest.approx(0.1)


def test_roc_auc_is_pooled_over_every_pair_against_each_label() -> None:
    data = _data(["a", "a", "b", "b"], [2, 0, 1, 0], [True, False, False, True])
    # The partial fit outscores the strong one, so the two relevance labels differ.
    scores = [0.5, 0.1, 0.9, 0.4]
    evaluation = evaluate_scores(data, scores)

    assert evaluation.roc_auc_advanced == pytest.approx(0.5)
    assert evaluation.roc_auc_relevant == pytest.approx(1.0)
    assert evaluation.roc_auc_strong == pytest.approx(2 / 3)


def test_a_label_with_one_class_has_no_roc_auc() -> None:
    assert safe_auc(np.array([False, False]), np.array([0.2, 0.3])) is None


def test_every_job_without_a_relevant_candidate_is_refused() -> None:
    data = _data(["a", "a"], [0, 0], [True, False])
    with pytest.raises(ValueError, match="NDCG is undefined"):
        evaluate_scores(data, [0.2, 0.1])


def test_scores_must_cover_every_pair() -> None:
    with pytest.raises(ValueError, match="3 scores for 2 pairs"):
        per_slate(_data(["a", "a"], [1, 0], [True, False]), [0.1, 0.2, 0.3])


# --- bootstrap ---------------------------------------------------------


def test_the_bootstrap_interval_brackets_the_mean_and_repeats_for_a_seed() -> None:
    values = np.random.default_rng(3).random(50)
    low, high = bootstrap_interval(values, seed=1)

    assert low < values.mean() < high
    assert bootstrap_interval(values, seed=1) == (low, high)


def test_a_constant_sample_has_a_zero_width_interval() -> None:
    assert bootstrap_interval([0.5] * 10) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(("values", "level"), [([], 0.95), ([0.5], 1.0), ([0.5], 0.0)])
def test_the_bootstrap_refuses_what_it_cannot_compute(values: list[float], level: float) -> None:
    with pytest.raises(ValueError):
        bootstrap_interval(values, level=level)
