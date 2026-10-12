"""Unit tests for app.ml.fairness, the shortlist fairness metrics. No database needed."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.fairness import (
    FOUR_FIFTHS,
    audit_selection,
    group_rates,
    top_k_selection,
)

# --- top-k selection ---------------------------------------------------


def test_the_top_k_of_a_slate_are_selected_and_the_rest_are_not() -> None:
    selected = top_k_selection(["J1"] * 4, [0.2, 0.9, 0.1, 0.5], k=2)

    assert selected.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_each_slate_gets_its_own_top_k() -> None:
    # J2's scores are all below J1's, but it still shortlists its own best two.
    refs = ["J1", "J2", "J1", "J2", "J1", "J2"]
    scores = [0.9, 0.3, 0.8, 0.2, 0.7, 0.1]

    assert top_k_selection(refs, scores, k=2).tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_a_tie_straddling_the_cut_shares_the_places_left() -> None:
    # One clear first, then three tied for the one place left.
    selected = top_k_selection(["J1"] * 5, [0.9, 0.5, 0.5, 0.5, 0.1], k=2)

    assert selected.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    assert selected.sum() == pytest.approx(2.0)


def test_a_tie_wholly_above_the_cut_is_selected_whole() -> None:
    selected = top_k_selection(["J1"] * 4, [0.5, 0.5, 0.4, 0.1], k=3)

    assert selected.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_scores_that_all_tie_select_everyone_with_probability_k_over_n() -> None:
    selected = top_k_selection(["J1"] * 8, [0.3] * 8, k=2)

    assert selected.tolist() == pytest.approx([0.25] * 8)


def test_a_slate_no_longer_than_k_is_shortlisted_whole() -> None:
    assert top_k_selection(["J1"] * 3, [0.1, 0.2, 0.3], k=5).tolist() == [1.0, 1.0, 1.0]


def test_selection_does_not_depend_on_the_order_pairs_arrive_in() -> None:
    refs = ["J1", "J2"] * 5
    scores = np.linspace(0.0, 1.0, 10)
    order = np.random.default_rng(0).permutation(10)

    straight = top_k_selection(refs, scores, k=2)
    shuffled = top_k_selection(np.array(refs)[order], scores[order], k=2)

    assert shuffled.tolist() == straight[order].tolist()


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_at_least_one(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        top_k_selection(["J1"], [0.5], k=k)


@pytest.mark.parametrize("k", [2.5, 2.0, True, "2"])
def test_k_must_be_a_whole_number(k: object) -> None:
    # 2.5 used to shortlist two and a half places a slate, and True one.
    with pytest.raises(TypeError, match="k must be a whole number"):
        top_k_selection(["J1"] * 4, [0.4, 0.3, 0.2, 0.1], k=k)


def test_a_numpy_integer_k_is_accepted() -> None:
    selected = top_k_selection(["J1"] * 3, [0.3, 0.2, 0.1], k=np.int64(2))

    assert selected.tolist() == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_score_that_is_not_finite_is_refused_by_name(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN and infinity cannot be ranked"):
        top_k_selection(["J1", "J1"], [0.5, bad], k=1)


def test_refs_and_scores_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="one length"):
        top_k_selection(["J1", "J1"], [0.5], k=1)


# --- group rates -------------------------------------------------------


def test_group_rates_match_a_hand_count() -> None:
    selected = [1.0, 0.0, 1.0, 0.5, 0.0, 1.0]
    groups = ["a", "a", "a", "b", "b", "b"]
    relevance = [2, 1, 0, 1, 0, 0]

    rates = group_rates(selected, groups, relevance)

    assert list(rates) == ["a", "b"]
    assert rates["a"].pairs == 3
    assert rates["a"].selection_rate == pytest.approx(2 / 3)
    assert rates["a"].qualified == 2
    assert rates["a"].true_positive_rate == pytest.approx(1 / 2)
    assert rates["b"].selected == pytest.approx(1.5)
    assert rates["b"].selection_rate == pytest.approx(0.5)
    assert rates["b"].true_positive_rate == pytest.approx(0.5)


def test_a_group_with_nobody_qualified_has_no_true_positive_rate() -> None:
    rates = group_rates([1.0, 0.0], ["a", "b"], [0, 1])

    assert rates["a"].qualified == 0
    assert rates["a"].true_positive_rate is None


def test_selected_must_be_shares_between_zero_and_one() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        group_rates([1.5], ["a"], [1])


@pytest.mark.parametrize(
    ("selected", "groups", "relevance"),
    [
        ([1.0, 0.0], ["a"], [1, 0]),
        ([1.0, 0.0], ["a", "b"], [1]),
        ([[1.0], [0.0]], [["a"], ["b"]], [[1], [0]]),
    ],
    ids=["groups-short", "relevance-short", "nested"],
)
def test_sequences_of_different_lengths_or_nested_ones_are_refused(
    selected: list, groups: list, relevance: list
) -> None:
    with pytest.raises(ValueError, match="flat sequences of one length"):
        group_rates(selected, groups, relevance)
    with pytest.raises(ValueError, match="flat sequences of one length"):
        audit_selection(["J1"] * len(selected), selected, groups, relevance, reference="a")


def test_an_empty_audit_is_refused() -> None:
    with pytest.raises(ValueError, match="no pairs"):
        group_rates([], [], [])


# --- the audit ---------------------------------------------------------

# Two jobs. Group "m" is selected at 3/4, group "w" at 1/4; among the
# qualified, "m" at 2/2 and "w" at 1/2.
REFS = ["J1"] * 4 + ["J2"] * 4
SELECTED = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
GROUPS = ["m", "m", "w", "w", "m", "m", "w", "w"]
RELEVANCE = [2, 0, 1, 0, 1, 0, 0, 1]


def _audit(**overrides):
    arguments = dict(reference="m", resamples=200, seed=0)
    arguments.update(overrides)
    return audit_selection(REFS, SELECTED, GROUPS, RELEVANCE, **arguments)


def test_disparate_impact_is_the_selection_rate_over_the_reference_groups() -> None:
    audit = _audit()

    assert audit.groups["w"].disparate_impact.value == pytest.approx((1 / 4) / (3 / 4))
    assert audit.min_disparate_impact == pytest.approx(1 / 3)


def test_the_lowest_ratio_is_reported_across_every_other_group() -> None:
    # Reference "m" at 1.0, "a" at 0.5 and "b" at 0.25.
    audit = audit_selection(
        ["J1"] * 6,
        [1.0, 1.0, 0.5, 0.5, 0.25, 0.25],
        ["m", "m", "a", "a", "b", "b"],
        [0] * 6,
        reference="m",
        resamples=10,
    )

    assert audit.min_disparate_impact == pytest.approx(0.25)


def test_equal_opportunity_difference_is_the_true_positive_rate_minus_the_reference() -> None:
    assert _audit().groups["w"].equal_opportunity_difference.value == pytest.approx(0.5 - 1.0)


def test_the_reference_group_is_one_and_zero_against_itself() -> None:
    reference = _audit().groups["m"]

    assert reference.disparate_impact.value == pytest.approx(1.0)
    assert reference.equal_opportunity_difference.value == pytest.approx(0.0)
    assert reference.passes_four_fifths is None


@pytest.mark.parametrize(
    ("w_selected", "passes"),
    [(0.4, True), (0.39, False)],
)
def test_the_four_fifths_rule_passes_at_exactly_eight_tenths(
    w_selected: float, passes: bool
) -> None:
    # Reference rate 0.5, so dividing is exact: a ratio of 0.8 needs a rate of 0.4.
    audit = audit_selection(
        ["J1"] * 5,
        [0.5, w_selected, 0.0, 0.0, 0.0],
        ["m", "w", "x", "x", "x"],
        [0] * 5,
        reference="m",
        resamples=10,
    )
    expected = w_selected / 0.5

    assert audit.groups["w"].disparate_impact.value == pytest.approx(expected)
    assert audit.groups["w"].passes_four_fifths is passes
    assert FOUR_FIFTHS == 0.8


def test_a_reference_group_never_selected_leaves_every_ratio_undefined() -> None:
    audit = audit_selection(["J1"] * 2, [0.0, 1.0], ["m", "w"], [1, 1], reference="m", resamples=10)

    assert audit.groups["w"].disparate_impact.value is None
    assert audit.groups["w"].passes_four_fifths is None
    assert audit.min_disparate_impact is None


def test_an_unknown_reference_group_is_refused() -> None:
    with pytest.raises(ValueError, match="reference group 'z'"):
        _audit(reference="z")


def test_job_refs_must_match_the_pairs() -> None:
    with pytest.raises(ValueError, match="job refs"):
        audit_selection(["J1"], SELECTED, GROUPS, RELEVANCE, reference="m")


def test_resamples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="resamples"):
        _audit(resamples=0)


def test_the_interval_holds_the_point_estimate_and_is_reproducible() -> None:
    first, second = _audit(seed=3), _audit(seed=3)
    impact = first.groups["w"].disparate_impact

    assert impact.low <= impact.value <= impact.high
    assert impact == second.groups["w"].disparate_impact


def test_identical_jobs_give_an_interval_of_no_width() -> None:
    # Every job is the same, so any resample of jobs gives the same rates.
    refs = ["J1"] * 4 + ["J2"] * 4 + ["J3"] * 4
    selected = [1.0, 0.0, 1.0, 1.0] * 3
    groups = ["m", "m", "w", "w"] * 3
    relevance = [1, 1, 1, 0] * 3

    audit = audit_selection(refs, selected, groups, relevance, reference="m", resamples=50)
    impact = audit.groups["w"]

    assert impact.disparate_impact.low == pytest.approx(impact.disparate_impact.value)
    assert impact.disparate_impact.high == pytest.approx(impact.disparate_impact.value)
    # Qualified "w" pairs are selected 1 of 1 per job, qualified "m" pairs 1 of 2.
    assert impact.equal_opportunity_difference.value == pytest.approx(0.5)
    assert impact.equal_opportunity_difference.low == pytest.approx(0.5)
    assert impact.equal_opportunity_difference.high == pytest.approx(0.5)


def test_the_interval_resamples_jobs_not_pairs() -> None:
    # J1 favours "w" and J2 favours "m"; a resample that draws one job twice
    # moves the ratio, so the interval has width. Resampling pairs within the
    # pooled set would too, but resampling jobs can only produce three ratios.
    refs = ["J1"] * 2 + ["J2"] * 2
    audit = audit_selection(
        refs, [0.0, 1.0, 1.0, 0.0], ["m", "w", "m", "w"], [1] * 4, reference="m", resamples=400
    )
    impact = audit.groups["w"].disparate_impact

    assert impact.value == pytest.approx(1.0)
    # Draws: J1 twice (ratio undefined, dropped), J2 twice (0.0), one of each (1.0).
    assert impact.low == pytest.approx(0.0)
    assert impact.high == pytest.approx(1.0)
