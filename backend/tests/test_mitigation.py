"""Unit tests for app.ml.mitigation, the bias mitigations the audit measures. No database needed."""

from __future__ import annotations

import dataclasses
import pickle
import warnings

import numpy as np
import pytest

from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.matrix import PairFeaturiser
from app.ml.mitigation import (
    LineDroppingFeaturiser,
    TermDroppingFeaturiser,
    make_mitigated_matcher,
    reweighing_weights,
)
from app.ml.train import make_matcher, training_data


@pytest.fixture(scope="module")
def data():
    return training_data(generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200)))


# --- reweighing --------------------------------------------------------


def test_reweighing_matches_a_hand_computation() -> None:
    # Group a: 3 pairs, 2 positive. Group b: 1 pair, 0 positive. Overall 2 of 4.
    weights = reweighing_weights(["a", "a", "a", "b"], [True, True, False, False])

    # a positive: (3/4 * 2/4) / (2/4) = 0.75; a negative: (3/4 * 2/4) / (1/4) = 1.5;
    # b negative: (1/4 * 2/4) / (1/4) = 0.5.
    assert weights.tolist() == pytest.approx([0.75, 0.75, 1.5, 0.5])


def test_after_reweighing_every_group_has_the_overall_positive_rate(data) -> None:
    groups = [candidate.gender for _, candidate in data.pairs]
    weights = reweighing_weights(groups, data.advanced)
    overall = data.advanced.mean()

    for group in set(groups):
        mask = np.array(groups) == group
        weighted_rate = (weights[mask] * data.advanced[mask]).sum() / weights[mask].sum()
        assert weighted_rate == pytest.approx(overall)
    assert weights.sum() == pytest.approx(len(data))


def test_a_group_and_label_already_independent_are_weighted_one() -> None:
    weights = reweighing_weights(["a", "a", "b", "b"], [True, False, True, False])

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_reweighing_refuses_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="one length"):
        reweighing_weights(["a", "b"], [True])
    with pytest.raises(ValueError, match="no pairs"):
        reweighing_weights([], [])


# --- dropping terms ----------------------------------------------------


def test_dropped_terms_are_not_columns_and_nothing_else_changes(data) -> None:
    plain = PairFeaturiser().fit(data.pairs)
    dropped = TermDroppingFeaturiser(("Women", "bootcamp")).fit(data.pairs)
    plain_names = set(plain.get_feature_names_out())
    dropped_names = set(dropped.get_feature_names_out())

    assert {"resume:women", "resume:bootcamp"} <= plain_names
    assert plain_names - dropped_names == {"resume:women", "resume:bootcamp"}
    assert dropped_names <= plain_names


def test_dropping_no_terms_builds_the_plain_matrix(data) -> None:
    pairs = data.pairs[:300]
    plain = PairFeaturiser().fit(pairs).transform(pairs)
    none_dropped = TermDroppingFeaturiser().fit(pairs).transform(pairs)

    assert (plain != none_dropped).nnz == 0


# --- dropping lines ----------------------------------------------------


def test_a_dropped_line_reaches_no_column_and_no_similarity(data) -> None:
    job, candidate = data.pairs[0]
    body = "Data Analyst with 3 years of experience.\n- Build dashboards using SQL."
    with_line = dataclasses.replace(
        candidate, ref="X1", resume_text=body + "\nCommunity: Captain of a chess club."
    )
    without_line = dataclasses.replace(candidate, ref="X2", resume_text=body)
    pairs = [*data.pairs[:500], (job, with_line), (job, without_line)]
    plain = PairFeaturiser().fit(pairs)

    featuriser = LineDroppingFeaturiser(("Community:",)).fit(pairs)
    with_row, without_row = featuriser.transform(pairs[-2:]).toarray()

    assert {"resume:chess", "resume:captain"} <= set(plain.get_feature_names_out())
    assert not {"resume:chess", "resume:captain"} & set(featuriser.get_feature_names_out())
    assert with_row.tolist() == without_row.tolist()


def test_headings_match_after_folding_and_leading_spaces(data) -> None:
    dropper = LineDroppingFeaturiser(("Community:",)).fit(data.pairs[:100])

    kept = dropper.vectoriser_.preprocessor("Keep\n  COMMUNITY: drop me\nalso keep")
    assert kept == "keep\nalso keep"


def test_a_heading_in_the_middle_of_a_line_does_not_drop_it(data) -> None:
    dropper = LineDroppingFeaturiser(("Community:",)).fit(data.pairs[:100])

    assert dropper.vectoriser_.preprocessor("Led the Community: outreach") == (
        "led the community: outreach"
    )


def test_dropping_no_lines_builds_the_plain_matrix(data) -> None:
    pairs = data.pairs[:300]
    plain = PairFeaturiser().fit(pairs).transform(pairs)
    none_dropped = LineDroppingFeaturiser().fit(pairs).transform(pairs)

    assert (plain != none_dropped).nnz == 0


def test_a_fitted_line_dropper_survives_pickling(data) -> None:
    pairs = data.pairs[:200]
    featuriser = LineDroppingFeaturiser(("Community:",)).fit(pairs)
    restored = pickle.loads(pickle.dumps(featuriser))

    assert (restored.transform(pairs) != featuriser.transform(pairs)).nnz == 0


# --- the mitigated matcher ---------------------------------------------


def test_unweighted_with_the_plain_featuriser_it_is_the_shipped_matcher(data) -> None:
    train, test = data.pairs[:1500], data.pairs[1500:]
    labels = data.advanced[:1500]

    shipped = make_matcher().fit(train, labels).predict_proba(test)[:, 1]
    mitigated = make_mitigated_matcher().fit(train, labels).predict_proba(test)[:, 1]

    assert mitigated.tolist() == shipped.tolist()


def test_weights_reach_the_classifier_without_a_warning(data) -> None:
    train, test = data.pairs[:1500], data.pairs[1500:]
    labels = data.advanced[:1500]
    weights = reweighing_weights([c.gender for _, c in train], labels)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weighted = make_mitigated_matcher().fit(train, labels, sample_weight=weights)
    model = weighted.calibrated_classifiers_[0].estimator
    unweighted = make_mitigated_matcher().fit(train, labels)

    # The fitted classifier differs from an unweighted one, so it saw the weights.
    coefficients = model.named_steps["classifier"].coef_
    plain = unweighted.calibrated_classifiers_[0].estimator.named_steps["classifier"].coef_
    assert not np.allclose(coefficients, plain)
    assert weighted.predict_proba(test).shape == (len(test), 2)


def test_uniform_weights_change_nothing(data) -> None:
    train, test = data.pairs[:1500], data.pairs[1500:]
    labels = data.advanced[:1500]

    plain = make_mitigated_matcher().fit(train, labels).predict_proba(test)[:, 1]
    ones = make_mitigated_matcher().fit(train, labels, sample_weight=np.ones(1500))
    assert ones.predict_proba(test)[:, 1] == pytest.approx(plain, abs=1e-6)
