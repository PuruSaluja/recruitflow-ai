"""Unit tests for training the matcher: the split, the pipeline, calibration. No database needed."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.matrix import PairFeaturiser
from app.ml.mitigation import LineDroppingFeaturiser, reweighing_weights
from app.ml.train import (
    CALIBRATION_METHOD,
    DEFAULT_TEST_SHARE,
    EXCLUDED_HEADINGS,
    REWEIGHED_ATTRIBUTE,
    TRAINING_LABEL,
    TrainingData,
    make_matcher,
    split_by_job,
    train_matcher,
    training_data,
    training_weights,
)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


@pytest.fixture(scope="module")
def data(corpus):
    return training_data(corpus)


@pytest.fixture(scope="module")
def split(data):
    return split_by_job(data)


@pytest.fixture(scope="module")
def train_set(split):
    return split[0]


@pytest.fixture(scope="module")
def test_set(split):
    return split[1]


@pytest.fixture(scope="module")
def fitted(train_set):
    return train_matcher(train_set)


@pytest.fixture(scope="module")
def scores(fitted, test_set):
    return fitted.predict_proba(test_set.pairs)[:, 1]


def fold_vectorisers(matcher):
    return [
        fold.estimator.named_steps["features"].vectoriser_
        for fold in matcher.calibrated_classifiers_
    ]


def held_out_jobs(data, **kwargs) -> set[str]:
    return set(split_by_job(data, **kwargs)[1].job_refs)


# --- resolving a corpus -----------------------------------------------------


def test_there_is_one_row_per_labelled_pair(corpus, data) -> None:
    assert len(data) == len(corpus.pairs)
    assert len(data.advanced) == len(data.relevance) == len(data.job_refs) == len(corpus.pairs)


def test_each_row_carries_its_own_objects_and_its_own_labels(corpus, data) -> None:
    jobs = {job.ref: job for job in corpus.jobs}
    people = {candidate.ref: candidate for candidate in corpus.candidates}

    for labelled, (job, candidate), advanced, relevance, job_ref in zip(
        corpus.pairs, data.pairs, data.advanced, data.relevance, data.job_refs, strict=True
    ):
        assert job is jobs[labelled.job_ref]
        assert candidate is people[labelled.candidate_ref]
        assert bool(advanced) == labelled.advanced
        assert int(relevance) == labelled.relevance
        assert job_ref == labelled.job_ref


def test_training_data_compares_and_hashes_by_identity(corpus, data) -> None:
    """Field-by-field equality compared the label arrays and raised ValueError."""
    twin = training_data(corpus)
    assert twin is not data
    assert (data == twin) is False
    assert data == data
    assert data != twin
    assert len({data, twin, data}) == 2


# --- the split --------------------------------------------------------------


def test_no_job_appears_on_both_sides(train_set, test_set) -> None:
    assert not set(train_set.job_refs) & set(test_set.job_refs)


def test_the_split_keeps_every_pair_exactly_once(data, train_set, test_set) -> None:
    assert len(train_set) + len(test_set) == len(data)
    assert set(train_set.pairs) | set(test_set.pairs) == set(data.pairs)


def test_a_held_out_job_takes_its_whole_slate_with_it(data, test_set) -> None:
    """Ranking metrics are per job over a whole slate, so half a slate is useless."""
    for ref in set(test_set.job_refs):
        assert test_set.job_refs.count(ref) == data.job_refs.count(ref)


def test_labels_travel_with_the_pair_they_belong_to(data, train_set, test_set) -> None:
    """Masking the rows but not the labels would keep every count right and every row wrong."""
    labels = {
        pair: (bool(advanced), int(relevance))
        for pair, advanced, relevance in zip(data.pairs, data.advanced, data.relevance, strict=True)
    }

    for side in (train_set, test_set):
        for pair, advanced, relevance in zip(
            side.pairs, side.advanced, side.relevance, strict=True
        ):
            assert labels[pair] == (bool(advanced), int(relevance))


def test_the_same_seed_holds_out_the_same_jobs(data) -> None:
    assert held_out_jobs(data, seed=3) == held_out_jobs(data, seed=3)


def test_another_seed_holds_out_different_jobs(data) -> None:
    assert held_out_jobs(data, seed=3) != held_out_jobs(data, seed=4)


def test_the_share_asked_for_is_the_share_of_jobs_held_out(data) -> None:
    n_jobs = len(set(data.job_refs))

    for share in (0.1, 0.25, 0.5):
        assert len(held_out_jobs(data, test_share=share)) == round(n_jobs * share)
    assert len(held_out_jobs(data)) == round(n_jobs * DEFAULT_TEST_SHARE)


def test_the_jobs_held_out_do_not_depend_on_the_order_the_pairs_arrive_in(data) -> None:
    order = np.random.default_rng(0).permutation(len(data))
    shuffled = TrainingData(
        pairs=tuple(data.pairs[i] for i in order),
        advanced=data.advanced[order],
        relevance=data.relevance[order],
        job_refs=tuple(data.job_refs[i] for i in order),
    )

    assert held_out_jobs(shuffled, seed=3) == held_out_jobs(data, seed=3)


@pytest.mark.parametrize("share", [0.0, 1.0, 0.01, 0.999])
def test_a_share_that_would_empty_one_side_is_refused(data, share) -> None:
    with pytest.raises(ValueError, match="at least one job on each side"):
        split_by_job(data, test_share=share)


def test_the_default_corpus_is_still_split_the_way_the_published_numbers_were() -> None:
    """Every held-out figure quoted anywhere came from this exact set of 50 jobs.

    Which jobs a seed picks is arbitrary, and every other test here passes just
    as well if it changes -- sorting the refs in reverse, or deriving the
    shuffle from `seed + 1`, both leave a working split and a different model.
    That would quietly strand the numbers in the README and the commit
    messages, so the choice is pinned by digest rather than left to whatever
    the code happens to do.

    `random.shuffle` is not guaranteed to draw alike across Python versions, so
    a future interpreter may well trip this. That is the point: it means the
    published numbers need re-measuring, not that the split is wrong.
    """
    held_out = sorted(held_out_jobs(training_data(generate_corpus(CorpusConfig()))))
    digest = hashlib.sha256(",".join(held_out).encode()).hexdigest()

    assert len(held_out) == 50
    assert digest[:16] == "0e7c8d6dc58ffb50"


# --- the matcher ------------------------------------------------------------


def test_the_matcher_calibrates_a_featuriser_and_a_classifier() -> None:
    matcher = make_matcher()

    assert isinstance(matcher, CalibratedClassifierCV)
    assert matcher.method == CALIBRATION_METHOD
    assert isinstance(matcher.estimator.named_steps["features"], PairFeaturiser)
    assert isinstance(matcher.estimator.named_steps["classifier"], LogisticRegression)


def test_nothing_about_the_classifier_is_tuned() -> None:
    """The module's claim is that the estimators keep scikit-learn's defaults.

    Nothing else here would notice a knob turned to chase a number. Calibration
    hides a reweighting from the average-score test, and neither `C` nor
    `class_weight` breaks any other assertion, though both move the ranking:
    `class_weight="balanced"` took held-out ROC-AUC from 0.7728 to 0.7657 and
    changed one score by as much as 0.10.
    """
    classifier = make_matcher().estimator.named_steps["classifier"]

    assert classifier.get_params() == LogisticRegression().get_params()


def test_training_fits_the_advanced_label_and_repeats_exactly(train_set, test_set, scores) -> None:
    """Two claims, one comparison: the target is ``advanced``, and a rerun reproduces it.

    Nothing in the pipeline draws at random, so a second fit on the same rows
    must give the same probabilities to the last bit -- which is what lets a
    persisted artifact be regenerated rather than kept.
    """
    again = make_matcher().fit(
        train_set.pairs, train_set.advanced, sample_weight=training_weights(train_set)
    )

    assert TRAINING_LABEL == "advanced"
    assert np.array_equal(again.predict_proba(test_set.pairs)[:, 1], scores)
    assert np.array_equal(train_matcher(train_set).predict_proba(test_set.pairs)[:, 1], scores)


def test_the_served_matcher_skips_community_lines_by_default() -> None:
    features = make_matcher().estimator.named_steps["features"]

    assert type(features) is LineDroppingFeaturiser
    assert features.dropped_headings == EXCLUDED_HEADINGS == ("Community:",)


def test_training_reweighs_the_pairs_against_gender(train_set) -> None:
    weights = training_weights(train_set)
    genders = [candidate.gender for _, candidate in train_set.pairs]

    assert REWEIGHED_ATTRIBUTE == "gender"
    assert weights.tolist() == reweighing_weights(genders, train_set.advanced).tolist()
    # The planted bias makes gender and the label dependent, so the weights vary.
    assert weights.min() < 1.0 < weights.max()


def test_the_weights_change_the_served_matcher(train_set, test_set, scores) -> None:
    unweighted = make_matcher().fit(train_set.pairs, train_set.advanced)

    assert not np.array_equal(unweighted.predict_proba(test_set.pairs)[:, 1], scores)


def test_a_community_line_never_moves_a_served_score(fitted, test_set) -> None:
    # Give every resume the planted women's line in place of its own: no score may move.
    swapped = []
    for job, candidate in test_set.pairs:
        lines = candidate.resume_text.split("\n")
        kept = [line for line in lines if not line.startswith("Community:")]
        other = "\n".join([*kept, "Community: Member of a women in engineering network."])
        swapped.append((job, dataclasses.replace(candidate, resume_text=other)))

    before = fitted.predict_proba(test_set.pairs)[:, 1]
    after = fitted.predict_proba(swapped)[:, 1]

    changed = [
        old.resume_text != new.resume_text
        for (_, old), (_, new) in zip(test_set.pairs, swapped, strict=True)
    ]
    # A few resumes already end with exactly that line.
    assert sum(changed) > 0.9 * len(changed)
    assert np.array_equal(before, after)


def test_the_relevance_grade_is_carried_but_never_fitted(train_set, test_set, scores) -> None:
    on_relevance = make_matcher().fit(train_set.pairs, train_set.relevance >= 1)

    assert not np.array_equal(on_relevance.predict_proba(test_set.pairs)[:, 1], scores)


def test_every_pair_gets_one_probability_in_range(test_set, scores) -> None:
    assert scores.shape == (len(test_set),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_no_two_candidates_come_out_with_the_same_score(scores) -> None:
    """Sigmoid is strictly monotone, so calibrating can neither reorder nor tie a slate.

    Isotonic is a step function and would: on these held-out pairs it returned
    264 to 310 fewer distinct probabilities than there are pairs, measured on
    corpus seeds 3, 5, 7 and 11.
    """
    assert len(np.unique(scores)) == len(scores)


def test_the_average_score_lands_near_the_rate_it_was_trained_on(train_set, scores) -> None:
    """Probabilities come out on the scale the matcher was trained on, not on a handy one.

    The gap measured on corpus seeds 3, 5, 7 and 11 is 0.002 to 0.011; the band
    here is wider than that spread on purpose, since it is four draws.

    This says nothing about the classifier underneath, and an earlier version of
    this docstring wrongly claimed it did. Pulling a score back onto the base
    rate is exactly calibration's job, so it absorbs a reweighted classifier
    too: `class_weight="balanced"` moves the *uncalibrated* mean from 0.140 to
    0.391, and calibrated it lands at 0.145 -- inside this band. The
    classifier's own settings are pinned by their own test.
    """
    assert abs(scores.mean() - train_set.advanced.mean()) < 0.03


# --- what the folds are allowed to see --------------------------------------


def test_each_calibration_fold_learns_its_own_vocabulary(fitted) -> None:
    """Calibration wraps the pipeline, so no fold calibrates against resumes it was fitted on.

    One featuriser fitted outside the folds would hand all five the same
    vocabulary. Measured on corpus seeds 3, 5, 7 and 11: three to five of the
    five folds differ.
    """
    vocabularies = {
        frozenset(vectoriser.get_feature_names_out()) for vectoriser in fold_vectorisers(fitted)
    }

    assert len(fitted.calibrated_classifiers_) == 5
    assert len(vocabularies) > 1


def test_the_vocabulary_is_learned_from_the_training_split_alone(
    train_set, test_set, fitted
) -> None:
    """A word only a held-out resume uses must not be a column of the trained model."""
    token = "zzqualifier"
    held_out = [
        (job, dataclasses.replace(candidate, resume_text=f"{candidate.resume_text}\n{token}"))
        for job, candidate in test_set.pairs[:2]
    ]

    for vectoriser in fold_vectorisers(fitted):
        assert token not in set(vectoriser.get_feature_names_out())
    # ...and it is a word this vectoriser does learn, when it is allowed to see it.
    both = PairFeaturiser().fit([*train_set.pairs, *held_out])
    assert token in set(both.vectoriser_.get_feature_names_out())
