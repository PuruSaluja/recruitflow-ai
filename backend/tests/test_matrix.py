"""Unit tests for the full feature matrix. No database needed."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from app.ml.corpus import CorpusConfig, SyntheticCandidate, SyntheticJob, generate_corpus
from app.ml.features import FEATURE_NAMES, feature_vector
from app.ml.matrix import PairFeaturiser
from app.ml.text import candidate_text, job_text, make_vectoriser
from app.models.candidate import Candidate
from app.models.job import Job

SIMILARITY = len(FEATURE_NAMES)


def pairs_of(corpus) -> list[tuple[SyntheticJob, SyntheticCandidate]]:
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}
    return [(jobs[pair.job_ref], people[pair.candidate_ref]) for pair in corpus.pairs]


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


@pytest.fixture(scope="module")
def pairs(corpus):
    return pairs_of(corpus)


@pytest.fixture(scope="module")
def fitted(pairs):
    return PairFeaturiser().fit(pairs)


@pytest.fixture(scope="module")
def matrix(fitted, pairs):
    return fitted.transform(pairs)


def rows_equal(left, right) -> bool:
    return left.shape == right.shape and (left != right).nnz == 0


# --- shape and names --------------------------------------------------------


def test_one_row_per_pair_and_one_name_per_column(fitted, matrix, pairs) -> None:
    names = fitted.get_feature_names_out()

    assert matrix.shape == (len(pairs), len(names))
    assert len(set(names)) == len(names)


def test_the_columns_are_engineered_then_similarity_then_resume_terms(fitted) -> None:
    names = list(fitted.get_feature_names_out())
    terms = fitted.vectoriser_.get_feature_names_out()

    assert tuple(names[:SIMILARITY]) == FEATURE_NAMES
    assert names[SIMILARITY] == "text_similarity"
    assert names[SIMILARITY + 1 :] == [f"resume:{term}" for term in terms]
    assert len(terms) > 100


def test_the_engineered_block_is_exactly_the_feature_vector(matrix, pairs) -> None:
    expected = np.array([feature_vector(job, candidate) for job, candidate in pairs])

    assert np.array_equal(matrix[:, :SIMILARITY].toarray(), expected)


def test_the_resume_block_is_each_candidates_own_tfidf_vector(fitted, matrix, pairs) -> None:
    expected = fitted.vectoriser_.transform([candidate_text(c) for _, c in pairs])

    assert rows_equal(matrix[:, SIMILARITY + 1 :], expected)


def test_text_similarity_is_the_cosine_of_the_two_documents(fitted, matrix, pairs) -> None:
    """Computed again from the raw vectors, so a job or candidate mixed up by row shows."""
    jobs = fitted.vectoriser_.transform([job_text(job) for job, _ in pairs]).toarray()
    people = fitted.vectoriser_.transform([candidate_text(c) for _, c in pairs]).toarray()
    cosine = (jobs * people).sum(axis=1) / (
        np.linalg.norm(jobs, axis=1) * np.linalg.norm(people, axis=1)
    )

    assert matrix[:, SIMILARITY].toarray().ravel() == pytest.approx(cosine, abs=1e-12)


def test_every_value_is_finite_and_in_range(matrix) -> None:
    similarity = matrix[:, SIMILARITY].toarray()
    resume = matrix[:, SIMILARITY + 1 :]

    assert np.isfinite(matrix.data).all()
    assert similarity.min() >= 0.0 and similarity.max() <= 1.0
    assert resume.min() >= 0.0 and resume.max() <= 1.0


# --- text similarity edges --------------------------------------------------


def test_a_job_that_reads_exactly_like_a_resume_scores_one_and_never_more(fitted, pairs) -> None:
    """Identical unit vectors can round past 1.0 in the dot product; the range is [0, 1]."""
    people = list(dict.fromkeys(candidate for _, candidate in pairs))
    job = pairs[0][0]
    twins = [
        (
            dataclasses.replace(
                job,
                title=person.current_title,
                description=person.resume_text,
                required_skills=person.skills,
                preferred_skills=(),
            ),
            person,
        )
        for person in people
    ]

    similarity = fitted.transform(twins)[:, SIMILARITY].toarray().ravel()

    assert similarity.max() <= 1.0
    assert similarity == pytest.approx(np.ones(len(people)), abs=1e-12)


def test_no_words_in_common_is_zero_similarity(fitted, pairs) -> None:
    job, candidate = pairs[0]
    chess = dataclasses.replace(
        job,
        title="Chess",
        description="Captain of a chess club",
        required_skills=(),
        preferred_skills=(),
    )
    cloud = dataclasses.replace(
        candidate, current_title="", resume_text="Kubernetes Terraform", skills=()
    )

    assert fitted.transform([(chess, cloud)])[0, SIMILARITY] == 0.0


def test_a_resume_with_no_known_word_is_a_zero_vector_not_a_nan(fitted, pairs) -> None:
    job, candidate = pairs[0]
    unknown = dataclasses.replace(
        candidate, current_title=None, resume_text="Zzyzx qwfpgj xylophonics", skills=()
    )

    row = fitted.transform([(job, unknown)])

    assert np.isfinite(row.toarray()).all()
    assert row[0, SIMILARITY] == 0.0
    assert row[:, SIMILARITY + 1 :].nnz == 0


# --- fitting ----------------------------------------------------------------


def test_the_vocabulary_is_learned_from_each_candidate_once_and_from_no_job(fitted, pairs) -> None:
    """Every candidate sits on several slates; counting each slate would skew the idf."""
    people = dict.fromkeys(candidate for _, candidate in pairs)
    assert len(people) < len(pairs)
    alone = make_vectoriser().fit([candidate_text(c) for c in people])

    assert fitted.vectoriser_.vocabulary_ == alone.vocabulary_
    assert np.array_equal(fitted.vectoriser_.idf_, alone.idf_)


def test_job_only_wording_never_becomes_a_column(fitted, corpus) -> None:
    """Every job says "We are hiring"; no resume does."""
    names = set(fitted.get_feature_names_out())

    assert all("hiring" in job.description for job in corpus.jobs)
    assert "resume:hiring" not in names
    assert "resume:we" not in names


def test_scoring_never_adds_a_term(fitted, pairs) -> None:
    job, candidate = pairs[0]
    novel = dataclasses.replace(candidate, resume_text=f"{candidate.resume_text} Zephyrology")
    before = list(fitted.get_feature_names_out())

    row = fitted.transform([(job, novel)])

    assert list(fitted.get_feature_names_out()) == before
    assert row.shape == (1, len(before))
    assert rows_equal(row, fitted.transform([(job, candidate)]))


def test_the_same_pairs_always_give_the_same_matrix(pairs, matrix) -> None:
    again = PairFeaturiser().fit(pairs)

    assert rows_equal(again.transform(pairs), matrix)


def test_rows_follow_the_order_of_the_pairs(fitted, matrix, pairs) -> None:
    backwards = fitted.transform(pairs[::-1])

    assert rows_equal(backwards, matrix[::-1])


def test_a_pair_repeated_in_one_batch_gives_the_same_row_each_time(fitted, pairs) -> None:
    rows = fitted.transform([pairs[3], pairs[7], pairs[3]])

    assert rows_equal(rows[0], rows[2])
    assert not rows_equal(rows[0], rows[1])


def test_scoring_before_fitting_is_refused(pairs) -> None:
    with pytest.raises(NotFittedError):
        PairFeaturiser().transform(pairs[:1])
    with pytest.raises(NotFittedError):
        PairFeaturiser().get_feature_names_out()


def test_an_empty_batch_is_refused_like_any_scikit_learn_transformer(fitted) -> None:
    with pytest.raises(ValueError, match="0 sample"):
        fitted.transform([])


def test_a_pickled_featuriser_gives_the_same_matrix(fitted, matrix, pairs) -> None:
    """The model artifact will carry the fitted vocabulary this way."""
    restored = pickle.loads(pickle.dumps(fitted))

    assert list(restored.get_feature_names_out()) == list(fitted.get_feature_names_out())
    assert rows_equal(restored.transform(pairs), matrix)


def test_a_clone_starts_unfitted(fitted) -> None:
    """What scikit-learn's pipelines and calibration do to an estimator before fitting it."""
    with pytest.raises(NotFittedError):
        clone(fitted).get_feature_names_out()


# --- what the matrix may and may not read -----------------------------------


def _as_orm(job: SyntheticJob, candidate: SyntheticCandidate) -> tuple[Job, Candidate]:
    return (
        Job(
            title=job.title,
            description=job.description,
            seniority=job.seniority,
            education_required=job.education_required,
            min_years_experience=job.min_years_experience,
            required_skills=list(job.required_skills),
            preferred_skills=list(job.preferred_skills),
        ),
        Candidate(
            full_name=candidate.full_name,
            email=candidate.email,
            current_title=candidate.current_title,
            resume_text=candidate.resume_text,
            seniority=candidate.seniority,
            education_level=candidate.education_level,
            years_experience=candidate.years_experience,
            skills=list(candidate.skills),
        ),
    )


def test_the_orm_models_and_the_synthetic_ones_give_the_same_rows(fitted, matrix, pairs) -> None:
    orm_pairs = [_as_orm(job, candidate) for job, candidate in pairs]

    assert rows_equal(fitted.transform(orm_pairs), matrix)


def test_the_matrix_never_reads_demographics_or_true_skills(fitted, pairs) -> None:
    sample = pairs[:200]
    changed = [
        (
            job,
            dataclasses.replace(
                person,
                gender="nonbinary" if person.gender != "nonbinary" else "woman",
                age_band="40_and_over" if person.age_band == "under_40" else "under_40",
                true_skills=frozenset(),
            ),
        )
        for job, person in sample
    ]

    assert rows_equal(fitted.transform(changed), fitted.transform(sample))


def test_the_planted_proxy_marks_exactly_the_resumes_that_carry_it(fitted, matrix, pairs) -> None:
    """Kept on purpose, so the fairness audit can measure what a model does with it."""
    column = list(fitted.get_feature_names_out()).index("resume:women")
    marked = matrix[:, column].toarray().ravel() != 0
    carries = np.array(["women" in candidate.resume_text.lower() for _, candidate in pairs])

    assert carries.any()
    assert np.array_equal(marked, carries)


def test_without_the_planted_phrase_there_is_no_proxy_column() -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200, proxy_share=0.0))
    names = set(PairFeaturiser().fit(pairs_of(corpus)).get_feature_names_out())

    assert "resume:women" not in names


def test_text_similarity_is_higher_for_better_fits(matrix, corpus) -> None:
    """Not a tuned threshold -- just that the feature points the way the labels do."""
    similarity = matrix[:, SIMILARITY].toarray().ravel()
    relevance = np.array([pair.relevance for pair in corpus.pairs])

    assert similarity[relevance == 2].mean() > similarity[relevance == 0].mean()
