"""Integration tests for scoring a candidate pool and caching it into match_scores.

These drive :func:`app.services.matching.score_pool` directly against real
``Job`` and ``Candidate`` rows, with a matcher that is actually fitted rather
than stubbed, the same way the ml layer's own tests do. The endpoints built
on it are covered in ``test_matching_api.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from psycopg.errors import ForeignKeyViolation
from sqlalchemy import Engine, delete, event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ml import features as features_module
from app.ml.artifact import matcher_version, save_matcher
from app.ml.corpus import CorpusConfig, SyntheticCandidate, SyntheticJob, generate_corpus
from app.ml.features import FEATURE_NAMES, feature_dict
from app.ml.train import train_matcher, training_data
from app.models import Candidate, Job, MatchScore
from app.services import matching
from app.services.matching import score_pool

Factory = Callable[..., dict[str, Any]]


class _ExplodingMatcher:
    """A matcher that fails the test if it is ever asked to predict anything."""

    def predict_proba(self, pairs: object) -> None:
        raise AssertionError("predict_proba must not be called for an empty pool")


@pytest.fixture(scope="module")
def corpus():
    """Small but real corpus, shared read-only across this module's tests."""
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


@pytest.fixture(scope="module")
def matcher(corpus):
    """A really-fitted matcher -- no stub, so a score is a score scikit-learn produced."""
    return train_matcher(training_data(corpus))


@pytest.fixture(scope="module")
def version(matcher, tmp_path_factory: pytest.TempPathFactory) -> str:
    """The matcher's real content-derived version, from an actual write to disk."""
    path = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    save_matcher(matcher, manifest={}, path=path)
    return matcher_version(path)


def _job_row(db: Session, make_job: Factory, synthetic_job: SyntheticJob) -> Job:
    created = make_job(**synthetic_job.api_payload())
    row = db.get(Job, created["id"])
    assert row is not None
    return row


def _candidate_row(
    db: Session, make_candidate: Factory, synthetic_candidate: SyntheticCandidate
) -> Candidate:
    created = make_candidate(**synthetic_candidate.api_payload())
    row = db.get(Candidate, created["id"])
    assert row is not None
    return row


@pytest.fixture
def pool(clean_db: Engine, make_job: Factory, make_candidate: Factory, corpus):
    """One real job and five real candidates from the corpus, as ORM rows in one session."""
    with Session(clean_db) as db:
        job = _job_row(db, make_job, corpus.jobs[0])
        candidates = [_candidate_row(db, make_candidate, c) for c in corpus.candidates[:5]]
        yield db, job, candidates


def _stored(db: Session, job: Job) -> list[MatchScore]:
    return list(db.scalars(select(MatchScore).where(MatchScore.job_id == job.id)))


# --- scoring a real pool -----------------------------------------------------


def test_scores_every_candidate_in_the_pool(pool, matcher, version) -> None:
    db, job, candidates = pool

    results = score_pool(db, job, candidates, matcher, version)

    assert {row.candidate_id for row in results} == {c.id for c in candidates}
    assert all(0.0 <= row.score <= 1.0 for row in results)


def test_ranks_are_a_permutation_of_1_through_n(pool, matcher, version) -> None:
    db, job, candidates = pool

    results = score_pool(db, job, candidates, matcher, version)

    assert sorted(row.rank for row in results) == list(range(1, len(candidates) + 1))


def test_rank_order_matches_descending_score_order(pool, matcher, version) -> None:
    db, job, candidates = pool

    results = score_pool(db, job, candidates, matcher, version)

    by_rank = sorted(results, key=lambda row: row.rank)
    scores_in_rank_order = [row.score for row in by_rank]
    assert scores_in_rank_order == sorted(scores_in_rank_order, reverse=True)


def test_persisted_features_are_exactly_the_engineered_feature_dict(pool, matcher, version) -> None:
    """Decision: the cached breakdown is the 8 engineered features, not the resume-term matrix.

    There is no single fitted vocabulary to read a `text_similarity` back from
    once the matcher is calibrated -- CalibratedClassifierCV keeps one fitted
    featuriser per fold, sometimes with different vocabularies (see
    app.ml.train's docstring) -- so a per-score explanation sticks to the half
    that is reproducible outside the model entirely.
    """
    db, job, candidates = pool
    by_id = {c.id: c for c in candidates}

    results = score_pool(db, job, candidates, matcher, version)

    for row in results:
        assert row.features == feature_dict(job, by_id[row.candidate_id])
        assert set(row.features) == set(FEATURE_NAMES)


def test_every_row_carries_the_given_model_version(pool, matcher, version) -> None:
    db, job, candidates = pool

    results = score_pool(db, job, candidates, matcher, version)

    assert all(row.model_version == version for row in results)


def test_results_are_persisted_once_committed(pool, matcher, version) -> None:
    db, job, candidates = pool

    score_pool(db, job, candidates, matcher, version)
    db.commit()

    assert len(_stored(db, job)) == len(candidates)


def test_nothing_is_written_before_the_caller_commits(pool, matcher, version) -> None:
    """The service adds rows but never commits -- a caller that rolls back writes nothing."""
    db, job, candidates = pool

    score_pool(db, job, candidates, matcher, version)
    db.rollback()

    assert _stored(db, job) == []


def test_the_jobs_skills_are_keyed_once_per_featurisation_not_per_candidate(
    pool, matcher, version, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured on 2026-09-22: keying the job's lists again for every candidate was
    18 of the 24 ``skill_keys`` calls each scored pair cost, and ``skill_keys``
    was about half of a whole-pool request's time.

    The pool is featurised once per calibration fold and once more for the
    stored breakdown. Each of those keys every candidate's list once and the
    job's three lists (required, preferred, both) once.
    """
    db, job, candidates = pool
    calls = 0
    real = features_module.skill_keys

    def counting(spellings):
        nonlocal calls
        calls += 1
        return real(spellings)

    monkeypatch.setattr(features_module, "skill_keys", counting)
    score_pool(db, job, candidates, matcher, version)

    featurisations = len(matcher.calibrated_classifiers_) + 1
    assert calls == featurisations * (len(candidates) + 3)


class _SpreadMatcher:
    """Scores a pool evenly from 0.01 to 0.99 in the order given, without featurising it."""

    def predict_proba(self, pairs: list[object]) -> np.ndarray:
        scores = np.linspace(0.01, 0.99, len(pairs))
        return np.column_stack([1 - scores, scores])


def test_a_pool_past_psycopgs_parameter_limit_is_written_over_its_cached_rows(pool) -> None:
    """Found 2026-09-22: the cached rows were looked up with one parameter per id.

    psycopg refuses a statement with more than 65,535 parameters, so with the
    query's two others, whole-pool scoring failed from 65,534 candidates on
    file, and the client was told the database was unavailable. The ids now
    travel as one array, and so does each column the write sends.

    The five real ones were scored first, and the big run must update their
    rows rather than add more. The 65,536 copies of the first are saved with
    one statement and rolled back with the run.
    """
    db, job, candidates = pool
    first = score_pool(db, job, candidates, _SpreadMatcher(), "v1")
    db.commit()

    db.execute(
        text(
            "INSERT INTO candidates (full_name, email, years_experience, seniority, "
            "education_level, resume_text, skills) "
            "SELECT full_name, 'pool' || n || '@example.com', years_experience, seniority, "
            "education_level, resume_text, skills "
            "FROM candidates, generate_series(1, 65536) AS n WHERE id = :id"
        ),
        {"id": candidates[0].id},
    )
    everyone = list(db.scalars(select(Candidate)))
    rows = score_pool(db, job, everyone, _SpreadMatcher(), "v1")

    assert len(rows) == len(everyone) == len(candidates) + 65_536
    cached = {row.candidate_id: row.id for row in first}
    assert {row.candidate_id: row.id for row in rows if row.candidate_id in cached} == cached
    stored = db.scalar(select(func.count()).select_from(MatchScore))
    assert stored == len(everyone)
    db.rollback()


# --- how the rows are written ------------------------------------------------


@contextmanager
def _score_writes(on_write: Callable[[int], None] | None = None) -> Iterator[list[list[int]]]:
    """Collect each score write's candidate ids; ``on_write(n)`` runs after the n-th."""
    writes: list[list[int]] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        if statement.startswith("INSERT INTO match_scores"):
            writes.append(list(parameters["candidate_ids"]))
            if on_write is not None:
                on_write(len(writes))

    event.listen(Engine, "after_cursor_execute", record)
    try:
        yield writes
    finally:
        event.remove(Engine, "after_cursor_execute", record)


def test_rows_are_written_in_candidate_id_order_a_batch_at_a_time(
    pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pool is shuffled so that the order given, the ranking and id order all differ."""
    db, job, candidates = pool
    monkeypatch.setattr(matching, "WRITE_BATCH", 2)
    by_id = sorted(candidates, key=lambda candidate: candidate.id)
    ids = [candidate.id for candidate in by_id]
    given = [by_id[i] for i in (2, 0, 4, 1, 3)]
    # _SpreadMatcher scores rise along the order given, so the last is ranked first.
    ranked = [candidate.id for candidate in given[::-1]]
    assert len({tuple(ids), tuple(c.id for c in given), tuple(ranked)}) == 3

    with _score_writes() as writes:
        rows = score_pool(db, job, given, _SpreadMatcher(), "v1")
    db.commit()

    assert writes == [ids[0:2], ids[2:4], ids[4:5]]
    assert [row.candidate_id for row in rows] == ranked
    assert {row.candidate_id: row.rank for row in _stored(db, job)} == {
        candidate_id: rank for rank, candidate_id in enumerate(ranked, start=1)
    }


def test_a_rescore_updates_each_row_in_place_whatever_its_batch(
    pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, job, candidates = pool
    first = score_pool(db, job, candidates, _SpreadMatcher(), "v1")
    db.commit()
    monkeypatch.setattr(matching, "WRITE_BATCH", 2)

    again = score_pool(db, job, candidates[::-1], _SpreadMatcher(), "v1")
    db.commit()

    first_ids = {row.candidate_id: row.id for row in first}
    assert {row.candidate_id: row.id for row in again} == first_ids
    stored = {row.candidate_id: row for row in _stored(db, job)}
    assert len(stored) == len(candidates)
    for row in again:
        assert (stored[row.candidate_id].score, stored[row.candidate_id].rank) == (
            row.score,
            row.rank,
        )


def test_a_rescore_that_changes_nothing_else_still_stores_its_scored_at(pool) -> None:
    """Checked in the table, not only on the returned rows, which carry the run's own time."""
    db, job, candidates = pool
    score_pool(db, job, candidates, _SpreadMatcher(), "v1")
    db.commit()
    before = {row.candidate_id: (row.score, row.rank, row.scored_at) for row in _stored(db, job)}

    again = score_pool(db, job, candidates, _SpreadMatcher(), "v1")
    db.commit()

    stored = {row.candidate_id: (row.score, row.rank, row.scored_at) for row in _stored(db, job)}
    for row in again:
        score, rank, scored_at = stored[row.candidate_id]
        assert (score, rank) == before[row.candidate_id][:2]
        assert scored_at == row.scored_at > before[row.candidate_id][2]


def test_a_candidate_deleted_between_two_batches_fails_the_write_and_nothing_lands(
    pool, clean_db: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The foreign key check fails at the later batch's statement, not at the commit.

    The delete gives up after 5 s rather than hanging if the run has already
    written that candidate's row, which a wrong write order would do.
    """
    db, job, candidates = pool
    monkeypatch.setattr(matching, "WRITE_BATCH", 2)
    last = max(candidate.id for candidate in candidates)

    def delete_last(written: int) -> None:
        if written == 1:
            with clean_db.begin() as other:
                other.execute(text("SET LOCAL lock_timeout = '5s'"))
                other.execute(delete(Candidate).where(Candidate.id == last))

    with _score_writes(delete_last) as writes, pytest.raises(IntegrityError) as caught:
        score_pool(db, job, candidates, _SpreadMatcher(), "v1")
    db.rollback()

    assert len(writes) == 2
    assert isinstance(caught.value.orig, ForeignKeyViolation)
    assert _stored(db, job) == []


# --- a duplicate in the pool --------------------------------------------------


def test_a_repeated_candidate_id_is_scored_once(pool, matcher, version) -> None:
    """A caller may build the pool from a list with repeats; a duplicate must not turn
    into two insert attempts for the same (job, candidate, model_version) row."""
    db, job, candidates = pool
    twice = [candidates[0], candidates[1], candidates[0]]

    results = score_pool(db, job, twice, matcher, version)
    db.commit()

    assert {row.candidate_id for row in results} == {candidates[0].id, candidates[1].id}
    assert sorted(row.rank for row in results) == [1, 2]
    assert len(_stored(db, job)) == 2


def test_a_repeated_candidate_id_collapses_even_as_two_distinct_objects(
    pool, clean_db: Engine, matcher, version
) -> None:
    """Dedup keys on candidate id, not object identity -- a second session's copy collapses too."""
    db, job, candidates = pool
    with Session(clean_db) as other_session:
        twin = other_session.get(Candidate, candidates[0].id)
        assert twin is not candidates[0]

        results = score_pool(db, job, [candidates[0], twin], matcher, version)

    assert len(results) == 1


def test_a_repeated_candidate_id_keeps_the_first_occurrences_data(
    clean_db: Engine, make_job: Factory, make_candidate: Factory, matcher, version
) -> None:
    """Proves *which* occurrence wins with data the two disagree on, not just that they
    collapse to one row -- two objects holding identical data could never tell the two
    apart."""
    job_payload = make_job(required_skills=["Python"], preferred_skills=[])
    candidate_payload = make_candidate(skills=["Python"])

    with Session(clean_db) as db, Session(clean_db) as other:
        job = db.get(Job, job_payload["id"])
        first = db.get(Candidate, candidate_payload["id"])
        second = other.get(Candidate, candidate_payload["id"])
        second.skills = []  # a genuine, never-committed divergence in a separate session

        results = score_pool(db, job, [first, second], matcher, version)

        assert len(results) == 1
        assert results[0].features["required_skill_coverage"] == 1.0  # first's data, not second's


# --- empty pool --------------------------------------------------------------


def test_an_empty_pool_is_answered_without_touching_the_model(pool) -> None:
    db, job, _candidates = pool

    assert score_pool(db, job, [], _ExplodingMatcher(), "v1") == []


def test_an_empty_pool_writes_nothing(pool) -> None:
    db, job, _candidates = pool

    score_pool(db, job, [], _ExplodingMatcher(), "v1")
    db.commit()

    assert _stored(db, job) == []


# --- ranking sanity: a real, obvious gap between two candidates -------------


def test_a_clearly_stronger_candidate_outranks_a_clearly_weaker_one(
    clean_db: Engine, make_job: Factory, make_candidate: Factory, matcher, version
) -> None:
    with Session(clean_db) as db:
        job = db.get(
            Job,
            make_job(
                title="Senior Data Engineer",
                description="Own the warehouse and the pipelines that feed every dashboard.",
                seniority="senior",
                education_required="bachelor",
                min_years_experience=6,
                required_skills=["Python", "SQL", "Airflow"],
                preferred_skills=["dbt"],
            )["id"],
        )
        strong = db.get(
            Candidate,
            make_candidate(
                full_name="Strong Fit",
                current_title="Senior Data Engineer",
                years_experience=8,
                seniority="senior",
                education_level="bachelor",
                resume_text=(
                    "Built and ran the Airflow-orchestrated warehouse pipelines that the "
                    "whole company's dashboards depend on, in Python and SQL."
                ),
                skills=["Python", "SQL", "Airflow", "dbt"],
            )["id"],
        )
        weak = db.get(
            Candidate,
            make_candidate(
                full_name="Weak Fit",
                years_experience=0,
                seniority="intern",
                education_level="high_school",
                resume_text="Looking for my first role, open to learning anything.",
                skills=[],
            )["id"],
        )

        results = score_pool(db, job, [strong, weak], matcher, version)

        by_id = {row.candidate_id: row for row in results}
        assert by_id[strong.id].score > by_id[weak.id].score
        assert by_id[strong.id].rank == 1
        assert by_id[weak.id].rank == 2


def test_ties_break_by_the_lower_candidate_id(
    clean_db: Engine, make_job: Factory, make_candidate: Factory, matcher, version, corpus
) -> None:
    """Two candidates identical on every field the matcher reads must score identically."""
    twin = {
        "years_experience": 4,
        "seniority": "mid",
        "education_level": "bachelor",
        "current_title": "Software Engineer",
        "resume_text": "Shipped internal tools end to end with Python, SQL and some React.",
        "skills": ["Python", "SQL", "React"],
    }
    with Session(clean_db) as db:
        job = _job_row(db, make_job, corpus.jobs[0])
        first = db.get(Candidate, make_candidate(**twin)["id"])
        second = db.get(Candidate, make_candidate(**twin)["id"])
        assert first.id < second.id

        # Passed in reverse id order, so a correct tie-break has to sort them,
        # not just preserve whatever order they arrived in.
        results = score_pool(db, job, [second, first], matcher, version)

        by_id = {row.candidate_id: row for row in results}
        assert by_id[first.id].score == by_id[second.id].score
        assert by_id[first.id].rank == 1
        assert by_id[second.id].rank == 2


# --- rescoring: cache semantics ----------------------------------------------


def test_rescoring_the_same_pool_updates_rows_instead_of_duplicating(
    pool, matcher, version
) -> None:
    db, job, candidates = pool

    first_run = score_pool(db, job, candidates, matcher, version)
    db.commit()
    first_ids = {row.candidate_id: row.id for row in first_run}

    second_run = score_pool(db, job, candidates, matcher, version)
    db.commit()

    assert {row.candidate_id: row.id for row in second_run} == first_ids
    assert len(_stored(db, job)) == len(candidates)


def test_rescoring_bumps_scored_at(pool, matcher, version) -> None:
    db, job, candidates = pool

    first_run = score_pool(db, job, candidates, matcher, version)
    db.commit()
    first_scored_at = {row.candidate_id: row.scored_at for row in first_run}

    second_run = score_pool(db, job, candidates, matcher, version)
    db.commit()

    assert all(row.scored_at > first_scored_at[row.candidate_id] for row in second_run)


def test_a_different_model_version_gets_its_own_rows(pool, matcher, version) -> None:
    db, job, candidates = pool
    other_version = ("0" if version[0] != "0" else "1") + version[1:]
    assert other_version != version

    score_pool(db, job, candidates, matcher, version)
    db.commit()
    score_pool(db, job, candidates, matcher, other_version)
    db.commit()

    stored = _stored(db, job)
    assert len(stored) == 2 * len(candidates)
    assert {row.model_version for row in stored} == {version, other_version}


def test_scoring_a_subset_does_not_renumber_rows_left_out_of_it(pool, matcher, version) -> None:
    db, job, candidates = pool
    assert len(candidates) >= 3

    score_pool(db, job, candidates, matcher, version)
    db.commit()
    before = {row.candidate_id: row.rank for row in _stored(db, job)}

    subset = candidates[:2]
    score_pool(db, job, subset, matcher, version)
    db.commit()

    after = {row.candidate_id: row.rank for row in _stored(db, job)}
    left_out_ids = {c.id for c in candidates} - {c.id for c in subset}
    assert left_out_ids, "the pool fixture must be larger than the subset for this to mean anything"
    for candidate_id in left_out_ids:
        assert after[candidate_id] == before[candidate_id]
    assert sorted(after[c.id] for c in subset) == list(range(1, len(subset) + 1))
