"""Scoring a candidate pool against a job with the persisted matcher.

Scores are cached into ``match_scores`` rather than recomputed per request
(see that model's docstring): the same slate gets paged through repeatedly,
and re-running the matcher on every page view would waste the work.
``model_version`` identifies which trained artifact produced a score --
:func:`app.ml.artifact.matcher_version`, a hash of the matcher file's bytes --
so a retrain does not silently overwrite the previous model's cached rows
under the same key, and scoring again with an unchanged artifact updates
them in place instead of duplicating them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.ml.features import feature_dict, job_skill_keys
from app.models.candidate import Candidate
from app.models.job import Job

#: Rows written per statement. Each batch is one statement with one array
#: parameter per column, so this bounds a statement's size, not its parameter
#: count.
WRITE_BATCH = 1000

# Inserts each row, or updates the one already cached under the same (job,
# candidate, model_version), in one statement per batch instead of one ORM
# insert or update per row: the job stays locked for as long as the write
# takes (see score_pool). ORDER BY makes the rows go in by candidate id
# whatever order the arrays arrive in; without it the order would rest on
# the plan feeding the function scan straight to the insert.
_UPSERT = text(
    "INSERT INTO match_scores "
    "(job_id, candidate_id, model_version, score, rank, features, scored_at) "
    "SELECT :job_id, pool.candidate_id, :model_version, pool.score, pool.rank, "
    "CAST(pool.features AS jsonb), :scored_at "
    "FROM unnest("
    "CAST(:candidate_ids AS integer[]), CAST(:scores AS double precision[]), "
    "CAST(:ranks AS integer[]), CAST(:features AS text[])"
    ") AS pool(candidate_id, score, rank, features) "
    "ORDER BY pool.candidate_id "
    "ON CONFLICT ON CONSTRAINT uq_match_job_cand_version DO UPDATE SET "
    "score = excluded.score, rank = excluded.rank, features = excluded.features, "
    "scored_at = excluded.scored_at "
    "RETURNING id, candidate_id"
)


@dataclass(frozen=True)
class CachedScore:
    """One candidate's ``match_scores`` row, as :func:`score_pool` wrote it."""

    id: int
    job_id: int
    candidate_id: int
    model_version: str
    score: float
    rank: int
    features: dict[str, float]
    scored_at: datetime


class JobGone(LookupError):
    """The job was deleted before its pool's scores could be written."""

    def __init__(self, job_id: int):
        super().__init__(f"Job {job_id} not found")
        self.job_id = job_id


def score_pool(
    db: Session,
    job: Job,
    candidates: Sequence[Candidate],
    matcher: CalibratedClassifierCV,
    model_version: str,
) -> list[CachedScore]:
    """Score every candidate in ``candidates`` against ``job``, rank and cache the result.

    Returns the pool's rows as :class:`CachedScore` values, ranked best first.
    They are what was written, not ORM objects: the rows go in with a
    set-based statement, so the session does not track them. Ties in
    score keep the lower candidate id first, so a repeat run against an
    unchanged model and pool reproduces the same ranks rather than whatever
    order the database happened to hand back the candidates in.

    Rank is only meaningful relative to *this* pool: scoring a subset of a
    job's candidates does not renumber the rows this call leaves untouched.

    An empty pool is answered without touching the model -- there is nothing
    to rank, and ``PairFeaturiser.transform`` raises scikit-learn's own
    0-sample error on an empty batch by design.

    Rows are written (inserted, or updated in place) but not committed, the
    same convention :func:`app.services.stage_history.record_stage_event`
    uses: the caller commits, so a partial pool can never land half-scored.
    They are written sorted by candidate id, ``WRITE_BATCH`` to a statement.

    A candidate id repeated in ``candidates`` is scored once, keeping its
    first occurrence: a "pool" means the distinct people in it, and a caller
    building one from a request's id list should not have a duplicate turn
    into two insert attempts for the same (job, candidate, model_version)
    row.

    A row someone else committed first, under the same (job, candidate,
    model_version), is updated rather than inserted twice. A candidate
    deleted before its row is written fails that statement's foreign key
    check, so the ``IntegrityError`` is raised here, not at the caller's
    commit; the caller rolls back and scores again. The job's foreign key is
    checked by the same statements, but it cannot fail that way: the job's
    row is held from before the first write (below).

    The job's row is locked FOR NO KEY UPDATE, the lock an edit or a delete of
    the job takes first, from before the scores are written until the caller
    commits or rolls back, and :class:`JobGone` is raised if the job
    has been deleted by then. So this run's write takes turns with a delete of
    the job, an edit of it and another run scoring it: whichever locked the job
    first finishes, and the other waits. A run that waited on another updates
    the scores it saved rather than racing it to insert them.
    Interleaved, a run and a delete deadlocked in several orders, and so did
    two runs that ranked the same new candidates differently, or where one
    rescored a row the other was inserting. New applications to the job take
    only FOR KEY SHARE on it, so they do not wait. Scores, breakdowns and the
    write's parameters are computed before the lock is taken, so a wait
    covers only the other write.
    """
    if not candidates:
        return []

    deduped: dict[int, Candidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.id, candidate)
    candidates = list(deduped.values())
    pairs = [(job, candidate) for candidate in candidates]
    scores = matcher.predict_proba(pairs)[:, 1]
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].id))
    job_keys = job_skill_keys(job)
    breakdowns = [feature_dict(job, candidate, job_keys=job_keys) for candidate in candidates]

    ranks = {index: rank for rank, index in enumerate(order, start=1)}
    # Written in candidate id order, whatever the ranking, so every run takes
    # its row locks in the same order. Built before the lock, like the scores.
    by_id = sorted(range(len(candidates)), key=lambda i: candidates[i].id)
    batches = [
        {
            "job_id": job.id,
            "model_version": model_version,
            # One array parameter per column, not one parameter per value:
            # psycopg refuses a statement with more than 65,535.
            "candidate_ids": [candidates[i].id for i in batch],
            "scores": [float(scores[i]) for i in batch],
            "ranks": [ranks[i] for i in batch],
            "features": [json.dumps(breakdowns[i]) for i in batch],
        }
        for batch in (
            by_id[start : start + WRITE_BATCH] for start in range(0, len(by_id), WRITE_BATCH)
        )
    ]

    # WRITE_LOCK in app.api.deps, the lock every other write to a job takes.
    held = select(Job.id).where(Job.id == job.id).with_for_update(key_share=True)
    if db.scalar(held) is None:
        raise JobGone(job.id)

    # One timestamp for the whole run rather than `now()` per row, so two rows
    # scored in the same call can't disagree about when "this run" happened.
    # Written on update too, so a rescore that reproduces the same score, rank
    # and features still moves it. Taken once the lock is held, so a run that
    # waited on another is stamped after it.
    scored_at = datetime.now(UTC)
    row_ids: dict[int, int] = {}
    for params in batches:
        written = db.execute(_UPSERT, {**params, "scored_at": scored_at})
        row_ids.update((candidate_id, row_id) for row_id, candidate_id in written)

    return [
        CachedScore(
            id=row_ids[candidates[index].id],
            job_id=job.id,
            candidate_id=candidates[index].id,
            model_version=model_version,
            score=float(scores[index]),
            rank=rank,
            features=breakdowns[index],
            scored_at=scored_at,
        )
        for rank, index in enumerate(order, start=1)
    ]
