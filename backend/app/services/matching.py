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

from collections.abc import Sequence
from datetime import UTC, datetime

from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import feature_dict, job_skill_keys
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.match import MatchScore


def score_pool(
    db: Session,
    job: Job,
    candidates: Sequence[Candidate],
    matcher: CalibratedClassifierCV,
    model_version: str,
) -> list[MatchScore]:
    """Score every candidate in ``candidates`` against ``job``, rank and cache the result.

    Returns the pool's :class:`MatchScore` rows, ranked best first. Ties in
    score keep the lower candidate id first, so a repeat run against an
    unchanged model and pool reproduces the same ranks rather than whatever
    order the database happened to hand back the candidates in.

    Rank is only meaningful relative to *this* pool: scoring a subset of a
    job's candidates does not renumber the rows this call leaves untouched.

    An empty pool is answered without touching the model -- there is nothing
    to rank, and ``PairFeaturiser.transform`` raises scikit-learn's own
    0-sample error on an empty batch by design.

    Rows are added (or updated in place) but not committed, the same
    convention :func:`app.services.stage_history.record_stage_event` uses:
    the caller commits, so a partial pool can never land half-scored.

    A candidate id repeated in ``candidates`` is scored once, keeping its
    first occurrence: a "pool" means the distinct people in it, and a caller
    building one from a request's id list should not have a duplicate turn
    into two insert attempts for the same (job, candidate, model_version)
    row.

    Not safe against two concurrent calls scoring the *same new* (job,
    candidate, model_version) row: both can see it as missing before either
    inserts, and the caller's commit -- not this function's, since it never
    commits -- is what raises the resulting ``IntegrityError``. The caller
    should catch it there the way ``commit_row`` catches a lost race for the
    other write routes; measured at 1 to 4 of 10 concurrent first-time scores of
    one row failing this way, with the row always left correct regardless.
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

    existing = {
        row.candidate_id: row
        for row in db.scalars(
            select(MatchScore).where(
                MatchScore.job_id == job.id,
                MatchScore.model_version == model_version,
                MatchScore.candidate_id.in_([candidate.id for candidate in candidates]),
            )
        )
    }

    job_keys = job_skill_keys(job)
    scored_at = datetime.now(UTC)
    ranked: list[MatchScore] = []
    for rank, index in enumerate(order, start=1):
        candidate = candidates[index]
        row = existing.get(candidate.id)
        if row is None:
            row = MatchScore(job_id=job.id, candidate_id=candidate.id, model_version=model_version)
            db.add(row)
        row.score = float(scores[index])
        row.rank = rank
        row.features = feature_dict(job, candidate, job_keys=job_keys)
        # Assigned explicitly, not left to the column's default: a rescore
        # that reproduces the same score, rank and features would otherwise
        # leave nothing for SQLAlchemy to consider changed, and an existing
        # row's timestamp would silently stay stale. One timestamp for the
        # whole run rather than `datetime.now()` per row, so two rows scored
        # in the same call can't disagree about when "this run" happened.
        row.scored_at = scored_at
        ranked.append(row)

    return ranked
