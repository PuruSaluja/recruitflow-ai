"""Matching endpoints: score a job's candidate pool, and read back the cached ranking."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi import status as http_status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.exc import StaleDataError

from app.api.deps import Pagination, pagination
from app.db.session import get_db
from app.ml.artifact import load_versioned_matcher
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.match import MatchScore
from app.schemas.candidate import CandidateSummary
from app.schemas.common import MAX_ID
from app.schemas.match import MatchPage, RankedCandidate, ScoreJobRequest, ScoreJobResponse
from app.services.matching import JobGone, score_pool

router = APIRouter(prefix="/jobs", tags=["matching"])

#: How many times a scoring run is tried before a lost race is reported as a
#: 409: one retry for a lost insert race, one more for a delete that knocks
#: back every concurrent run at once. See :func:`score_job`.
SCORING_ATTEMPTS = 3


def _job_not_found(job_id: int) -> HTTPException:
    return HTTPException(
        status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
    )


def _get_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise _job_not_found(job_id)
    return job


def _resolve_pool(db: Session, candidate_ids: list[int] | None) -> list[Candidate]:
    """The candidates to score: the ones asked for, or everyone on file.

    Every id asked for must exist. Scoring the rest and leaving one out would
    return a ranking the caller has no way to tell is incomplete.
    """
    if candidate_ids is None:
        return list(db.scalars(select(Candidate).order_by(Candidate.id)))

    wanted = sorted(set(candidate_ids))
    found = list(db.scalars(select(Candidate).where(Candidate.id.in_(wanted))))
    missing = sorted(set(wanted) - {candidate.id for candidate in found})
    if missing:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Candidates not found: {', '.join(map(str, missing))}",
        )
    return found


def _ranked(row: MatchScore, candidate: Candidate, rank: int) -> RankedCandidate:
    return RankedCandidate(
        rank=rank,
        score=row.score,
        scored_at=row.scored_at,
        candidate=CandidateSummary.model_validate(candidate),
        features=row.features,
    )


@router.post(
    "/{job_id}/score",
    response_model=ScoreJobResponse,
    summary="Score a candidate pool against a job and cache the ranking",
)
def score_job(
    payload: ScoreJobRequest | None = None,
    job_id: int = Path(gt=0, le=MAX_ID),
    db: Session = Depends(get_db),
) -> ScoreJobResponse:
    """Score candidates against a job with the trained matcher, rank them and cache the scores.

    Leaving out ``candidate_ids`` scores every candidate on file. The whole
    pool is scored and cached in ``match_scores``; ``limit`` only trims what
    comes back, and ranks are positions within this pool. Scoring again with
    an unchanged matcher updates the cached rows rather than adding more.

    A missing job or candidate is a 404, checked before the matcher is
    loaded. A missing or unreadable matcher is a 503 naming the command that
    regenerates it (see ``app.main``).

    A candidate deleted mid-run fails the commit. The run is then rolled back,
    re-read and scored again, and the candidate has dropped out of the pool
    (or is now a 404, if it was asked for by id). A row someone else inserted
    first fails it the same way, and the retry then updates that row. Up to
    three attempts are made, and only a third failure is answered with a 409.

    Other runs of the same job, an edit of the job and a delete of it take
    turns with this run's write (see :func:`score_pool`). A run that waited on
    another updates the rows that one saved instead of racing it to insert
    them, and a delete already under way finishes first, making this a 404.
    """
    request = payload or ScoreJobRequest()
    job = _get_job_or_404(db, job_id)
    candidates = _resolve_pool(db, request.candidate_ids)

    started = time.perf_counter()
    matcher, version = load_versioned_matcher()
    for attempt in range(1, SCORING_ATTEMPTS + 1):
        try:
            ranked = score_pool(db, job, candidates, matcher, version)
        except JobGone as exc:
            db.rollback()
            raise _job_not_found(job_id) from exc
        try:
            db.commit()
            break
        except (IntegrityError, StaleDataError) as exc:
            db.rollback()
            if attempt == SCORING_ATTEMPTS:
                raise HTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail=(
                        f"Scoring job {job_id} collided with concurrent changes to the same "
                        f"scores {SCORING_ATTEMPTS} times in a row. Retry the request."
                    ),
                ) from exc
            job = _get_job_or_404(db, job_id)
            candidates = _resolve_pool(db, request.candidate_ids)
    elapsed_ms = (time.perf_counter() - started) * 1000

    by_id = {candidate.id: candidate for candidate in candidates}
    return ScoreJobResponse(
        job_id=job_id,
        model_version=version,
        scored_count=len(ranked),
        elapsed_ms=round(elapsed_ms, 1),
        results=[
            _ranked(row, by_id[row.candidate_id], row.rank) for row in ranked[: request.limit]
        ],
    )


@router.get(
    "/{job_id}/matches",
    response_model=MatchPage,
    summary="A job's cached ranking under the current matcher",
)
def list_matches(
    job_id: int = Path(gt=0, le=MAX_ID),
    page: Pagination = Depends(pagination),
    db: Session = Depends(get_db),
) -> MatchPage:
    """Page through the scores cached for a job, best first. Nothing is scored here.

    Only scores from the matcher currently on disk are listed; rows an
    earlier model left behind stay in the table but are never mixed in. Ranks
    run across everything cached for the job under that matcher, so they stay
    1 to ``total`` even when some candidates were scored in a separate,
    smaller run -- which is the one case where they differ from the rank that
    run returned. A score is as of its ``scored_at``: editing the job or a
    candidate afterwards does not rescore it.

    The matcher is loaded here too, not just hashed, so a file that exists
    but cannot be used is the same 503 scoring gives. Hashing alone would
    list nothing for it -- no score can carry the version of a file that
    never loaded -- and every shortlist would look emptied.
    """
    _get_job_or_404(db, job_id)
    _, version = load_versioned_matcher()
    cached = (MatchScore.job_id == job_id, MatchScore.model_version == version)

    total = db.execute(select(func.count()).select_from(MatchScore).where(*cached)).scalar_one()
    rows = db.scalars(
        select(MatchScore)
        .where(*cached)
        .options(joinedload(MatchScore.candidate))
        .order_by(MatchScore.score.desc(), MatchScore.candidate_id)
        .limit(page.limit)
        .offset(page.offset)
    ).all()

    return MatchPage(
        items=[
            _ranked(row, row.candidate, rank)
            for rank, row in enumerate(rows, start=page.offset + 1)
        ],
        total=total,
        limit=page.limit,
        offset=page.offset,
        job_id=job_id,
        model_version=version,
    )
