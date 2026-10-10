"""Report endpoints: a job's cached ranking, and one candidate's place in it, as PDFs."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi import status as http_status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.api.v1.matches import _get_job_or_404, _ranked
from app.db.session import get_db
from app.ml.artifact import load_versioned_matcher
from app.models.candidate import Candidate
from app.models.match import MatchScore
from app.schemas.common import MAX_ID
from app.services.reports import MAX_SHORTLIST, scorecard_pdf, shortlist_pdf

router = APIRouter(prefix="/jobs", tags=["reports"])

_PDF_RESPONSE = {200: {"content": {"application/pdf": {}}, "description": "The report as a PDF"}}


def _pdf(content: bytes, filename: str) -> Response:
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


def _not_scored(detail: str) -> HTTPException:
    return HTTPException(status_code=http_status.HTTP_409_CONFLICT, detail=detail)


@router.get(
    "/{job_id}/report.pdf",
    response_class=Response,
    responses=_PDF_RESPONSE,
    summary="A job's ranked shortlist as a PDF",
)
def shortlist_report(
    job_id: int = Path(gt=0, le=MAX_ID),
    limit: int = Query(default=25, ge=1, le=MAX_SHORTLIST, description="Candidates to list"),
    db: Session = Depends(get_db),
) -> Response:
    """The best ``limit`` candidates cached for the job under the current matcher.

    Nothing is scored here: the ranks, scores and features are the ones
    ``GET /jobs/{id}/matches`` lists. A job with nothing cached under the
    current matcher is a 409 pointing at the scoring endpoint, rather than an
    empty report that reads as if nobody fitted.
    """
    job = _get_job_or_404(db, job_id)
    _, version = load_versioned_matcher()
    cached = (MatchScore.job_id == job_id, MatchScore.model_version == version)

    # The total comes with the rows, from the same snapshot: counted first and read
    # second, a scoring run committed in between made a report of "the top 7 of 6",
    # and scores deleted in between left nothing to list after a count that said
    # there was.
    found = db.execute(
        select(MatchScore, func.count().over().label("total"))
        .where(*cached)
        .options(joinedload(MatchScore.candidate))
        .order_by(MatchScore.score.desc(), MatchScore.candidate_id)
        .limit(limit)
    ).all()
    if not found:
        raise _not_scored(
            f"Job {job_id} has no scores under the current matcher. "
            f"Score it with POST /api/v1/jobs/{job_id}/score first."
        )
    rows = [row for row, _ in found]
    total = found[0].total
    ranked = [_ranked(row, row.candidate, rank) for rank, row in enumerate(rows, start=1)]

    pdf = shortlist_pdf(
        job, ranked, total=total, model_version=version, generated_at=datetime.now(UTC)
    )
    return _pdf(pdf, f"job-{job_id}-shortlist.pdf")


@router.get(
    "/{job_id}/candidates/{candidate_id}/scorecard.pdf",
    response_class=Response,
    responses=_PDF_RESPONSE,
    summary="One candidate's scorecard for a job as a PDF",
)
def scorecard_report(
    job_id: int = Path(gt=0, le=MAX_ID),
    candidate_id: int = Path(gt=0, le=MAX_ID),
    db: Session = Depends(get_db),
) -> Response:
    """The candidate's cached score for the job, its rank, skills, profile and features.

    The rank is the candidate's position in the job's cached ranking, the same
    one ``GET /jobs/{id}/matches`` gives it: best score first, ties to the lower
    candidate id. A missing job or candidate is a 404; a candidate with no
    score for the job under the current matcher is a 409.
    """
    job = _get_job_or_404(db, job_id)
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Candidate {candidate_id} not found",
        )
    _, version = load_versioned_matcher()
    cached = (MatchScore.job_id == job_id, MatchScore.model_version == version)

    row = db.scalars(
        select(MatchScore).where(*cached, MatchScore.candidate_id == candidate_id)
    ).first()
    if row is None:
        raise _not_scored(
            f"Candidate {candidate_id} has no score for job {job_id} under the current "
            f"matcher. Score it with POST /api/v1/jobs/{job_id}/score first."
        )
    ahead, total = db.execute(
        select(
            func.count().filter(
                or_(
                    MatchScore.score > row.score,
                    and_(MatchScore.score == row.score, MatchScore.candidate_id < candidate_id),
                )
            ),
            func.count(),
        ).where(*cached)
    ).one()

    pdf = scorecard_pdf(
        job,
        candidate,
        _ranked(row, candidate, ahead + 1),
        total=total,
        model_version=version,
        generated_at=datetime.now(UTC),
    )
    return _pdf(pdf, f"job-{job_id}-candidate-{candidate_id}-scorecard.pdf")
