"""Application endpoints: a candidate's progress through one job's pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi import status as http_status
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from app.api.deps import Pagination, pagination
from app.db.session import get_db
from app.models.application import Application
from app.models.candidate import Candidate
from app.models.enums import ApplicationStatus, JobStatus
from app.models.job import Job
from app.schemas.application import (
    ApplicationCreate,
    ApplicationRead,
    ApplicationStageEventRead,
    ApplicationUpdate,
)
from app.schemas.common import Page
from app.services.pipeline import TransitionError, check_transition
from app.services.stage_history import record_stage_event

router = APIRouter(prefix="/applications", tags=["applications"])


def _get_or_404(db: Session, application_id: int) -> Application:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Application {application_id} not found",
        )
    return application


def _commit_or_conflict(db: Session, application_id: int) -> None:
    """Commit a write to one application, answering 409 if it lost a race.

    The version column makes an UPDATE that raced another write match no row.
    Rolling back drops any stage event added alongside it, so the trail never
    records a move that did not take effect.
    """
    try:
        db.commit()
    except StaleDataError as exc:
        db.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                f"Application {application_id} was changed by someone else while this "
                "update was in flight. Re-read it and retry."
            ),
        ) from exc


def _apply_filters(
    stmt: Select,
    *,
    job_id: int | None,
    candidate_id: int | None,
    status: ApplicationStatus | None,
    source: str | None,
) -> Select:
    """Apply the same predicates to both the page query and the count query.

    Shared so ``total`` stays honest: a filtered page whose total was computed
    unfiltered would silently break pagination in the UI.
    """
    if job_id is not None:
        stmt = stmt.where(Application.job_id == job_id)
    if candidate_id is not None:
        stmt = stmt.where(Application.candidate_id == candidate_id)
    if status is not None:
        stmt = stmt.where(Application.status == status)
    if source is not None:
        stmt = stmt.where(Application.source == source)
    return stmt


@router.get("", response_model=Page[ApplicationRead], summary="List applications")
def list_applications(
    db: Session = Depends(get_db),
    page: Pagination = Depends(pagination),
    job_id: int | None = Query(None, gt=0, description="Only this job's pipeline"),
    candidate_id: int | None = Query(None, gt=0, description="Only this candidate's applications"),
    status: ApplicationStatus | None = Query(None, description="Filter by pipeline stage"),
    source: str | None = Query(None, description="Exact match on how the application arrived"),
) -> Page[ApplicationRead]:
    filters = {
        "job_id": job_id,
        "candidate_id": candidate_id,
        "status": status,
        "source": source,
    }

    total = db.execute(
        _apply_filters(select(func.count()).select_from(Application), **filters)
    ).scalar_one()

    stmt = _apply_filters(select(Application), **filters)
    stmt = (
        stmt.order_by(Application.applied_at.desc(), Application.id.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    applications = db.execute(stmt).scalars().all()

    return Page[ApplicationRead](
        items=[ApplicationRead.model_validate(a) for a in applications],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "",
    response_model=ApplicationRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="Apply a candidate to a job",
)
def create_application(payload: ApplicationCreate, db: Session = Depends(get_db)) -> Application:
    """Open an application at the first pipeline stage.

    A missing job or candidate is a 404 naming which one; an unopen requisition
    or a repeat application is a 409, since both conflict with existing state
    rather than being malformed requests.
    """
    job = db.get(Job, payload.job_id)
    if job is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Job {payload.job_id} not found",
        )
    if db.get(Candidate, payload.candidate_id) is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Candidate {payload.candidate_id} not found",
        )

    if job.status is not JobStatus.OPEN:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job.id} is {job.status.value}; only open requisitions "
                "accept new applications"
            ),
        )

    data = payload.model_dump()
    changed_by = data.pop("changed_by")
    application = Application(**data)
    db.add(application)
    try:
        db.flush()
    except IntegrityError as exc:
        # The unique constraint is the real guard against a double-apply: two
        # simultaneous requests both pass the checks above, and the database
        # settles it. Anything else is left to the global handler.
        db.rollback()
        if "uq_applications_job_candidate" not in str(exc.orig):
            raise
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                f"Candidate {payload.candidate_id} has already applied to job {payload.job_id}"
            ),
        ) from exc

    # from_status=None marks this as the opening event: there is no prior stage.
    record_stage_event(
        db, application, from_status=None, to_status=application.status, changed_by=changed_by
    )
    db.commit()
    db.refresh(application)
    return application


@router.get("/{application_id}", response_model=ApplicationRead, summary="Fetch one application")
def get_application(application_id: int = Path(gt=0), db: Session = Depends(get_db)) -> Application:
    return _get_or_404(db, application_id)


@router.get(
    "/{application_id}/stage-events",
    response_model=list[ApplicationStageEventRead],
    summary="Full pipeline history for one application",
)
def list_stage_events(application_id: int = Path(gt=0), db: Session = Depends(get_db)) -> list:
    """Every status move this application has made, oldest first.

    Answers "how long did screening take" and similar questions without
    reconstructing history from ``updated_at``. A handful of rows per
    application, so this returns the whole list rather than a page.
    """
    application = _get_or_404(db, application_id)
    return application.stage_events


@router.patch(
    "/{application_id}",
    response_model=ApplicationRead,
    summary="Update an application, including its pipeline stage",
)
def update_application(
    payload: ApplicationUpdate,
    application_id: int = Path(gt=0),
    db: Session = Depends(get_db),
) -> Application:
    """Partially update an application.

    A status change must be a legal move through the pipeline; an illegal one
    is a 409 naming the stages that *are* reachable. Re-sending the current
    status is a no-op, so a retried request does not fail.
    """
    application = _get_or_404(db, application_id)

    # exclude_unset distinguishes "field omitted" from "field set to null", so
    # a PATCH only touches what the caller actually sent.
    updates = payload.model_dump(exclude_unset=True)
    target = updates.pop("status", None)
    # Not a column on Application; only meaningful when it accompanies a real
    # status move below, so it never reaches the setattr loop.
    changed_by = updates.pop("changed_by", None)

    if target is not None and target != application.status:
        try:
            check_transition(application.status, target)
        except TransitionError as exc:
            raise HTTPException(status_code=http_status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        previous_status = application.status
        application.status = target
        updates["status"] = target
        record_stage_event(
            db, application, from_status=previous_status, to_status=target, changed_by=changed_by
        )

    for field, value in updates.items():
        setattr(application, field, value)

    if updates:
        _commit_or_conflict(db, application_id)
        db.refresh(application)
    return application


@router.delete(
    "/{application_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete an application",
)
def delete_application(application_id: int = Path(gt=0), db: Session = Depends(get_db)) -> None:
    """Hard-delete an application.

    Withdrawing a candidate is a status change; this is for removing a record
    created in error, which is why it leaves no pipeline history behind.
    """
    application = _get_or_404(db, application_id)
    db.delete(application)
    db.commit()
