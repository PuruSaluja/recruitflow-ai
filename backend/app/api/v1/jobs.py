"""Job requisition endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi import status as http_status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    WRITE_LOCK,
    Pagination,
    commit_row,
    delete_with_applications,
    pagination,
    refresh_row,
)
from app.db.session import get_db
from app.models.enums import EmploymentType, JobStatus, SeniorityLevel
from app.models.job import Job
from app.schemas.common import MAX_ID, Page, contains_pattern
from app.schemas.job import JobCreate, JobRead, JobUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _get_or_404(db: Session, job_id: int, *, lock: bool | dict[str, bool] = False) -> Job:
    job = db.get(Job, job_id, with_for_update=lock)
    if job is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )
    return job


def _apply_filters(
    stmt: Select,
    *,
    status: JobStatus | None,
    seniority: SeniorityLevel | None,
    employment_type: EmploymentType | None,
    department: str | None,
    q: str | None,
) -> Select:
    """Apply the same predicates to both the page query and the count query.

    Sharing this is what keeps `total` honest: a filtered page whose total was
    computed unfiltered would silently break pagination in the UI.
    """
    if status is not None:
        stmt = stmt.where(Job.status == status)
    if seniority is not None:
        stmt = stmt.where(Job.seniority == seniority)
    if employment_type is not None:
        stmt = stmt.where(Job.employment_type == employment_type)
    if department is not None:
        stmt = stmt.where(Job.department == department)
    if q:
        stmt = stmt.where(Job.title.ilike(contains_pattern(q), escape="\\"))
    return stmt


@router.get("", response_model=Page[JobRead], summary="List job requisitions")
def list_jobs(
    db: Session = Depends(get_db),
    page: Pagination = Depends(pagination),
    status: JobStatus | None = Query(None, description="Filter by requisition status"),
    seniority: SeniorityLevel | None = Query(None, description="Filter by seniority level"),
    employment_type: EmploymentType | None = Query(None, description="Filter by employment type"),
    department: str | None = Query(None, description="Exact department match"),
    q: str | None = Query(None, description="Case-insensitive substring match on title"),
) -> Page[JobRead]:
    filters = {
        "status": status,
        "seniority": seniority,
        "employment_type": employment_type,
        "department": department,
        "q": q,
    }

    total = db.execute(
        _apply_filters(select(func.count()).select_from(Job), **filters)
    ).scalar_one()

    stmt = _apply_filters(select(Job), **filters)
    stmt = stmt.order_by(Job.created_at.desc(), Job.id.desc()).limit(page.limit).offset(page.offset)
    jobs = db.execute(stmt).scalars().all()

    return Page[JobRead](
        items=[JobRead.model_validate(job) for job in jobs],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "",
    response_model=JobRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a job requisition",
)
def create_job(payload: JobCreate, db: Session = Depends(get_db)) -> Job:
    job = Job(**payload.model_dump())
    db.add(job)
    db.commit()
    refresh_row(db, job, Job, job.id)
    return job


@router.get("/{job_id}", response_model=JobRead, summary="Fetch one job requisition")
def get_job(job_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)) -> Job:
    return _get_or_404(db, job_id)


@router.patch("/{job_id}", response_model=JobRead, summary="Partially update a job")
def update_job(
    payload: JobUpdate, job_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)
) -> Job:
    job = _get_or_404(db, job_id, lock=WRITE_LOCK)

    # exclude_unset distinguishes "field omitted" from "field set to null", so
    # a PATCH only touches what the caller actually sent.
    updates = payload.model_dump(exclude_unset=True)

    # JobUpdate can only compare the band when both ends arrive together.
    # Raising just the minimum can still invert the band stored on the row, and
    # the check constraint would then report it as a misleading 409 conflict.
    salary_min = updates.get("salary_min", job.salary_min)
    salary_max = updates.get("salary_max", job.salary_max)
    if salary_min is not None and salary_max is not None and salary_max < salary_min:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="salary_max must be greater than or equal to salary_min",
        )

    for field, value in updates.items():
        setattr(job, field, value)

    if updates:
        commit_row(db, Job, job_id)
        refresh_row(db, job, Job, job_id)
    return job


@router.post("/{job_id}/close", response_model=JobRead, summary="Close a job requisition")
def close_job(job_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)) -> Job:
    """Close a requisition. Idempotent: closing a closed job is a no-op."""
    job = _get_or_404(db, job_id)
    if job.status is not JobStatus.CLOSED:
        job.status = JobStatus.CLOSED
        commit_row(db, Job, job_id)
        refresh_row(db, job, Job, job_id)
    return job


@router.delete(
    "/{job_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a job requisition",
)
def delete_job(job_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)) -> None:
    """Hard-delete a job; applications and cached match scores cascade.

    See :func:`app.api.deps.delete_with_applications` for what a race with
    another write to one of its applications answers.
    """
    delete_with_applications(db, Job, job_id, lambda: _get_or_404(db, job_id, lock=WRITE_LOCK))
