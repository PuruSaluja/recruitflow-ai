"""Job requisition endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, pagination
from app.db.session import get_db
from app.models.enums import EmploymentType, JobStatus, SeniorityLevel
from app.models.job import Job
from app.schemas.common import Page
from app.schemas.job import JobCreate, JobRead, JobUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _get_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
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
        stmt = stmt.where(Job.title.ilike(f"%{q}%"))
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
    db.refresh(job)
    return job


@router.get("/{job_id}", response_model=JobRead, summary="Fetch one job requisition")
def get_job(job_id: int, db: Session = Depends(get_db)) -> Job:
    return _get_or_404(db, job_id)


@router.patch("/{job_id}", response_model=JobRead, summary="Partially update a job")
def update_job(job_id: int, payload: JobUpdate, db: Session = Depends(get_db)) -> Job:
    job = _get_or_404(db, job_id)

    # exclude_unset distinguishes "field omitted" from "field set to null", so
    # a PATCH only touches what the caller actually sent.
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(job, field, value)

    if updates:
        db.commit()
        db.refresh(job)
    return job


@router.post("/{job_id}/close", response_model=JobRead, summary="Close a job requisition")
def close_job(job_id: int, db: Session = Depends(get_db)) -> Job:
    """Close a requisition. Idempotent: closing a closed job is a no-op."""
    job = _get_or_404(db, job_id)
    if job.status is not JobStatus.CLOSED:
        job.status = JobStatus.CLOSED
        db.commit()
        db.refresh(job)
    return job


@router.delete(
    "/{job_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a job requisition",
)
def delete_job(job_id: int, db: Session = Depends(get_db)) -> None:
    """Hard-delete a job; applications and cached match scores cascade."""
    job = _get_or_404(db, job_id)
    db.delete(job)
    db.commit()
