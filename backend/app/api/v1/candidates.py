"""Candidate endpoints, including resume file ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from fastapi import status as http_status
from fastapi.responses import PlainTextResponse
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.api.deps import DELETE_LOCK, Pagination, commit_row, pagination, refresh_row
from app.db.session import get_db
from app.models.candidate import Candidate
from app.models.enums import EducationLevel, SeniorityLevel
from app.schemas.candidate import CandidateCreate, CandidateRead, CandidateUpdate, ResumePreview
from app.schemas.common import MAX_ID, Page, contains_pattern
from app.services.resume import MAX_UPLOAD_BYTES, ResumeIngestError, extract_text

router = APIRouter(prefix="/candidates", tags=["candidates"])


def _get_or_404(db: Session, candidate_id: int, *, lock: bool = False) -> Candidate:
    candidate = db.get(Candidate, candidate_id, with_for_update=lock)
    if candidate is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Candidate {candidate_id} not found",
        )
    return candidate


def _has_skill(skill: str) -> object:
    """Case-insensitive membership test against the JSONB ``skills`` array.

    JSONB containment (``@>``) would be index-friendly but is case- and
    whitespace-exact, and skills arrive from resumes as "Python", "python" and
    "PYTHON" alike. Unnesting the array and comparing lowercased is correct at
    the pool sizes this endpoint pages over; if the candidate table outgrows a
    sequential scan the fix is a normalised skills table, not a stricter filter.
    """
    element = func.jsonb_array_elements_text(Candidate.skills).table_valued("value")
    return (
        select(1)
        .select_from(element)
        .where(func.lower(element.c.value) == skill.strip().lower())
        .exists()
    )


def _apply_filters(
    stmt: Select,
    *,
    seniority: SeniorityLevel | None,
    education_level: EducationLevel | None,
    min_years_experience: int | None,
    max_years_experience: int | None,
    location: str | None,
    skill: list[str] | None,
    q: str | None,
) -> Select:
    """Apply the same predicates to both the page query and the count query.

    Shared so ``total`` stays honest: a filtered page whose total was computed
    unfiltered would silently break pagination in the UI.
    """
    if seniority is not None:
        stmt = stmt.where(Candidate.seniority == seniority)
    if education_level is not None:
        stmt = stmt.where(Candidate.education_level == education_level)
    if min_years_experience is not None:
        stmt = stmt.where(Candidate.years_experience >= min_years_experience)
    if max_years_experience is not None:
        stmt = stmt.where(Candidate.years_experience <= max_years_experience)
    if location:
        stmt = stmt.where(Candidate.location.ilike(contains_pattern(location), escape="\\"))
    for wanted in skill or []:
        if wanted.strip():
            # Repeated ?skill= narrows conjunctively: every listed skill must
            # be present, which is how a recruiter reads a "must have" list.
            stmt = stmt.where(_has_skill(wanted))
    if q:
        pattern = contains_pattern(q)
        stmt = stmt.where(
            Candidate.full_name.ilike(pattern, escape="\\")
            | Candidate.email.ilike(pattern, escape="\\")
            | Candidate.current_title.ilike(pattern, escape="\\")
        )
    return stmt


@router.get("", response_model=Page[CandidateRead], summary="List candidates")
def list_candidates(
    db: Session = Depends(get_db),
    page: Pagination = Depends(pagination),
    seniority: SeniorityLevel | None = Query(None, description="Filter by seniority level"),
    education_level: EducationLevel | None = Query(None, description="Filter by education level"),
    min_years_experience: int | None = Query(None, ge=0, le=60),
    max_years_experience: int | None = Query(None, ge=0, le=60),
    location: str | None = Query(None, description="Case-insensitive substring match"),
    skill: list[str] | None = Query(
        None, description="Repeatable; a candidate must have every skill listed"
    ),
    q: str | None = Query(None, description="Substring match on name, email or current title"),
) -> Page[CandidateRead]:
    filters = {
        "seniority": seniority,
        "education_level": education_level,
        "min_years_experience": min_years_experience,
        "max_years_experience": max_years_experience,
        "location": location,
        "skill": skill,
        "q": q,
    }

    total = db.execute(
        _apply_filters(select(func.count()).select_from(Candidate), **filters)
    ).scalar_one()

    stmt = _apply_filters(select(Candidate), **filters)
    stmt = (
        stmt.order_by(Candidate.created_at.desc(), Candidate.id.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    candidates = db.execute(stmt).scalars().all()

    return Page[CandidateRead](
        items=[CandidateRead.model_validate(c) for c in candidates],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post(
    "",
    response_model=CandidateRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a candidate",
)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)) -> Candidate:
    """Create a candidate. A duplicate email is a 409, not a 500."""
    candidate = Candidate(**payload.model_dump())
    db.add(candidate)
    db.commit()
    refresh_row(db, candidate, Candidate, candidate.id)
    return candidate


@router.post(
    "/resume-preview",
    response_model=ResumePreview,
    summary="Parse a resume file without storing anything",
)
async def preview_resume(file: UploadFile = File(...)) -> ResumePreview:
    """Extract text from an upload so a client can show it before committing.

    Declared before ``/{candidate_id}`` so the literal path wins the match.
    """
    text = await _read_resume(file)
    return ResumePreview(
        filename=file.filename,
        characters=len(text),
        words=len(text.split()),
        text=text,
    )


@router.get("/{candidate_id}", response_model=CandidateRead, summary="Fetch one candidate")
def get_candidate(
    candidate_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)
) -> Candidate:
    return _get_or_404(db, candidate_id)


@router.patch("/{candidate_id}", response_model=CandidateRead, summary="Partially update")
def update_candidate(
    payload: CandidateUpdate,
    candidate_id: int = Path(gt=0, le=MAX_ID),
    db: Session = Depends(get_db),
) -> Candidate:
    candidate = _get_or_404(db, candidate_id)

    # exclude_unset distinguishes "field omitted" from "field set to null", so
    # a PATCH only touches what the caller actually sent.
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(candidate, field, value)

    if updates:
        commit_row(db, Candidate, candidate_id)
        refresh_row(db, candidate, Candidate, candidate_id)
    return candidate


@router.delete(
    "/{candidate_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a candidate",
)
def delete_candidate(
    candidate_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)
) -> None:
    """Hard-delete a candidate; applications and cached match scores cascade."""
    candidate = _get_or_404(db, candidate_id, lock=DELETE_LOCK)
    db.delete(candidate)
    commit_row(db, Candidate, candidate_id)


@router.get(
    "/{candidate_id}/resume",
    response_class=PlainTextResponse,
    summary="Fetch the stored resume text",
)
def get_resume(candidate_id: int = Path(gt=0, le=MAX_ID), db: Session = Depends(get_db)) -> str:
    return _get_or_404(db, candidate_id).resume_text


@router.put(
    "/{candidate_id}/resume",
    response_model=CandidateRead,
    summary="Upload a resume file, replacing the stored text",
)
async def upload_resume(
    candidate_id: int = Path(gt=0, le=MAX_ID),
    file: UploadFile = File(..., description="PDF, .txt or .md, up to 2 MB"),
    db: Session = Depends(get_db),
) -> Candidate:
    """Ingest a PDF or plain-text resume into ``resume_text``.

    PUT rather than POST: the stored text is replaced wholesale, so re-uploading
    the same file twice leaves the candidate in the same state.
    """
    candidate = _get_or_404(db, candidate_id)
    candidate.resume_text = await _read_resume(file)
    commit_row(db, Candidate, candidate_id)
    refresh_row(db, candidate, Candidate, candidate_id)
    return candidate


async def _read_resume(file: UploadFile) -> str:
    """Read an upload and extract text, mapping ingest failures to 400."""
    # UploadFile is unbounded, so cap the read rather than trusting the
    # advertised size: content-length is client-supplied.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    try:
        return extract_text(file.filename, data)
    except ResumeIngestError as exc:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
