"""Request/response schemas for candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.models.enums import EducationLevel, SeniorityLevel
from app.schemas.common import reject_null_fields

#: Columns the database declares NOT NULL, so a PATCH may omit them but must
#: never send an explicit null.
REQUIRED_CANDIDATE_FIELDS = frozenset(
    {
        "full_name",
        "email",
        "years_experience",
        "seniority",
        "education_level",
        "resume_text",
        "skills",
    }
)


def _tidy_skills(value: list[str] | None) -> list[str] | None:
    """Trim, drop blanks and de-duplicate case-insensitively.

    Skills arrive from resumes and forms as "Python", " python " and "PYTHON".
    Normalising on the way in keeps the JSONB arrays comparable and stops the
    same skill being counted twice by the matcher; the first spelling seen is
    kept so the UI still shows what the candidate wrote.
    """
    if value is None:
        return None
    seen: set[str] = set()
    tidied: list[str] = []
    for raw in value:
        skill = " ".join(raw.split())
        key = skill.casefold()
        if not skill or key in seen:
            continue
        seen.add(key)
        tidied.append(skill)
    return tidied


class CandidateBase(BaseModel):
    full_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=150)
    current_title: str | None = Field(default=None, max_length=200)
    years_experience: int = Field(default=0, ge=0, le=60)
    seniority: SeniorityLevel = SeniorityLevel.MID
    education_level: EducationLevel = EducationLevel.BACHELOR
    resume_text: str = Field(min_length=20)
    skills: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("skills")
    @classmethod
    def _normalise_skills(cls, value: list[str]) -> list[str]:
        return _tidy_skills(value) or []


class CandidateCreate(CandidateBase):
    pass


class CandidateUpdate(BaseModel):
    """Partial update; every field is optional."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_required_nulls(cls, data: Any) -> Any:
        return reject_null_fields(data, REQUIRED_CANDIDATE_FIELDS)

    full_name: str | None = Field(default=None, min_length=2, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=150)
    current_title: str | None = Field(default=None, max_length=200)
    years_experience: int | None = Field(default=None, ge=0, le=60)
    seniority: SeniorityLevel | None = None
    education_level: EducationLevel | None = None
    resume_text: str | None = Field(default=None, min_length=20)
    skills: list[str] | None = Field(default=None, max_length=100)

    @field_validator("skills")
    @classmethod
    def _normalise_skills(cls, value: list[str] | None) -> list[str] | None:
        return _tidy_skills(value)


class CandidateRead(CandidateBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class ResumePreview(BaseModel):
    """Result of parsing a resume upload without storing it."""

    filename: str | None = None
    characters: int = Field(description="Length of the extracted text")
    words: int = Field(description="Whitespace-delimited token count")
    text: str


class CandidateSummary(BaseModel):
    """Trimmed projection used inside match listings and PDF reports."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    current_title: str | None
    years_experience: int
    seniority: SeniorityLevel
    skills: list[str]
