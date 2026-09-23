"""Request/response schemas for candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.models.enums import EducationLevel, SeniorityLevel
from app.schemas.common import reject_null_fields
from app.services.resume import MAX_RESUME_CHARS

#: The longest single skill accepted, measured after tidying. The taxonomy's
#: longest is 27 characters; a 50,000-character "skill" used to be stored.
MAX_SKILL_CHARS = 100

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


def _check_skill_lengths(value: list[str] | None) -> list[str] | None:
    """Refuse any skill longer than ``MAX_SKILL_CHARS`` once tidied.

    Measured on the tidied spelling, so padding that tidying removes never
    counts, whichever order pydantic runs this and the tidying in. The message
    names the length rather than quoting the skill, which may be enormous.
    """
    for skill in value or []:
        length = len(" ".join(skill.split()))
        if length > MAX_SKILL_CHARS:
            raise ValueError(
                f"Each skill must be at most {MAX_SKILL_CHARS} characters; one is {length:,}"
            )
    return value


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
    # The length caps sit on the request models, not on ``CandidateBase``,
    # because ``CandidateRead`` inherits the base: a row stored before the caps
    # existed must still be readable rather than failing every response.
    resume_text: str = Field(min_length=20, max_length=MAX_RESUME_CHARS)

    @field_validator("skills")
    @classmethod
    def _cap_skill_length(cls, value: list[str]) -> list[str]:
        return _check_skill_lengths(value) or []


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
    resume_text: str | None = Field(default=None, min_length=20, max_length=MAX_RESUME_CHARS)
    skills: list[str] | None = Field(default=None, max_length=100)

    @field_validator("skills")
    @classmethod
    def _normalise_skills(cls, value: list[str] | None) -> list[str] | None:
        return _tidy_skills(_check_skill_lengths(value))


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
