"""Request/response schemas for candidates."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import EducationLevel, SeniorityLevel


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
    skills: list[str] = Field(default_factory=list)


class CandidateCreate(CandidateBase):
    pass


class CandidateUpdate(BaseModel):
    """Partial update; every field is optional."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(default=None, min_length=2, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=150)
    current_title: str | None = Field(default=None, max_length=200)
    years_experience: int | None = Field(default=None, ge=0, le=60)
    seniority: SeniorityLevel | None = None
    education_level: EducationLevel | None = None
    resume_text: str | None = Field(default=None, min_length=20)
    skills: list[str] | None = None


class CandidateRead(CandidateBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


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
