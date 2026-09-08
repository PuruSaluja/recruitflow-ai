"""Request/response schemas for job requisitions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import EducationLevel, EmploymentType, JobStatus, SeniorityLevel


class JobBase(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=150)
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    seniority: SeniorityLevel = SeniorityLevel.MID
    education_required: EducationLevel = EducationLevel.BACHELOR
    min_years_experience: int = Field(default=0, ge=0, le=50)
    description: str = Field(min_length=20)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_salary_band(self) -> JobBase:
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_max < self.salary_min
        ):
            raise ValueError("salary_max must be greater than or equal to salary_min")
        return self


class JobCreate(JobBase):
    status: JobStatus = JobStatus.OPEN


class JobUpdate(BaseModel):
    """Partial update; every field is optional."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=150)
    employment_type: EmploymentType | None = None
    seniority: SeniorityLevel | None = None
    education_required: EducationLevel | None = None
    min_years_experience: int | None = Field(default=None, ge=0, le=50)
    description: str | None = Field(default=None, min_length=20)
    required_skills: list[str] | None = None
    preferred_skills: list[str] | None = None
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    status: JobStatus | None = None


class JobRead(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: JobStatus
    created_at: datetime
    updated_at: datetime
