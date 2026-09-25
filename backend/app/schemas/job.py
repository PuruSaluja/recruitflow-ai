"""Request/response schemas for job requisitions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import EducationLevel, EmploymentType, JobStatus, SeniorityLevel
from app.schemas.candidate import MAX_SKILL_CHARS
from app.schemas.common import reject_null_fields

#: The longest job description accepted: as long as a normal job description
#: would or should be. A typical one runs to a few
#: thousand characters and a long one to about 10,000 (rough figures, as for
#: the resume cap); 20,000, some 3,000 words, leaves room for the long ones
#: without taking in whole pasted documents. The generated corpus's
#: descriptions run 327 to 467 characters (seeds 42 and 5).
MAX_DESCRIPTION_CHARS = 20_000

#: A job's skills take a candidate's caps: at most 100 in each list, each at
#: most ``MAX_SKILL_CHARS`` characters. Jobs store their skills as sent, where
#: a candidate's are tidied first, so the length counted here is the stored one.
MAX_SKILLS_PER_LIST = 100
SkillName = Annotated[str, Field(max_length=MAX_SKILL_CHARS)]

#: Mirrors the salary columns' ``NUMERIC(10, 2)``: at most 99,999,999.99, in
#: whole cents. A larger value, or one that rounds up past it, made PostgreSQL
#: raise "numeric field overflow", reported as an outage, and a fraction of a
#: cent was rounded away without a word.
SALARY_DIGITS = 10
SALARY_PLACES = 2

#: Columns the database declares NOT NULL, so a PATCH may omit them but must
#: never send an explicit null.
REQUIRED_JOB_FIELDS = frozenset(
    {
        "title",
        "employment_type",
        "seniority",
        "education_required",
        "min_years_experience",
        "description",
        "required_skills",
        "preferred_skills",
        "status",
    }
)


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
    salary_min: Decimal | None = Field(
        default=None, ge=0, max_digits=SALARY_DIGITS, decimal_places=SALARY_PLACES
    )
    salary_max: Decimal | None = Field(
        default=None, ge=0, max_digits=SALARY_DIGITS, decimal_places=SALARY_PLACES
    )

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
    # The caps sit on the request models, not on ``JobBase``, because
    # ``JobRead`` inherits the base: a row stored before the caps existed must
    # still be readable rather than failing every response.
    description: str = Field(min_length=20, max_length=MAX_DESCRIPTION_CHARS)
    required_skills: list[SkillName] = Field(default_factory=list, max_length=MAX_SKILLS_PER_LIST)
    preferred_skills: list[SkillName] = Field(default_factory=list, max_length=MAX_SKILLS_PER_LIST)
    status: JobStatus = JobStatus.OPEN


class JobUpdate(BaseModel):
    """Partial update; every field is optional."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_required_nulls(cls, data: Any) -> Any:
        return reject_null_fields(data, REQUIRED_JOB_FIELDS)

    title: str | None = Field(default=None, min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=150)
    employment_type: EmploymentType | None = None
    seniority: SeniorityLevel | None = None
    education_required: EducationLevel | None = None
    min_years_experience: int | None = Field(default=None, ge=0, le=50)
    description: str | None = Field(default=None, min_length=20, max_length=MAX_DESCRIPTION_CHARS)
    required_skills: list[SkillName] | None = Field(default=None, max_length=MAX_SKILLS_PER_LIST)
    preferred_skills: list[SkillName] | None = Field(default=None, max_length=MAX_SKILLS_PER_LIST)
    salary_min: Decimal | None = Field(
        default=None, ge=0, max_digits=SALARY_DIGITS, decimal_places=SALARY_PLACES
    )
    salary_max: Decimal | None = Field(
        default=None, ge=0, max_digits=SALARY_DIGITS, decimal_places=SALARY_PLACES
    )
    status: JobStatus | None = None


class JobRead(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: JobStatus
    created_at: datetime
    updated_at: datetime
