"""Job requisition model."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Integer, Numeric, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import EducationLevel, EmploymentType, JobStatus, SeniorityLevel, enum_values

if TYPE_CHECKING:
    from app.models.application import Application
    from app.models.match import MatchScore


class Job(Base, TimestampMixin):
    """An open (or closed) requisition that candidates are scored against."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("min_years_experience >= 0", name="ck_jobs_min_years_nonneg"),
        CheckConstraint(
            "salary_max IS NULL OR salary_min IS NULL OR salary_max >= salary_min",
            name="ck_jobs_salary_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    department: Mapped[str | None] = mapped_column(String(100))
    location: Mapped[str | None] = mapped_column(String(150))

    employment_type: Mapped[EmploymentType] = mapped_column(
        SAEnum(
            EmploymentType,
            name="employment_type",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=EmploymentType.FULL_TIME,
    )
    seniority: Mapped[SeniorityLevel] = mapped_column(
        SAEnum(
            SeniorityLevel,
            name="seniority_level",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=SeniorityLevel.MID,
    )
    education_required: Mapped[EducationLevel] = mapped_column(
        SAEnum(
            EducationLevel,
            name="education_level",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=EducationLevel.BACHELOR,
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(
            JobStatus,
            name="job_status",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=JobStatus.OPEN,
        index=True,
    )

    min_years_experience: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Skill lists are stored as JSONB so they can be queried with containment
    # operators without a join table; the taxonomy is small and read-mostly.
    # none_as_null so a Python None becomes SQL NULL and trips the NOT NULL
    # constraint. Without it JSONB stores the JSON scalar `null`, which the
    # constraint accepts and which then fails to deserialise on every read.
    required_skills: Mapped[list[str]] = mapped_column(
        JSONB(none_as_null=True), nullable=False, default=list
    )
    preferred_skills: Mapped[list[str]] = mapped_column(
        JSONB(none_as_null=True), nullable=False, default=list
    )

    salary_min: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    salary_max: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))

    applications: Mapped[list[Application]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    match_scores: Mapped[list[MatchScore]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def matching_text(self) -> str:
        """Free-text blob handed to the vectoriser."""
        skills = " ".join(self.required_skills or []) + " " + " ".join(self.preferred_skills or [])
        return f"{self.title}\n{self.description}\n{skills}".strip()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Job id={self.id} title={self.title!r} status={self.status.value}>"
