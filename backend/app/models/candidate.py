"""Candidate / applicant model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import EducationLevel, SeniorityLevel, enum_values

if TYPE_CHECKING:
    from app.models.application import Application
    from app.models.match import MatchScore


class Candidate(Base, TimestampMixin):
    """A person in the talent pool, with the resume text used for matching."""

    __tablename__ = "candidates"
    __table_args__ = (CheckConstraint("years_experience >= 0", name="ck_candidates_years_nonneg"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(40))
    location: Mapped[str | None] = mapped_column(String(150))

    current_title: Mapped[str | None] = mapped_column(String(200))
    years_experience: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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
    education_level: Mapped[EducationLevel] = mapped_column(
        SAEnum(
            EducationLevel,
            name="education_level",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=EducationLevel.BACHELOR,
    )

    resume_text: Mapped[str] = mapped_column(Text, nullable=False)
    # none_as_null so a Python None becomes SQL NULL and trips the NOT NULL
    # constraint. Without it JSONB stores the JSON scalar `null`, which the
    # constraint accepts and which then fails to deserialise on every read.
    skills: Mapped[list[str]] = mapped_column(
        JSONB(none_as_null=True), nullable=False, default=list
    )

    applications: Mapped[list[Application]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    match_scores: Mapped[list[MatchScore]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Candidate id={self.id} email={self.email!r}>"
