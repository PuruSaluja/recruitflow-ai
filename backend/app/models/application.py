"""Application model: a candidate applying to a specific job."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import ApplicationStatus, enum_values

if TYPE_CHECKING:
    from app.models.candidate import Candidate
    from app.models.job import Job
    from app.models.stage_event import ApplicationStageEvent


class Application(Base, TimestampMixin):
    """Join of candidate -> job, carrying pipeline state."""

    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", name="uq_applications_job_candidate"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[ApplicationStatus] = mapped_column(
        SAEnum(
            ApplicationStatus,
            name="application_status",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=ApplicationStatus.APPLIED,
        index=True,
    )
    source: Mapped[str | None] = mapped_column(String(80))
    notes: Mapped[str | None] = mapped_column(Text)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Bumped by SQLAlchemy on every UPDATE, which then carries
    # "WHERE version = <the value we read>". A write that raced another one
    # matches no row and raises StaleDataError instead of silently overwriting
    # it. A database constraint cannot do this job: each of two simultaneous
    # stage changes is legal on its own, and it is only the pair that loses a
    # recruiter's action and leaves two stage events leaving the same stage.
    version: Mapped[int] = mapped_column(nullable=False, server_default="1")
    __mapper_args__ = {"version_id_col": version}

    job: Mapped[Job] = relationship(back_populates="applications")
    candidate: Mapped[Candidate] = relationship(back_populates="applications")
    # Deleted with the application by the foreign key's ON DELETE CASCADE, not
    # loaded first: when another request had deleted the application, the
    # ORM's own DELETE of its events matched no row and SQLAlchemy warned.
    stage_events: Mapped[list[ApplicationStageEvent]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationStageEvent.occurred_at",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<Application id={self.id} job_id={self.job_id} "
            f"candidate_id={self.candidate_id} status={self.status.value}>"
        )
