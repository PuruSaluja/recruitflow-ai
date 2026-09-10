"""Persisted output of the ML matching layer."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate import Candidate
    from app.models.job import Job


class MatchScore(Base):
    """A single (job, candidate) score produced by a specific model version.

    Scores are cached rather than recomputed per request: scoring a full
    candidate pool is a batch operation, and recruiters page through the same
    ranked list repeatedly.
    """

    __tablename__ = "match_scores"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "candidate_id", "model_version", name="uq_match_job_cand_version"
        ),
        CheckConstraint("score >= 0.0 AND score <= 1.0", name="ck_match_score_range"),
        # Recruiters always read this table as "top N for a job", so the
        # covering index is ordered descending by score.
        Index("ix_match_job_score_desc", "job_id", text("score DESC")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Calibrated probability that this candidate is a strong match, in [0, 1].
    score: Mapped[float] = mapped_column(Float, nullable=False)
    #: 1-based position within this job's ranked candidate list.
    rank: Mapped[int | None] = mapped_column(Integer)

    #: Per-feature contributions, kept so the UI and PDF report can explain
    #: *why* a candidate ranked where they did.
    features: Mapped[dict] = mapped_column(
        JSONB(none_as_null=True), nullable=False, default=dict
    )
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    job: Mapped[Job] = relationship(back_populates="match_scores")
    candidate: Mapped[Candidate] = relationship(back_populates="match_scores")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<MatchScore job_id={self.job_id} candidate_id={self.candidate_id} "
            f"score={self.score:.4f} rank={self.rank}>"
        )
