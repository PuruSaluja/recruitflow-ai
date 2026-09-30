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
    # A job's delete and a candidate's both take the pair's score with them,
    # and whichever comes second finds it gone. That is the outcome both
    # wanted, so the row count of a DELETE here is not checked; SQLAlchemy
    # otherwise warns. Scores carry no version: nothing else is lost.
    __mapper_args__ = {"confirm_deleted_rows": False}

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Calibrated probability, in [0, 1], that a past recruiter would have
    #: advanced this candidate for this job -- the label the matcher learns.
    score: Mapped[float] = mapped_column(Float, nullable=False)
    #: 1-based position within the pool this row was last scored in, which
    #: may be a subset of the job's candidates.
    rank: Mapped[int | None] = mapped_column(Integer)

    #: Per-feature contributions, kept so the UI and PDF report can explain
    #: *why* a candidate ranked where they did.
    features: Mapped[dict] = mapped_column(JSONB(none_as_null=True), nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    #: When this row was last (re)computed. The default only covers a bare
    #: INSERT outside the ORM; the one real writer (``services.matching``)
    #: sets this explicitly on every row it touches, insert or update. An
    #: ``onupdate=func.now()`` here would look like it did that, but it only
    #: fires when some *other* column is also dirty -- a rescore that
    #: reproduces the same score, rank and features leaves nothing changed
    #: for SQLAlchemy to see, and the timestamp would silently go stale.
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
