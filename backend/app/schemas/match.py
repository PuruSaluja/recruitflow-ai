"""Request/response schemas for the matching layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.candidate import CandidateSummary


class MatchFeatures(BaseModel):
    """Per-feature breakdown behind a score, so a ranking can be explained."""

    text_similarity: float = Field(description="Cosine similarity of TF-IDF vectors")
    required_skill_coverage: float = Field(description="Fraction of required skills matched")
    preferred_skill_coverage: float = Field(description="Fraction of preferred skills matched")
    experience_gap: float = Field(description="Candidate years minus job minimum")
    seniority_distance: float = Field(description="Signed distance in seniority ranks")
    education_gap: float = Field(description="Signed distance in education ranks")


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: int
    candidate_id: int
    score: float = Field(ge=0.0, le=1.0)
    rank: int | None
    features: dict[str, float]
    model_version: str
    scored_at: datetime


class RankedCandidate(BaseModel):
    """A scored candidate as returned by the ranked-shortlist endpoint."""

    rank: int
    score: float = Field(ge=0.0, le=1.0)
    candidate: CandidateSummary
    features: dict[str, float]


class ScoreJobRequest(BaseModel):
    """Trigger scoring of a candidate pool against one job."""

    candidate_ids: list[int] | None = Field(
        default=None,
        description="Restrict scoring to these candidates; omit to score the whole pool.",
    )
    limit: int = Field(default=50, ge=1, le=500, description="Rows to return after ranking")


class ScoreJobResponse(BaseModel):
    job_id: int
    model_version: str
    scored_count: int = Field(description="Candidates scored in this run")
    elapsed_ms: float
    results: list[RankedCandidate]
