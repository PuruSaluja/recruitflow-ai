"""Request/response schemas for the matching endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.ml.features import YEARS_SCALE
from app.schemas.candidate import CandidateSummary
from app.schemas.common import MAX_ID, Page

#: Most candidate ids one scoring request may name. Leaving the list out scores
#: every candidate on file, so this only bounds an explicit selection -- and
#: keeps its IN clause far below PostgreSQL's limit on bind parameters.
MAX_CANDIDATE_IDS = 1000


class MatchFeatures(BaseModel):
    """The engineered features behind a score, as ``match_scores.features`` stores them.

    Exactly :data:`app.ml.features.FEATURE_NAMES`, in that order. The TF-IDF
    half of the matcher's input is not broken out per score: the calibrated
    matcher keeps one fitted vocabulary per calibration fold, so there is no
    single text similarity to report back.
    """

    required_skill_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of the job's required skills the candidate lists; 1.0 if the job requires none"
        ),
    )
    preferred_skill_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of the job's preferred skills the candidate lists; 1.0 if the job prefers none"
        ),
    )
    skill_jaccard: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overlap between the job's skills, required and preferred, and the candidate's, "
            "over their union; 0.0 when both are empty"
        ),
    )
    experience_gap: float = Field(
        ge=0.0,
        le=1.0,
        description=f"Years short of the job's minimum, divided by {YEARS_SCALE:g}, capped at 1.0",
    )
    experience_surplus: float = Field(
        ge=0.0,
        le=1.0,
        description=f"Years beyond the job's minimum, divided by {YEARS_SCALE:g}, capped at 1.0",
    )
    seniority_distance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Seniority levels between candidate and job, either direction, as a share of "
            "the widest possible gap"
        ),
    )
    seniority_direction: float = Field(
        ge=-1.0,
        le=1.0,
        description="seniority_distance with a sign: negative under-levelled, positive over",
    )
    education_fit: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 when the candidate meets or exceeds the education required, down to 0.0",
    )


class RankedCandidate(BaseModel):
    """One candidate's place in a job's ranking, and what the score was built from."""

    rank: int = Field(ge=1, description="1-based position, best first")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated probability that this candidate would be advanced, learned from "
            "past recruiter decisions reweighed so that gender and the decision are "
            "independent"
        ),
    )
    scored_at: datetime
    candidate: CandidateSummary
    features: MatchFeatures


class ScoreJobRequest(BaseModel):
    """Body for scoring a candidate pool against one job. Every field is optional."""

    # Without this a misspelt field is dropped silently, and a request meant
    # for three candidates scores everyone on file.
    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[Annotated[int, Field(gt=0, le=MAX_ID)]] | None = Field(
        default=None,
        max_length=MAX_CANDIDATE_IDS,
        description="Score only these candidates; omit to score every candidate on file",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Ranked rows to return; the whole pool is still scored and cached",
    )


class ScoreJobResponse(BaseModel):
    job_id: int
    model_version: str = Field(description="Content hash of the matcher that produced the scores")
    scored_count: int = Field(description="Distinct candidates scored and cached in this run")
    elapsed_ms: float = Field(
        description="Server time spent loading the matcher, scoring the pool and caching it"
    )
    results: list[RankedCandidate] = Field(
        description="The best `limit` of the pool, ranked within that pool"
    )


class MatchPage(Page[RankedCandidate]):
    """A page of one job's cached ranking under the current matcher, best first."""

    job_id: int
    model_version: str = Field(description="Content hash of the matcher whose scores are listed")
