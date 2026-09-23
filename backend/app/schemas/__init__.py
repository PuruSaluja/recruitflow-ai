"""Pydantic request/response schemas."""

from app.schemas.application import ApplicationCreate, ApplicationRead, ApplicationUpdate
from app.schemas.candidate import (
    CandidateCreate,
    CandidateRead,
    CandidateSummary,
    CandidateUpdate,
)
from app.schemas.common import ErrorDetail, Page
from app.schemas.job import JobCreate, JobRead, JobUpdate
from app.schemas.match import (
    MatchFeatures,
    MatchPage,
    RankedCandidate,
    ScoreJobRequest,
    ScoreJobResponse,
)

__all__ = [
    "ApplicationCreate",
    "ApplicationRead",
    "ApplicationUpdate",
    "CandidateCreate",
    "CandidateRead",
    "CandidateSummary",
    "CandidateUpdate",
    "ErrorDetail",
    "JobCreate",
    "JobRead",
    "JobUpdate",
    "MatchFeatures",
    "MatchPage",
    "Page",
    "RankedCandidate",
    "ScoreJobRequest",
    "ScoreJobResponse",
]
