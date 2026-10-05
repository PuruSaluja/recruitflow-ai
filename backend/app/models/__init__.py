"""ORM models.

Importing this package registers every model on ``Base.metadata``, which is
what Alembic's autogenerate and the test fixtures rely on.
"""

from app.models.application import Application
from app.models.candidate import Candidate
from app.models.embedding import CandidateEmbedding, JobEmbedding
from app.models.enums import (
    ApplicationStatus,
    EducationLevel,
    EmploymentType,
    JobStatus,
    SeniorityLevel,
)
from app.models.job import Job
from app.models.match import MatchScore
from app.models.stage_event import ApplicationStageEvent

__all__ = [
    "Application",
    "ApplicationStageEvent",
    "ApplicationStatus",
    "Candidate",
    "CandidateEmbedding",
    "EducationLevel",
    "EmploymentType",
    "Job",
    "JobEmbedding",
    "JobStatus",
    "MatchScore",
    "SeniorityLevel",
]
