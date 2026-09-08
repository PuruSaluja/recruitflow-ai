"""ORM models.

Importing this package registers every model on ``Base.metadata``, which is
what Alembic's autogenerate and the test fixtures rely on.
"""

from app.models.application import Application
from app.models.candidate import Candidate
from app.models.enums import (
    ApplicationStatus,
    EducationLevel,
    EmploymentType,
    JobStatus,
    SeniorityLevel,
)
from app.models.job import Job
from app.models.match import MatchScore

__all__ = [
    "Application",
    "ApplicationStatus",
    "Candidate",
    "EducationLevel",
    "EmploymentType",
    "Job",
    "JobStatus",
    "MatchScore",
    "SeniorityLevel",
]
