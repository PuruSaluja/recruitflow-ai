"""Shared enumerations, mapped to native PostgreSQL enum types."""

from __future__ import annotations

import enum


class EmploymentType(str, enum.Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"


class SeniorityLevel(str, enum.Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"

    @property
    def rank(self) -> int:
        """Ordinal position, used by the matcher to measure seniority distance."""
        return _SENIORITY_ORDER[self]


class EducationLevel(str, enum.Enum):
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"

    @property
    def rank(self) -> int:
        """Ordinal position, so 'meets or exceeds' comparisons are cheap."""
        return _EDUCATION_ORDER[self]


class JobStatus(str, enum.Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


class ApplicationStatus(str, enum.Enum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


_SENIORITY_ORDER: dict[SeniorityLevel, int] = {
    SeniorityLevel.INTERN: 0,
    SeniorityLevel.JUNIOR: 1,
    SeniorityLevel.MID: 2,
    SeniorityLevel.SENIOR: 3,
    SeniorityLevel.STAFF: 4,
    SeniorityLevel.PRINCIPAL: 5,
}

_EDUCATION_ORDER: dict[EducationLevel, int] = {
    EducationLevel.HIGH_SCHOOL: 0,
    EducationLevel.ASSOCIATE: 1,
    EducationLevel.BACHELOR: 2,
    EducationLevel.MASTER: 3,
    EducationLevel.DOCTORATE: 4,
}
