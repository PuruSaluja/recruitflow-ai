"""Request/response schemas for applications."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ApplicationStatus


class ApplicationCreate(BaseModel):
    job_id: int
    candidate_id: int
    source: str | None = Field(default=None, max_length=80)
    notes: str | None = None


class ApplicationUpdate(BaseModel):
    """Partial update; used mainly to advance pipeline status."""

    model_config = ConfigDict(extra="forbid")

    status: ApplicationStatus | None = None
    source: str | None = Field(default=None, max_length=80)
    notes: str | None = None


class ApplicationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    candidate_id: int
    status: ApplicationStatus
    source: str | None
    notes: str | None
    applied_at: datetime
    created_at: datetime
    updated_at: datetime
