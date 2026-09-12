"""Request/response schemas for applications."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import ApplicationStatus
from app.schemas.common import reject_null_fields

#: Columns the database declares NOT NULL, so a PATCH may omit them but must
#: never send an explicit null. ``source`` and ``notes`` are nullable and can
#: legitimately be cleared.
REQUIRED_APPLICATION_FIELDS = frozenset({"status"})


def _tidy_changed_by(value: str | None) -> str | None:
    """Collapse whitespace and treat a blank label as absent.

    An audit row crediting three spaces says nothing, and it reads as a real
    author rather than a missing one. Same treatment the candidate schemas
    give skill lists, for the same reason: this is free text typed by people.
    """
    if value is None:
        return None
    return " ".join(value.split()) or None


class ApplicationCreate(BaseModel):
    # gt=0 so a malformed id is a 422 about the payload rather than a 404
    # claiming the job does not exist.
    job_id: int = Field(gt=0)
    candidate_id: int = Field(gt=0)
    source: str | None = Field(default=None, max_length=80)
    notes: str | None = None
    #: Who opened this application, recorded on the opening stage event only.
    #: Not a column on the application itself.
    changed_by: str | None = Field(default=None, max_length=120)

    @field_validator("changed_by")
    @classmethod
    def _normalise_changed_by(cls, value: str | None) -> str | None:
        return _tidy_changed_by(value)


class ApplicationUpdate(BaseModel):
    """Partial update; used mainly to advance pipeline status."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_required_nulls(cls, data: Any) -> Any:
        return reject_null_fields(data, REQUIRED_APPLICATION_FIELDS)

    status: ApplicationStatus | None = None
    source: str | None = Field(default=None, max_length=80)
    notes: str | None = None
    #: Who made this change. Only recorded when it accompanies an actual
    #: status move; otherwise there is no stage event to attach it to.
    changed_by: str | None = Field(default=None, max_length=120)

    @field_validator("changed_by")
    @classmethod
    def _normalise_changed_by(cls, value: str | None) -> str | None:
        return _tidy_changed_by(value)


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


class ApplicationStageEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    application_id: int
    from_status: ApplicationStatus | None
    to_status: ApplicationStatus
    changed_by: str | None
    occurred_at: datetime
