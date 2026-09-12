"""Recording pipeline moves into the application_stage_events audit trail.

Kept separate from ``services.pipeline`` because that module is deliberately
database-free and unit-tested without one; this one only ever appends a row.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.application import Application
from app.models.enums import ApplicationStatus
from app.models.stage_event import ApplicationStageEvent


def record_stage_event(
    db: Session,
    application: Application,
    *,
    from_status: ApplicationStatus | None,
    to_status: ApplicationStatus,
    changed_by: str | None,
) -> ApplicationStageEvent:
    """Add (but do not commit) one audit row for a move that already happened.

    The caller commits alongside whatever else its transaction touches, so a
    move and its audit row always land or fail together.
    """
    event = ApplicationStageEvent(
        application_id=application.id,
        from_status=from_status,
        to_status=to_status,
        changed_by=changed_by,
    )
    db.add(event)
    return event
