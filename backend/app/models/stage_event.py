"""Audit trail of every pipeline move an application makes.

There is no auth system yet, so ``changed_by`` is a free-text label supplied
by the caller rather than a foreign key to a user table -- the same trade-off
``Application.source`` already makes for "how did this arrive".
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import ApplicationStatus, enum_values

if TYPE_CHECKING:
    from app.models.application import Application


class ApplicationStageEvent(Base):
    """One row per status change: where it came from, where it went, when, and who.

    Rows are append-only and never edited or deleted except by the cascade
    when their application is deleted. ``from_status`` is null for the event
    that opens an application, since there is no prior stage to record.
    """

    __tablename__ = "application_stage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, index=True
    )

    from_status: Mapped[ApplicationStatus | None] = mapped_column(
        SAEnum(
            ApplicationStatus,
            name="application_status",
            native_enum=True,
            values_callable=enum_values,
        ),
    )
    to_status: Mapped[ApplicationStatus] = mapped_column(
        SAEnum(
            ApplicationStatus,
            name="application_status",
            native_enum=True,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    changed_by: Mapped[str | None] = mapped_column(String(120))
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    application: Mapped[Application] = relationship(back_populates="stage_events")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        frm = self.from_status.value if self.from_status else "none"
        return (
            f"<ApplicationStageEvent id={self.id} application_id={self.application_id} "
            f"{frm}->{self.to_status.value}>"
        )
