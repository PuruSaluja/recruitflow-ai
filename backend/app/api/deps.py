"""Shared FastAPI dependencies, and the helpers write routes commit and read rows back with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Query
from fastapi import status as http_status
from sqlalchemy import select
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from app.schemas.common import MAX_OFFSET

#: Row locks, as ``Session.get``'s ``with_for_update``. ``WRITE_LOCK`` is for
#: a write whose answer depends on the row as it stands when the write lands: a
#: job's PATCH checks the salary band against the stored ends, and a DELETE
#: answers 404 once the row is gone. Taken when the row is read and held to the
#: commit, so a second request writing the same row waits, then sees what the
#: first left and answers as it would have one after the other.
#:
#: It is FOR NO KEY UPDATE, the lock an UPDATE takes anyway, for deletes too.
#: A DELETE statement takes FOR UPDATE, but only once the rows referencing
#: this one are gone; taken at the read, FOR UPDATE also blocks every new row
#: referencing it, whose foreign key check takes FOR KEY SHARE, while the
#: delete waits on child rows that new row's transaction may hold. A job delete
#: and a scoring run that had rescored one of the job's cached rows deadlocked
#: that way, and the loser answered 503.
#:
#: ``REFERENCE_LOCK`` is FOR KEY SHARE, that foreign key check's own lock,
#: taken on the job and candidate a new application names when they are
#: checked: a DELETE of either then waits for the application, or the
#: application waits for the DELETE and finds the row gone, while edits of
#: either go ahead.
WRITE_LOCK: dict[str, bool] = {"key_share": True}
REFERENCE_LOCK: dict[str, bool] = {"read": True, "key_share": True}


@dataclass(frozen=True)
class Pagination:
    """Offset pagination, shared by every collection endpoint."""

    limit: int
    offset: int


def pagination(
    limit: int = Query(50, ge=1, le=200, description="Maximum rows to return"),
    offset: int = Query(0, ge=0, le=MAX_OFFSET, description="Rows to skip"),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


def _row_exists(db: Session, model: Any, row_id: int) -> bool:
    return db.scalar(select(model.id).where(model.id == row_id)) is not None


def commit_row(db: Session, model: Any, row_id: int) -> None:
    """Commit a write to one row, answering a race it lost for what it was.

    Between reading a row and committing, another request can delete it, or,
    for an application, whose version column notices, edit it. An UPDATE then
    matches no row and SQLAlchemy raises ``StaleDataError``, which the database
    handler would report as an outage, and so does a DELETE that reaches a
    versioned row: an application, or one a job or candidate takes with it.
    Instead: 404 if the row is gone, the same answer the request would have had
    a moment later, and 409 if it is still there but changed, where the change
    can be in an application the DELETE was taking with it. Rolling back also
    drops anything written alongside, such as a stage event, so no trail
    records a move that failed.

    Jobs and candidates carry no version, so a DELETE of one with no
    applications would have nothing to check: SQLAlchemy only warns when it
    matches no row. Their DELETE routes read the row under ``WRITE_LOCK``
    instead, so a second DELETE waits and then finds the row gone.
    """
    try:
        db.commit()
    except StaleDataError as exc:
        db.rollback()
        name = model.__name__
        if not _row_exists(db, model, row_id):
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, detail=f"{name} {row_id} not found"
            ) from exc
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                f"{name} {row_id} was changed by someone else while this request was in "
                "flight. Re-read it and retry."
            ),
        ) from exc


def refresh_row(db: Session, instance: Any, model: Any, row_id: int) -> None:
    """Read a row back after committing, answering 404 if it was deleted since.

    The write did commit, but another request removed the row before it could
    be read back, and ``refresh`` raises "Could not refresh instance". Any
    other failure to refresh is not a lost race and is raised as it was.
    """
    try:
        db.refresh(instance)
    except InvalidRequestError as exc:
        if _row_exists(db, model, row_id):
            raise
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"{model.__name__} {row_id} not found: another request deleted it just "
                "after this change was saved"
            ),
        ) from exc
