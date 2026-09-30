"""Shared FastAPI dependencies, and the helpers write routes commit and read rows back with."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Query
from fastapi import status as http_status
from sqlalchemy import Integer, any_, bindparam, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from app.models.application import Application
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
#: For the same reason nothing takes FOR KEY SHARE on a job or candidate ahead
#: of an insert that refers to it: the delete's FOR UPDATE comes last, so it
#: would wait on that lock while the insert waited on a row the delete was
#: removing. A repeat application deadlocked that way with a delete of its job
#: or candidate, waiting to learn whether the first application was gone. The
#: foreign key check takes the lock itself, at the end of the insert.
#:
#: A scoring run takes this lock on its job before it writes the job's scores
#: and holds it to its commit, so a delete of the job, an edit of it and
#: another run scoring it take turns with that write. See
#: ``services.matching.score_pool``.
WRITE_LOCK: dict[str, bool] = {"key_share": True}


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


def row_exists(db: Session, model: Any, row_id: int) -> bool:
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
        raise _lost_race(db, model, row_id) from exc


def _lost_race(db: Session, model: Any, row_id: int) -> HTTPException:
    """The answer to a write that lost a race: 404 if its row is gone, else 409."""
    name = model.__name__
    if not row_exists(db, model, row_id):
        return HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"{name} {row_id} not found"
        )
    return HTTPException(
        status_code=http_status.HTTP_409_CONFLICT,
        detail=(
            f"{name} {row_id} was changed by someone else while this request was in "
            "flight. Re-read it and retry."
        ),
    )


#: How many times a job or candidate DELETE is tried when all it lost was
#: applications another request deleted first. See :func:`delete_with_applications`.
DELETE_ATTEMPTS = 3


def delete_with_applications(db: Session, model: Any, row_id: int, load: Callable[[], Any]) -> None:
    """Delete a job or candidate, and the applications it takes with it.

    ``load`` reads the row under ``WRITE_LOCK``, answering 404 if it is gone.
    The applications are deleted by the ORM, which checks each one's version,
    so an application edited meanwhile still makes this a 409, as
    :func:`commit_row` answers. One that another request deleted instead, a
    delete of the other side of it say, is no conflict: this DELETE would have
    removed it anyway. Until 2026-09-29 that was a 409 too. Now the DELETE is
    rolled back and tried again, up to ``DELETE_ATTEMPTS`` times, and the
    applications it read are checked first: any still there under a newer
    version makes it a 409 at once. The row itself cannot go meanwhile, since
    ``load`` holds it until the commit; if it had, the next ``load`` is a 404.

    Cached scores are deleted by the ORM before the row, so the row's FOR
    UPDATE is the last lock taken (see ``Job.match_scores``); stage events go
    by their foreign key's ON DELETE CASCADE and are not read.
    """
    for _ in range(DELETE_ATTEMPTS):
        row = load()
        read = {application.id: application.version for application in row.applications}
        db.delete(row)
        try:
            db.commit()
            return
        except StaleDataError as exc:
            db.rollback()
            lost = exc
            now = dict(
                db.execute(
                    select(Application.id, Application.version).where(
                        # One array parameter, not ``in_``'s one per id (see
                        # app.services.matching for psycopg's 65,535 limit).
                        Application.id == any_(bindparam("ids", list(read), type_=ARRAY(Integer)))
                    )
                ).all()
            )
            edited = any(version != read[application_id] for application_id, version in now.items())
            if edited:
                raise _lost_race(db, model, row_id) from exc
    raise _lost_race(db, model, row_id) from lost


def refresh_row(db: Session, instance: Any, model: Any, row_id: int) -> None:
    """Read a row back after committing, answering 404 if it was deleted since.

    The write did commit, but another request removed the row before it could
    be read back, and ``refresh`` raises "Could not refresh instance". Any
    other failure to refresh is not a lost race and is raised as it was.
    """
    try:
        db.refresh(instance)
    except InvalidRequestError as exc:
        if row_exists(db, model, row_id):
            raise
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"{model.__name__} {row_id} not found: another request deleted it just "
                "after this change was saved"
            ),
        ) from exc
