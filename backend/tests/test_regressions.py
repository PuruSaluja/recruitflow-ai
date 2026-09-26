"""Pins for the bugs found by probing the API with awkward input on 2026-09-10.

The pins for each bug fail when that bug's fix is reverted. Alongside them sit
guards against over-correcting -- nullable fields must still accept null -- which
are expected to pass either way. Kept together so the record of what broke
stays readable in one place.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from psycopg.errors import LockNotAvailable
from pydantic import BaseModel
from sqlalchemy import (
    BigInteger,
    Engine,
    Integer,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import DataError, IntegrityError, InvalidRequestError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.api.body_limit import MAX_REQUEST_BYTES
from app.api.deps import refresh_row
from app.api.v1 import applications as applications_api
from app.api.v1 import matches as matches_api
from app.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.models import ApplicationStageEvent, Candidate, Job, MatchScore
from app.models.application import Application
from app.models.enums import ApplicationStatus
from app.schemas.candidate import MAX_SKILL_CHARS, CandidateUpdate
from app.schemas.common import MAX_ID, MAX_OFFSET
from app.schemas.job import SALARY_DIGITS, SALARY_PLACES, JobUpdate
from app.services.resume import MAX_RESUME_CHARS, MAX_UPLOAD_BYTES

Factory = Callable[..., dict[str, Any]]


def _patchable(model: type[Base], schema: type[BaseModel], *, nullable: bool) -> list[str]:
    """Update-schema fields whose column is (or is not) nullable.

    Derived from the table definition rather than the schemas' own constants, so
    a field dropped from REQUIRED_*_FIELDS still gets tested -- and fails.
    """
    columns = {column.name for column in model.__table__.columns if column.nullable is nullable}
    return sorted(columns & set(schema.model_fields))


CANDIDATE_REQUIRED = _patchable(Candidate, CandidateUpdate, nullable=False)
CANDIDATE_NULLABLE = _patchable(Candidate, CandidateUpdate, nullable=True)
JOB_REQUIRED = _patchable(Job, JobUpdate, nullable=False)
JOB_NULLABLE = _patchable(Job, JobUpdate, nullable=True)


def test_field_lists_were_derived() -> None:
    """Parametrising over an empty list would silently skip every test below."""
    assert CANDIDATE_REQUIRED and CANDIDATE_NULLABLE and JOB_REQUIRED and JOB_NULLABLE


# --- 1. explicit null on a NOT NULL field ------------------------------------
#
# Was: {"skills": null} returned 200 and stored an unreadable row, after which
# every read of it -- including the whole list endpoint -- returned 500. Other
# required fields returned a misleading 409 "conflicts with an existing record".


@pytest.mark.parametrize("field", CANDIDATE_REQUIRED)
def test_candidate_patch_rejects_null_for_a_required_field(
    client: TestClient, make_candidate: Factory, field: str
) -> None:
    candidate = make_candidate(skills=["SQL"])

    response = client.patch(f"/candidates/{candidate['id']}", json={field: None})

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["msg"]
    assert client.get(f"/candidates/{candidate['id']}").json() == candidate
    assert client.get("/candidates").status_code == 200


@pytest.mark.parametrize("field", JOB_REQUIRED)
def test_job_patch_rejects_null_for_a_required_field(
    client: TestClient, make_job: Factory, field: str
) -> None:
    job = make_job(required_skills=["SQL"])

    response = client.patch(f"/jobs/{job['id']}", json={field: None})

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["msg"]
    assert client.get(f"/jobs/{job['id']}").json() == job
    assert client.get("/jobs").status_code == 200


@pytest.mark.parametrize("field", CANDIDATE_NULLABLE)
def test_candidate_patch_still_accepts_null_for_a_nullable_field(
    client: TestClient, make_candidate: Factory, field: str
) -> None:
    candidate = make_candidate()

    response = client.patch(f"/candidates/{candidate['id']}", json={field: None})

    assert response.status_code == 200
    assert response.json()[field] is None


@pytest.mark.parametrize("field", JOB_NULLABLE)
def test_job_patch_still_accepts_null_for_a_nullable_field(
    client: TestClient, make_job: Factory, field: str
) -> None:
    job = make_job()

    response = client.patch(f"/jobs/{job['id']}", json={field: None})

    assert response.status_code == 200
    assert response.json()[field] is None


# --- 2. JSONB storing the JSON scalar null -----------------------------------
#
# Was: JSONB wrote Python None as JSON `null`, which a NOT NULL constraint
# accepts. The API-level check above is the first defence; this is the second,
# for any code path that writes through the ORM directly.


def test_every_jsonb_column_stores_none_as_sql_null() -> None:
    jsonb_columns = [
        column
        for table in Base.metadata.sorted_tables
        for column in table.columns
        if isinstance(column.type, JSONB)
    ]

    assert jsonb_columns
    assert [f"{c.table.name}.{c.name}" for c in jsonb_columns if not c.type.none_as_null] == []


def test_orm_write_of_none_to_candidate_skills_hits_the_not_null_constraint(
    clean_db: Engine, make_candidate: Factory
) -> None:
    candidate = make_candidate(skills=["SQL"])

    with Session(clean_db) as session:
        session.get(Candidate, candidate["id"]).skills = None
        with pytest.raises(IntegrityError, match="violates not-null constraint"):
            session.commit()


def test_orm_write_of_none_to_job_skills_hits_the_not_null_constraint(
    clean_db: Engine, make_job: Factory
) -> None:
    job = make_job(required_skills=["SQL"])

    with Session(clean_db) as session:
        session.get(Job, job["id"]).required_skills = None
        with pytest.raises(IntegrityError, match="violates not-null constraint"):
            session.commit()


# --- 3. % and _ in search text acting as SQL wildcards -----------------------
#
# Was: user text went straight into an ILIKE pattern, so "100%" matched
# "1000 users" and a lone "_" matched every row. Each case below pairs a row
# that should match with one that only an unescaped pattern would match.


@pytest.fixture
def search_pool(make_candidate: Factory, make_job: Factory) -> None:
    make_candidate(
        email="percent@example.com",
        current_title="Owns 100% of delivery",
        location="Remote, 100% async",
    )
    make_candidate(
        email="thousand@example.com",
        current_title="Scaled to 1000 users",
        location="Boston, MA",
    )
    make_candidate(
        email="underscore@example.com",
        current_title="Owns data_pipeline work",
        location="Austin_TX",
    )
    make_candidate(
        email="space@example.com",
        current_title="Owns data pipeline work",
        location="Austin TX",
    )
    make_job(title="Analyst, 100% remote")
    make_job(title="Analyst, 1000 users")
    make_job(title="BI_Developer")
    make_job(title="BI Developer")


def _candidate_emails(client: TestClient, **params: Any) -> list[str]:
    return sorted(c["email"] for c in client.get("/candidates", params=params).json()["items"])


def _job_titles(client: TestClient, **params: Any) -> list[str]:
    return sorted(j["title"] for j in client.get("/jobs", params=params).json()["items"])


@pytest.mark.usefixtures("search_pool")
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({"q": "100%"}, ["percent@example.com"], id="q-percent"),
        pytest.param({"q": "%"}, ["percent@example.com"], id="q-lone-percent"),
        pytest.param({"q": "data_pipeline"}, ["underscore@example.com"], id="q-underscore"),
        pytest.param({"location": "Austin_TX"}, ["underscore@example.com"], id="loc-underscore"),
        pytest.param({"location": "%"}, ["percent@example.com"], id="loc-lone-percent"),
    ],
)
def test_candidate_search_treats_wildcards_literally(
    client: TestClient, params: dict[str, str], expected: list[str]
) -> None:
    assert _candidate_emails(client, **params) == expected


@pytest.mark.usefixtures("search_pool")
@pytest.mark.parametrize(
    ("q", "expected"),
    [
        pytest.param("100%", ["Analyst, 100% remote"], id="percent"),
        pytest.param("%", ["Analyst, 100% remote"], id="lone-percent"),
        pytest.param("BI_D", ["BI_Developer"], id="underscore"),
    ],
)
def test_job_search_treats_wildcards_literally(
    client: TestClient, q: str, expected: list[str]
) -> None:
    assert _job_titles(client, q=q) == expected


@pytest.mark.usefixtures("search_pool")
def test_a_backslash_in_search_text_is_harmless(client: TestClient) -> None:
    response = client.get("/candidates", params={"q": "\\"})

    assert response.status_code == 200
    assert response.json()["items"] == []


# --- 4. a stage change overwriting a simultaneous one (found 2026-09-12) -----
#
# Was: two recruiters moving the same application at the same moment both got
# 200 in 11 of 12 races. Each move was legal on its own, so no constraint
# objected; the later write won and the loser's action vanished. Once stage
# history landed, both moves also left an event behind, so the trail claimed
# the application left the same stage twice and stage durations double-counted.
#
# The fix is the version counter on applications: SQLAlchemy adds
# "WHERE version = <value read>" to the UPDATE, so the losing write matches no
# row. These pins inject the interfering write deliberately rather than racing
# threads, so they fail the same way every time.


def _interfere(engine: Engine, application_id: int) -> None:
    """Move an application from another connection, as a second recruiter would."""
    with Session(engine) as other:
        application = other.get(Application, application_id)
        application.status = ApplicationStatus.REJECTED
        other.commit()


def test_a_stage_change_that_lost_a_race_is_refused(
    client: TestClient,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
) -> None:
    application = make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])
    client.patch(f"/applications/{application['id']}", json={"status": "screening"})

    # Fires after the endpoint has read the row and checked the transition,
    # which is exactly the window the two recruiters raced in.
    real_record = applications_api.record_stage_event

    def record_then_interfere(*args: Any, **kwargs: Any) -> Any:
        event = real_record(*args, **kwargs)
        _interfere(clean_db, application["id"])
        return event

    monkeypatch.setattr(applications_api, "record_stage_event", record_then_interfere)

    response = client.patch(f"/applications/{application['id']}", json={"status": "interview"})

    assert response.status_code == 409, response.text
    assert "changed by someone else" in response.json()["detail"]


def test_a_stage_change_that_lost_a_race_leaves_no_event(
    client: TestClient,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
) -> None:
    application = make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])
    client.patch(f"/applications/{application['id']}", json={"status": "screening"})

    real_record = applications_api.record_stage_event

    def record_then_interfere(*args: Any, **kwargs: Any) -> Any:
        event = real_record(*args, **kwargs)
        _interfere(clean_db, application["id"])
        return event

    monkeypatch.setattr(applications_api, "record_stage_event", record_then_interfere)
    client.patch(f"/applications/{application['id']}", json={"status": "interview"})

    events = client.get(f"/applications/{application['id']}/stage-events").json()
    moves = [(event["from_status"], event["to_status"]) for event in events]

    # The refused move must not appear, and nothing may leave a stage twice.
    assert ("screening", "interview") not in moves
    departures = [move[0] for move in moves if move[0] is not None]
    assert len(departures) == len(set(departures)), moves


def test_two_sessions_cannot_both_write_the_same_application(
    clean_db: Engine, make_job: Factory, make_candidate: Factory, make_application: Factory
) -> None:
    """The mechanism itself, without the endpoint in the way."""
    application = make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])

    with Session(clean_db) as first, Session(clean_db) as second:
        mine = first.get(Application, application["id"])
        theirs = second.get(Application, application["id"])

        mine.status = ApplicationStatus.SCREENING
        first.commit()

        theirs.status = ApplicationStatus.REJECTED
        with pytest.raises(StaleDataError):
            second.commit()


def test_version_is_not_part_of_the_api_contract(
    client: TestClient, make_job: Factory, make_candidate: Factory, make_application: Factory
) -> None:
    """It is a concurrency mechanism, not a field clients are meant to send back."""
    application = make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])

    assert "version" not in application

    response = client.patch(f"/applications/{application['id']}", json={"version": 99})

    assert response.status_code == 422, response.text


# --- 5. non-positive ids in a path (found 2026-09-12) ------------------------
#
# Was: /applications/0 and /applications/-1 answered 404 "Application -1 not
# found", claiming to have looked for a row that could never exist. Body and
# query ids already answered 422 about the payload; path ids did not, because
# no route declared gt=0.


@pytest.mark.parametrize("bad_id", [0, -1])
@pytest.mark.parametrize(
    ("method", "template"),
    [
        pytest.param("get", "/jobs/{id}", id="get-job"),
        pytest.param("patch", "/jobs/{id}", id="patch-job"),
        pytest.param("post", "/jobs/{id}/close", id="close-job"),
        pytest.param("delete", "/jobs/{id}", id="delete-job"),
        pytest.param("get", "/candidates/{id}", id="get-candidate"),
        pytest.param("patch", "/candidates/{id}", id="patch-candidate"),
        pytest.param("delete", "/candidates/{id}", id="delete-candidate"),
        pytest.param("get", "/candidates/{id}/resume", id="get-resume"),
        pytest.param("get", "/applications/{id}", id="get-application"),
        pytest.param("patch", "/applications/{id}", id="patch-application"),
        pytest.param("delete", "/applications/{id}", id="delete-application"),
        pytest.param("get", "/applications/{id}/stage-events", id="stage-events"),
        pytest.param("post", "/applications/{id}/reopen", id="reopen-application"),
        pytest.param("post", "/jobs/{id}/score", id="score-job"),
        pytest.param("get", "/jobs/{id}/matches", id="job-matches"),
    ],
)
def test_a_non_positive_path_id_is_a_payload_error(
    client: TestClient, method: str, template: str, bad_id: int
) -> None:
    path = template.format(id=bad_id)
    request = getattr(client, method)

    response = request(path, json={}) if method in {"patch", "post"} else request(path)

    assert response.status_code == 422, f"{method.upper()} {path} -> {response.status_code}"


@pytest.mark.parametrize(
    ("method", "template"),
    [
        pytest.param("get", "/applications/{id}", id="get-application"),
        pytest.param("get", "/applications/{id}/stage-events", id="stage-events"),
        pytest.param("get", "/jobs/{id}", id="get-job"),
        pytest.param("get", "/candidates/{id}", id="get-candidate"),
        pytest.param("post", "/jobs/{id}/score", id="score-job"),
        pytest.param("get", "/jobs/{id}/matches", id="job-matches"),
    ],
)
def test_a_plausible_but_absent_id_is_still_a_404(
    client: TestClient, method: str, template: str
) -> None:
    """Guards against over-correcting: gt=0 must not swallow genuine 404s."""
    response = getattr(client, method)(template.format(id=999999))

    assert response.status_code == 404


# --- 6. ids and offsets past what PostgreSQL can hold (found 2026-09-22) -----
#
# Was: an id above 2,147,483,647, the INTEGER keys' maximum, answered 503
# "Database unavailable" on every route, and so did an offset above BIGINT's
# maximum. PostgreSQL raises "integer out of range", a DataError, and the
# database error handler reported it as an outage.

TOO_BIG_ID = MAX_ID + 1


def _id_routes(app: FastAPI) -> list[tuple[str, str, list[str]]]:
    """Every (method, path, path id names) the app serves with an id in its path.

    Read from the app's OpenAPI document rather than listed by hand, so a route
    added later is covered without anyone remembering to add it here.
    """
    prefix = get_settings().api_prefix
    routes = []
    for path, operations in app.openapi()["paths"].items():
        for method, operation in operations.items():
            names = [p["name"] for p in operation.get("parameters", []) if p["in"] == "path"]
            if names:
                routes.append((method.upper(), path.removeprefix(prefix), names))
    assert len(routes) >= 16  # guards against the walk silently finding nothing
    return routes


def _path_errors(response: Any) -> set[str]:
    if response.status_code != 422:
        return set()
    return {error["loc"][1] for error in response.json()["detail"] if error["loc"][0] == "path"}


def test_every_path_id_past_the_integer_range_is_a_payload_error(
    app: FastAPI, client: TestClient
) -> None:
    wrong = []
    for method, template, names in _id_routes(app):
        path = template.format(**{name: TOO_BIG_ID for name in names})
        response = client.request(method, path, json={} if method in {"PATCH", "POST"} else None)
        if response.status_code != 422 or _path_errors(response) != set(names):
            wrong.append(f"{method} {path} -> {response.status_code}")

    assert wrong == []


def test_the_largest_integer_id_is_still_a_plain_404(app: FastAPI, client: TestClient) -> None:
    """Guards against over-correcting: the bound must admit every id a row can have."""
    upload = {"file": ("cv.txt", b"Built reporting pipelines in SQL and Python.", "text/plain")}
    wrong = []
    for method, template, names in _id_routes(app):
        path = template.format(**{name: MAX_ID for name in names})
        response = client.request(
            method,
            path,
            json={} if method in {"PATCH", "POST"} else None,
            files=upload if method == "PUT" else None,
        )
        if response.status_code != 404:
            wrong.append(f"{method} {path} -> {response.status_code}")

    assert wrong == []


def test_the_bounds_are_postgresqls_own_limits(clean_db: Engine) -> None:
    """Pins the constants themselves: every test above is relative to them.

    MAX_ID is the largest id a row can have only while every primary key is an
    INTEGER, so that is checked too; a move to BIGINT keys must move MAX_ID.
    """
    with clean_db.connect() as connection:
        for limit, sql_type in ((MAX_ID, "INTEGER"), (MAX_OFFSET, "BIGINT")):
            cast = text(f"SELECT CAST(:value AS {sql_type})")
            assert connection.scalar(cast, {"value": limit}) == limit
            with pytest.raises(DataError, match="out of range"):
                connection.execute(cast, {"value": limit + 1})
            connection.rollback()

    keys = [column for table in Base.metadata.sorted_tables for column in table.primary_key]
    assert keys
    assert [
        f"{c.table.name}.{c.name}"
        for c in keys
        if not isinstance(c.type, Integer) or isinstance(c.type, BigInteger)
    ] == []


@pytest.mark.parametrize("field", ["job_id", "candidate_id"])
def test_a_body_id_past_the_integer_range_is_a_payload_error(
    client: TestClient, make_job: Factory, make_candidate: Factory, field: str
) -> None:
    ids = {"job_id": make_job()["id"], "candidate_id": make_candidate()["id"]}

    response = client.post("/applications", json={**ids, field: TOO_BIG_ID})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["body", field]]
    assert client.post("/applications", json={**ids, field: MAX_ID}).status_code == 404


@pytest.mark.parametrize("field", ["job_id", "candidate_id"])
def test_a_query_id_past_the_integer_range_is_a_payload_error(
    client: TestClient, field: str
) -> None:
    response = client.get("/applications", params={field: TOO_BIG_ID})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["query", field]]
    assert client.get("/applications", params={field: MAX_ID}).json()["total"] == 0


def test_a_candidate_id_to_score_past_the_integer_range_is_a_payload_error(
    client: TestClient, make_job: Factory
) -> None:
    url = f"/jobs/{make_job()['id']}/score"

    response = client.post(url, json={"candidate_ids": [1, TOO_BIG_ID]})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["body", "candidate_ids", 1]]
    # At the bound it is an ordinary unknown candidate, refused before any matcher loads.
    assert client.post(url, json={"candidate_ids": [MAX_ID]}).status_code == 404


@pytest.mark.parametrize("path", ["/jobs", "/candidates", "/applications"])
def test_an_offset_past_the_bigint_range_is_a_payload_error(
    client: TestClient,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
    path: str,
) -> None:
    make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])

    response = client.get(path, params={"offset": MAX_OFFSET + 1})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["query", "offset"]]
    at_bound = client.get(path, params={"offset": MAX_OFFSET})
    assert at_bound.status_code == 200
    assert at_bound.json()["items"] == []
    assert at_bound.json()["total"] == 1


# --- 7. a write that loses a race to another request (found 2026-09-14) ------
#
# Was: a lost race answered 503 "Database unavailable", except a PATCH or
# reopen of an application, which answered 409 even when the row was gone. A
# DELETE whose row another request removed first raised StaleDataError; a
# write whose row was deleted just after it committed raised "Could not
# refresh instance". Measured on 2026-09-14 for applications (19 of 50 racing
# deletes answered 503), and on 2026-09-23 by replaying each race below
# against the previous code: 14 of the 16 shapes in LOST_RACES answered 503
# and the other two 409 although the row was gone; the upload, refused-delete
# and create cases answered 503. Three of those shapes can no longer happen,
# since their routes now hold the row from reading it to committing; they are
# pinned in section 9.
#
# Each case replays its race deterministically: the other request's statement
# runs on its own connection when the endpoint's session reaches its commit
# (before it, when the endpoint has only read the row) or just after it.


@contextmanager
def _interloper(engine: Engine, moment: str, statement: Any) -> Iterator[list[str]]:
    """Run ``statement`` once, from another connection, at ``moment`` in any session.

    Records ``"ran"``, or ``"waited"`` when a row lock held by the endpoint's
    session stood in the way. The statement gives up after half a second rather
    than wait for a session that cannot finish until it returns.
    """
    fired: list[str] = []

    def run(session: Session) -> None:
        if fired:
            return
        try:
            with engine.begin() as connection:
                connection.execute(text("SET LOCAL lock_timeout = '500ms'"))
                connection.execute(statement)
        except OperationalError as exc:
            if not isinstance(exc.orig, LockNotAvailable):
                raise
            fired.append("waited")
        else:
            fired.append("ran")

    event.listen(Session, moment, run)
    try:
        yield fired
    finally:
        event.remove(Session, moment, run)


def _delete_application(ids: dict[str, int]) -> Any:
    return delete(Application).where(Application.id == ids["application"])


def _edit_application(ids: dict[str, int]) -> Any:
    """Bump the version, as any committed edit by another request does."""
    table = Application.__table__
    return update(table).where(table.c.id == ids["application"]).values(version=table.c.version + 1)


def _delete_job(ids: dict[str, int]) -> Any:
    return delete(Job).where(Job.id == ids["job"])


def _delete_candidate(ids: dict[str, int]) -> Any:
    return delete(Candidate).where(Candidate.id == ids["candidate"])


BEFORE, AFTER = "before_commit", "after_commit"
RACE_RESUME = "Built reporting pipelines in SQL and Python."
GONE = "{} {{{}}} not found"
CHANGED = "{} {{{}}} was changed by someone else while this request was in flight"
GONE_AFTER = "{} {{{}}} not found: another request deleted it just after this change was saved"

# fmt: off
LOST_RACES = [
    pytest.param("delete", "/applications/{application}", None, BEFORE, _delete_application,
                 404, GONE.format("Application", "application"), id="delete-application-deleted"),
    pytest.param("delete", "/applications/{application}", None, BEFORE, _edit_application,
                 409, CHANGED.format("Application", "application"), id="delete-application-edited"),
    pytest.param("patch", "/applications/{application}", {"notes": "n"}, BEFORE,
                 _delete_application, 404, GONE.format("Application", "application"),
                 id="patch-application-deleted"),
    pytest.param("patch", "/applications/{application}", {"notes": "n"}, AFTER,
                 _delete_application, 404, GONE_AFTER.format("Application", "application"),
                 id="patch-application-deleted-after"),
    pytest.param("post", "/applications/{application}/reopen", None, BEFORE, _delete_application,
                 404, GONE.format("Application", "application"), id="reopen-deleted"),
    pytest.param("post", "/applications/{application}/reopen", None, AFTER, _delete_application,
                 404, GONE_AFTER.format("Application", "application"), id="reopen-deleted-after"),
    pytest.param("patch", "/jobs/{job}", {"title": "Senior Analyst"}, AFTER, _delete_job,
                 404, GONE_AFTER.format("Job", "job"), id="patch-job-deleted-after"),
    pytest.param("post", "/jobs/{job}/close", None, BEFORE, _delete_job,
                 404, GONE.format("Job", "job"), id="close-job-deleted"),
    pytest.param("post", "/jobs/{job}/close", None, AFTER, _delete_job,
                 404, GONE_AFTER.format("Job", "job"), id="close-job-deleted-after"),
    pytest.param("delete", "/jobs/{job}", None, BEFORE, _edit_application,
                 409, CHANGED.format("Job", "job"), id="delete-job-application-edited"),
    pytest.param("patch", "/candidates/{candidate}", {"location": "Atlanta, GA"}, BEFORE,
                 _delete_candidate, 404, GONE.format("Candidate", "candidate"),
                 id="patch-candidate-deleted"),
    pytest.param("patch", "/candidates/{candidate}", {"location": "Atlanta, GA"}, AFTER,
                 _delete_candidate, 404, GONE_AFTER.format("Candidate", "candidate"),
                 id="patch-candidate-deleted-after"),
    pytest.param("delete", "/candidates/{candidate}", None, BEFORE, _delete_application,
                 409, CHANGED.format("Candidate", "candidate"),
                 id="delete-candidate-application-gone"),
]
# fmt: on


@pytest.mark.parametrize(
    ("method", "path", "body", "moment", "interloper", "status", "detail"), LOST_RACES
)
def test_a_write_that_lost_a_race_says_what_happened_not_that_the_database_is_down(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    moment: str,
    interloper: Callable[[dict[str, int]], Any],
    status: int,
    detail: str,
) -> None:
    ids = {"job": make_job()["id"], "candidate": make_candidate()["id"]}
    ids["application"] = make_application(job_id=ids["job"], candidate_id=ids["candidate"])["id"]
    if "reopen" in path:
        client.patch(f"/applications/{ids['application']}", json={"status": "rejected"})

    with _interloper(clean_db, moment, interloper(ids)) as fired:
        response = client.request(method.upper(), path.format(**ids), json=body)

    assert fired == ["ran"], f"the race was not replayed: {fired}"
    assert response.status_code == status, response.text
    assert response.json()["detail"].startswith(detail.format(**ids))


@pytest.mark.parametrize(
    ("moment", "detail"), [(BEFORE, GONE), (AFTER, GONE_AFTER)], ids=["before", "after"]
)
def test_a_resume_upload_whose_row_was_deleted_is_a_404(
    client: TestClient, clean_db: Engine, make_candidate: Factory, moment: str, detail: str
) -> None:
    ids = {"candidate": make_candidate()["id"]}
    upload = {"file": ("cv.txt", b"Built reporting pipelines in SQL and Python.", "text/plain")}

    with _interloper(clean_db, moment, _delete_candidate(ids)) as fired:
        response = client.put(f"/candidates/{ids['candidate']}/resume", files=upload)

    assert fired == ["ran"], f"the race was not replayed: {fired}"
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == detail.format("Candidate", "candidate").format(**ids)


@pytest.mark.parametrize("model", [Job, Candidate, Application], ids=lambda m: m.__name__)
def test_a_create_whose_new_row_was_deleted_before_it_was_read_back_is_a_404(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    model: Any,
) -> None:
    job_id, candidate_id = make_job()["id"], make_candidate()["id"]
    path, body = {
        Job: (
            "/jobs",
            {"title": "Data Engineer", "description": "Build and run the nightly loads."},
        ),
        Candidate: (
            "/candidates",
            {"full_name": "Ada Okafor", "email": "ada@example.com", "resume_text": RACE_RESUME},
        ),
        Application: ("/applications", {"job_id": job_id, "candidate_id": candidate_id}),
    }[model]
    newest = select(func.max(model.id)).scalar_subquery()

    with _interloper(clean_db, AFTER, delete(model).where(model.id == newest)) as fired:
        response = client.post(path, json=body)

    assert fired == ["ran"], f"the race was not replayed: {fired}"
    assert response.status_code == 404, response.text
    detail = response.json()["detail"]
    assert detail.startswith(f"{model.__name__} ")
    assert detail.endswith(
        " not found: another request deleted it just after this change was saved"
    )


def test_a_delete_refused_because_the_row_changed_leaves_it_whole(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
) -> None:
    """Guards the 409 case: the row and its stage trail must survive untouched."""
    application = make_application(job_id=make_job()["id"], candidate_id=make_candidate()["id"])

    with _interloper(clean_db, BEFORE, _edit_application({"application": application["id"]})):
        assert client.delete(f"/applications/{application['id']}").status_code == 409

    assert client.get(f"/applications/{application['id']}").status_code == 200
    assert len(client.get(f"/applications/{application['id']}/stage-events").json()) == 1


def test_a_refresh_that_fails_for_another_reason_is_not_reported_as_a_lost_race(
    clean_db: Engine, make_job: Factory
) -> None:
    """Only a row that is really gone becomes a 404; any other failure is raised as it was."""
    job_id = make_job()["id"]

    # A transient instance cannot be refreshed, although its row exists.
    with Session(clean_db) as session, pytest.raises(InvalidRequestError, match="not persistent"):
        refresh_row(session, Job(id=job_id), Job, job_id)


# --- 8. values the database, or the answer, cannot hold (found 2026-09-23) ---
#
# Was: text holding a NUL character (U+0000), which no PostgreSQL text column
# can store, answered 503 "Database unavailable" in all 13 body fields and
# search parameters tried, and so did a lone surrogate inside a skill list,
# which PostgreSQL refuses as JSON.
#
# Was: a salary past the NUMERIC(10, 2) columns, from 100,000,000 up, answered
# 503 as well, and a fraction of a cent was rounded away without a word.
#
# Was: every 422 repeated its input, 5,000,160 bytes back for a
# 5,000,000-character resume. A lone surrogate in a plain text field was
# refused by validation, but could not be encoded into that answer, so it
# became a 500.

NUL = chr(0)
LONE_SURROGATE = chr(0xD800)
JOB_BODY = {"title": "Data Analyst", "description": "Own the weekly metrics review."}


def _post_ascii_json(client: TestClient, path: str, payload: Any) -> Any:
    """Send ``payload`` as ASCII JSON, so a lone surrogate travels as a valid escape."""
    body = json.dumps(payload, ensure_ascii=True).encode("ascii")
    return client.post(path, content=body, headers={"content-type": "application/json"})


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        pytest.param("/jobs", {**JOB_BODY, "title": f"Data{NUL}Analyst"}, id="job-title"),
        pytest.param("/jobs", {**JOB_BODY, "description": f"{NUL}" * 30}, id="job-description"),
        pytest.param("/jobs", {**JOB_BODY, "required_skills": [f"S{NUL}QL"]}, id="job-skills"),
        pytest.param(
            "/candidates",
            {
                "full_name": "Nul Byte",
                "email": "nul@example.com",
                "resume_text": f"{RACE_RESUME}{NUL}",
            },
            id="candidate-resume",
        ),
        pytest.param(
            "/candidates",
            {
                "full_name": "Nul Byte",
                "email": "nul@example.com",
                "resume_text": RACE_RESUME,
                "skills": [f"S{NUL}QL"],
            },
            id="candidate-skills",
        ),
        pytest.param(
            "/candidates",
            {
                "full_name": "Lone Surrogate",
                "email": "surrogate@example.com",
                "resume_text": RACE_RESUME,
                "skills": [LONE_SURROGATE],
            },
            id="candidate-skills-surrogate",
        ),
    ],
)
def test_a_value_postgresql_cannot_store_is_a_422_not_an_outage(
    client: TestClient, path: str, payload: dict[str, Any]
) -> None:
    response = _post_ascii_json(client, path, payload)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_value"
    assert client.get(path).json()["total"] == 0


@pytest.mark.parametrize(
    ("path", "params"),
    [
        pytest.param("/jobs", {"q": f"a{NUL}b"}, id="job-search"),
        pytest.param("/candidates", {"q": f"a{NUL}b"}, id="candidate-search"),
        pytest.param("/candidates", {"skill": f"S{NUL}QL"}, id="candidate-skill-filter"),
        pytest.param("/candidates", {"location": f"a{NUL}b"}, id="candidate-location-filter"),
        pytest.param("/applications", {"source": f"a{NUL}b"}, id="application-source-filter"),
    ],
)
def test_a_search_for_a_nul_character_is_a_422_not_an_outage(
    client: TestClient, path: str, params: dict[str, str]
) -> None:
    response = client.get(path, params=params)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_value"


def test_an_update_with_a_nul_character_is_a_422_and_changes_nothing(
    client: TestClient, make_job: Factory, make_candidate: Factory, make_application: Factory
) -> None:
    job, candidate = make_job(), make_candidate()
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    updates = [
        (f"/jobs/{job['id']}", {"department": f"Data{NUL}"}, job),
        (f"/candidates/{candidate['id']}", {"location": f"Atlanta{NUL}"}, candidate),
        (f"/applications/{application['id']}", {"notes": f"Strong{NUL}"}, application),
        (
            f"/applications/{application['id']}",
            {"status": "screening", "changed_by": f"Ana{NUL}"},
            application,
        ),
    ]

    for path, body, before in updates:
        response = client.patch(path, json=body)

        assert response.status_code == 422, f"{path} {body} -> {response.text}"
        assert response.json()["code"] == "invalid_value"
        assert client.get(path).json() == before


SALARIES = [
    pytest.param("99999999.99", True, id="largest"),
    pytest.param("100000000", False, id="one-digit-too-many"),
    pytest.param("99999999.995", False, id="rounds-up-past-the-column"),
    pytest.param("85000.001", False, id="a-fraction-of-a-cent"),
    pytest.param(10**15, False, id="a-json-integer"),
    pytest.param("85000.5", True, id="half-a-unit"),
]


@pytest.mark.parametrize("field", ["salary_min", "salary_max"])
@pytest.mark.parametrize(("salary", "fits"), SALARIES)
def test_a_new_salary_must_fit_the_numeric_10_2_columns(
    client: TestClient, field: str, salary: str | int, fits: bool
) -> None:
    response = client.post("/jobs", json={**JOB_BODY, field: salary})

    if fits:
        assert response.status_code == 201, response.text
        assert Decimal(response.json()[field]) == Decimal(salary)
    else:
        assert response.status_code == 422, response.text
        assert [error["loc"] for error in response.json()["detail"]] == [["body", field]]
        assert client.get("/jobs").json()["total"] == 0


@pytest.mark.parametrize("field", ["salary_min", "salary_max"])
@pytest.mark.parametrize(("salary", "fits"), SALARIES)
def test_an_updated_salary_must_fit_the_numeric_10_2_columns(
    client: TestClient, make_job: Factory, field: str, salary: str | int, fits: bool
) -> None:
    job = make_job()

    response = client.patch(f"/jobs/{job['id']}", json={field: salary})

    if fits:
        assert response.status_code == 200, response.text
        assert Decimal(response.json()[field]) == Decimal(salary)
    else:
        assert response.status_code == 422, response.text
        assert [error["loc"] for error in response.json()["detail"]] == [["body", field]]
        assert client.get(f"/jobs/{job['id']}").json() == job


def test_the_salary_bounds_are_the_columns_own() -> None:
    for column in (Job.__table__.c.salary_min, Job.__table__.c.salary_max):
        assert (column.type.precision, column.type.scale) == (SALARY_DIGITS, SALARY_PLACES)


def test_a_validation_error_does_not_send_the_request_back(client: TestClient) -> None:
    # One character past the limit: a body past the request size cap (section 10)
    # is refused before validation, so it can no longer reach this handler.
    resume = "a" * (MAX_RESUME_CHARS + 1)
    payload = {"full_name": "Long Resume", "email": "long@example.com", "resume_text": resume}

    response = client.post("/candidates", json=payload)

    assert response.status_code == 422
    assert len(response.content) < 500
    assert response.json()["detail"] == [
        {
            "type": "string_too_long",
            "loc": ["body", "resume_text"],
            "msg": "String should have at most 100000 characters",
            "ctx": {"max_length": 100000},
        }
    ]


def test_a_missing_field_does_not_send_the_whole_body_back(client: TestClient) -> None:
    response = client.post("/jobs", json={"title": "Data Analyst", "department": "x" * 100})

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {"type": "missing", "loc": ["body", "description"], "msg": "Field required"}
    ]


@pytest.mark.parametrize(
    ("overrides", "loc"),
    [
        pytest.param({"title": f"Data{LONE_SURROGATE}"}, ["body", "title"], id="title"),
        # A job's skills are length-checked since 2026-09-24, so validation
        # reads each one and refuses the surrogate before the database sees it.
        pytest.param(
            {"preferred_skills": [LONE_SURROGATE]}, ["body", "preferred_skills", 0], id="skill"
        ),
    ],
)
def test_a_lone_surrogate_in_a_text_field_is_a_422_not_a_500(
    client: TestClient, overrides: dict[str, Any], loc: list[Any]
) -> None:
    response = _post_ascii_json(client, "/jobs", {**JOB_BODY, **overrides})

    assert response.status_code == 422
    assert response.content.isascii()
    assert [(e["type"], e["loc"]) for e in response.json()["detail"]] == [("string_unicode", loc)]


def test_a_422_stays_encodable_whatever_its_messages_hold(app: FastAPI) -> None:
    """Pins the handler's promise directly, since no request tried can reach it now.

    With the input left out, a lone surrogate has no way into the answer: one in
    a field name is refused at the body's own location. The answer is written as
    ASCII JSON regardless, so a message that ever quotes one stays encodable.
    """
    handler = app.exception_handlers[RequestValidationError]
    message = f"Unknown skill {LONE_SURROGATE}"
    error = {"type": "value_error", "loc": ("body", "skills"), "msg": message, "input": "x"}

    response = handler(None, RequestValidationError([error]))

    assert response.status_code == 422
    assert response.body.isascii()
    assert json.loads(response.body) == {
        "detail": [{"type": "value_error", "loc": ["body", "skills"], "msg": message}]
    }


# --- 9. two requests writing one job or candidate at once (2026-09-11, -23) --
#
# Was: two PATCHes of a job, each valid but together inverting its salary band,
# answered 409 "Resource conflicts with an existing record" in 11 of 20 races:
# both checked the band against the ends they had read, and the check
# constraint refused the second. One after the other, the second is a 422
# naming the band. And a DELETE of a job or candidate with no applications,
# beaten by another DELETE, answered 204 as though it had won, where one after
# the other it is a 404.
#
# The fix reads the row under the lock its write takes anyway, so the second
# request waits for the first and then sees what it left. Each case holds the
# other request's write open on its own connection, sends the request from a
# thread, and commits the other write once the request is seen waiting for a
# lock, or after ten seconds, so a case can fail but never hang.

# Client backends only: an autovacuum worker waiting on a lock would otherwise
# count, and the other write would commit before the request had reached it.
LOCK_WAITS = text(
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND backend_type = 'client backend' AND wait_event_type = 'Lock'"
)


def _waiting_for_a_lock(engine: Engine) -> bool:
    with engine.connect() as watcher:
        return watcher.scalar(LOCK_WAITS) > 0


def _behind_another_write(
    engine: Engine, statement: Any, request: Callable[[], Any]
) -> tuple[bool, Any]:
    """Send ``request`` while another request's ``statement`` is written but uncommitted.

    Returns whether the request was seen waiting for a lock, and its response.
    """
    with ThreadPoolExecutor(max_workers=1) as pool, engine.connect() as other:
        with other.begin():
            other.execute(statement)
            response = pool.submit(request)
            deadline = time.monotonic() + 10
            while not _waiting_for_a_lock(engine) and time.monotonic() < deadline:
                time.sleep(0.01)
            waited = _waiting_for_a_lock(engine)
        return waited, response.result(timeout=30)


def test_two_edits_that_together_invert_a_salary_band_refuse_the_second_as_invalid(
    client: TestClient, clean_db: Engine, make_job: Factory
) -> None:
    job = make_job(salary_min=50000, salary_max=100000)
    raise_the_minimum = update(Job).where(Job.id == job["id"]).values(salary_min=90000)

    waited, response = _behind_another_write(
        clean_db,
        raise_the_minimum,
        lambda: client.patch(f"/jobs/{job['id']}", json={"salary_max": 60000}),
    )

    assert waited, "the edit never waited for the other one"
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "salary_max must be greater than or equal to salary_min"
    stored = client.get(f"/jobs/{job['id']}").json()
    assert Decimal(stored["salary_min"]) == 90000
    assert Decimal(stored["salary_max"]) == 100000


def test_two_edits_of_different_fields_both_land(
    client: TestClient, clean_db: Engine, make_job: Factory
) -> None:
    """Guards against refusing every concurrent edit, as a version check would."""
    job = make_job()
    retitle = update(Job).where(Job.id == job["id"]).values(title="Senior Data Analyst")

    waited, response = _behind_another_write(
        clean_db,
        retitle,
        lambda: client.patch(f"/jobs/{job['id']}", json={"location": "Atlanta, GA"}),
    )

    assert waited, "the edit never waited for the other one"
    assert response.status_code == 200, response.text
    assert (response.json()["title"], response.json()["location"]) == (
        "Senior Data Analyst",
        "Atlanta, GA",
    )


@pytest.mark.parametrize("model", [Job, Candidate], ids=lambda m: m.__name__)
def test_a_delete_beaten_by_another_delete_is_a_404(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory, model: Any
) -> None:
    row_id = (make_job if model is Job else make_candidate)()["id"]

    waited, response = _behind_another_write(
        clean_db,
        delete(model).where(model.id == row_id),
        lambda: client.delete(f"/{model.__tablename__}/{row_id}"),
    )

    assert waited, "the delete never waited for the other one"
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == f"{model.__name__} {row_id} not found"


def test_a_job_delete_and_a_scoring_run_on_it_do_not_deadlock(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory
) -> None:
    """Found by a live soak on 2026-09-24: one of the two was answered 503.

    A scoring run rescores the job's cached rows before inserting new ones,
    whose foreign key check takes FOR KEY SHARE on the job. A delete that held
    the job FOR UPDATE from its read then waited on a rescored row, while the
    insert waited on the job.
    """
    job_id, rescored, added = make_job()["id"], make_candidate()["id"], make_candidate()["id"]
    score = {"job_id": job_id, "score": 0.5, "features": {}, "model_version": "v1"}
    with clean_db.begin() as setup:
        cached = setup.scalar(
            insert(MatchScore).values(candidate_id=rescored, **score).returning(MatchScore.id)
        )

    with ThreadPoolExecutor(max_workers=1) as pool, clean_db.connect() as scoring:
        with scoring.begin():
            scoring.execute(text("SET LOCAL lock_timeout = '5s'"))
            scoring.execute(update(MatchScore).where(MatchScore.id == cached).values(score=0.6))
            response = pool.submit(lambda: client.delete(f"/jobs/{job_id}"))
            deadline = time.monotonic() + 10
            while not _waiting_for_a_lock(clean_db) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert _waiting_for_a_lock(clean_db), "the delete never waited for the rescored row"
            scoring.execute(insert(MatchScore).values(candidate_id=added, **score))
        answer = response.result(timeout=30)

    assert answer.status_code == 204, answer.text


# fmt: off
HELD_ROWS = [
    pytest.param("patch", "/jobs/{job}", {"title": "Senior Analyst"}, _delete_job, 200,
                 id="patch-job"),
    pytest.param("delete", "/jobs/{job}", None, _delete_job, 204, id="delete-job"),
    pytest.param("delete", "/candidates/{candidate}", None, _delete_candidate, 204,
                 id="delete-candidate"),
]
# fmt: on


@pytest.mark.parametrize(("method", "path", "body", "interloper", "status"), HELD_ROWS)
def test_a_write_holds_its_row_from_reading_it_to_committing(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    interloper: Callable[[dict[str, int]], Any],
    status: int,
) -> None:
    """A delete arriving between the read and the commit waits, and this write lands."""
    ids = {"job": make_job()["id"], "candidate": make_candidate()["id"]}

    with _interloper(clean_db, BEFORE, interloper(ids)) as fired:
        response = client.request(method.upper(), path.format(**ids), json=body)

    assert fired == ["waited"], fired
    assert response.status_code == status, response.text


def test_an_edit_holding_its_job_lets_new_applications_through(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory
) -> None:
    """The edit takes FOR NO KEY UPDATE, which a new row referencing the job does not wait on."""
    job_id, candidate_id = make_job()["id"], make_candidate()["id"]
    apply = insert(Application).values(job_id=job_id, candidate_id=candidate_id)

    with _interloper(clean_db, BEFORE, apply) as fired:
        response = client.patch(f"/jobs/{job_id}", json={"title": "Senior Analyst"})

    assert fired == ["ran"], fired
    assert response.status_code == 200, response.text


# Found 2026-09-24: an application whose job or candidate was deleted between
# the create's check and its insert answered 409 "Resource conflicts with an
# existing record", the foreign key refusing the row. One after the other, it
# is a 404 naming what is gone.


@pytest.mark.parametrize("parent", [Job, Candidate], ids=lambda m: m.__name__)
def test_an_application_to_a_row_being_deleted_is_a_404(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory, parent: Any
) -> None:
    ids = {Job: make_job()["id"], Candidate: make_candidate()["id"]}
    body = {"job_id": ids[Job], "candidate_id": ids[Candidate]}

    waited, response = _behind_another_write(
        clean_db,
        delete(parent).where(parent.id == ids[parent]),
        lambda: client.post("/applications", json=body),
    )

    assert waited, "the application never waited for the delete"
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == f"{parent.__name__} {ids[parent]} not found"


def test_an_application_to_a_job_being_edited_does_not_wait(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory
) -> None:
    """Creating an application must not wait on an edit of its job."""
    job_id, candidate_id = make_job()["id"], make_candidate()["id"]
    retitle = update(Job).where(Job.id == job_id).values(title="Senior Data Analyst")
    body = {"job_id": job_id, "candidate_id": candidate_id}

    # The edit stays uncommitted until the application answers, or for five
    # seconds; an application that waited for it times out instead of hanging.
    with ThreadPoolExecutor(max_workers=1) as pool, clean_db.connect() as other, other.begin():
        other.execute(retitle)
        response = pool.submit(lambda: client.post("/applications", json=body)).result(5)

    assert response.status_code == 201, response.text


@pytest.mark.parametrize("parent", [Job, Candidate], ids=lambda m: m.__name__)
def test_a_repeat_application_racing_a_delete_of_its_job_or_candidate_is_a_404(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
    parent: Any,
) -> None:
    """Found in the 2026-09-24 recheck: the two deadlocked, and the loser answered 503.

    The delete has removed the first application and is about to remove the row
    itself, the way the delete route does it. The repeat application's insert
    waits to learn whether that first application is really gone, and must not
    be holding anything the delete then waits for.
    """
    ids = {Job: make_job()["id"], Candidate: make_candidate()["id"]}
    first = make_application(job_id=ids[Job], candidate_id=ids[Candidate])["id"]
    body = {"job_id": ids[Job], "candidate_id": ids[Candidate]}
    row = parent.__table__.c.id

    with ThreadPoolExecutor(max_workers=1) as pool, clean_db.connect() as deleting:
        with deleting.begin():
            deleting.execute(text("SET LOCAL lock_timeout = '5s'"))
            deleting.execute(select(row).where(row == ids[parent]).with_for_update(key_share=True))
            deleting.execute(
                delete(ApplicationStageEvent).where(ApplicationStageEvent.application_id == first)
            )
            deleting.execute(delete(Application).where(Application.id == first))
            response = pool.submit(lambda: client.post("/applications", json=body))
            deadline = time.monotonic() + 10
            while not _waiting_for_a_lock(clean_db) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert _waiting_for_a_lock(clean_db), "the repeat application never waited"
            try:
                deleting.execute(delete(parent).where(parent.id == ids[parent]))
            except OperationalError as exc:
                pytest.fail(f"the delete could not finish: {exc.orig}")
        answer = response.result(timeout=30)

    assert answer.status_code == 404, answer.text
    assert answer.json()["detail"] == f"{parent.__name__} {ids[parent]} not found"


class _EvenMatcher:
    """Scores every pair 0.5, so ranks, and therefore insert order, follow candidate id."""

    def predict_proba(self, pairs: list[Any]) -> Any:
        return np.full((len(pairs), 2), 0.5)


def test_a_job_delete_during_a_scoring_insert_of_several_batches_does_not_deadlock(
    client: TestClient,
    clean_db: Engine,
    make_job: Factory,
    make_candidate: Factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found 2026-09-26: the two deadlocked, and the delete answered 503.

    SQLAlchemy inserts new scores 1,000 rows to a statement, and each
    statement's foreign key checks take FOR KEY SHARE on the job, so a run
    holds that lock from its first batch on. Here another run commits the
    score this one is about to insert in its second batch, and a delete of
    the job removes that row and then waits on the lock to remove the job,
    while this run's second batch waits to learn whether the row is gone.
    """
    job_id, first = make_job()["id"], make_candidate()["id"]
    with clean_db.begin() as setup:
        setup.execute(
            text(
                "INSERT INTO candidates (full_name, email, years_experience, seniority, "
                "education_level, resume_text, skills) "
                "SELECT full_name, 'pool' || n || '@example.com', years_experience, seniority, "
                "education_level, resume_text, skills "
                "FROM candidates, generate_series(1, 1000) AS n WHERE id = :id"
            ),
            {"id": first},
        )
        last = setup.scalar(select(func.max(Candidate.id)))
    monkeypatch.setattr(matches_api, "load_versioned_matcher", lambda: (_EvenMatcher(), "v1"))
    batches: list[int] = []
    deleting: dict[str, Any] = {}

    def between_batches(conn, cursor, statement, parameters, context, executemany) -> None:
        if not statement.startswith("INSERT INTO match_scores"):
            return
        batches.append(len(batches) + 1)
        if len(batches) > 1:
            return
        with clean_db.begin() as other_run:
            other_run.execute(
                insert(MatchScore).values(
                    job_id=job_id, candidate_id=last, model_version="v1", score=0.5, features={}
                )
            )
        deleting["response"] = pool.submit(lambda: client.delete(f"/jobs/{job_id}"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not deleting["response"].done():
            if _waiting_for_a_lock(clean_db):
                break
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=1) as pool:
        event.listen(Engine, "after_cursor_execute", between_batches)
        try:
            scored = client.post(f"/jobs/{job_id}/score")
        finally:
            event.remove(Engine, "after_cursor_execute", between_batches)
        deleted = deleting["response"].result(timeout=30)

    assert len(batches) >= 2, "the pool fitted in one insert, so the race never ran"
    assert deleted.status_code == 204, deleted.text
    assert scored.status_code == 404, scored.text
    assert scored.json()["detail"] == f"Job {job_id} not found"


# --- 10. an oversized body read in full before it was refused (2026-09-12) --
#
# Was: the 2 MB resume limit was checked only after Starlette had received the
# whole body. Re-measured live on 2026-09-24: a 12,583,050-byte upload was taken
# in whole before its 400, whether its length was declared or it came chunked,
# and a request declaring that length but sending no body got no answer while
# the server waited for it. These run the app as ASGI directly, so they can
# count the body bytes it reads.

BOUNDARY = "limitprobe"
MULTIPART = f"multipart/form-data; boundary={BOUNDARY}"
CHUNK = 64 * 1024
TOO_LARGE = {
    "detail": f"Request body is larger than the {MAX_REQUEST_BYTES:,}-byte limit",
    "code": "request_too_large",
}


def _upload_chunks(file_bytes: int) -> Iterator[bytes]:
    yield (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="cv.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    for start in range(0, file_bytes, CHUNK):
        yield b"a" * min(CHUNK, file_bytes - start)
    yield f"\r\n--{BOUNDARY}--\r\n".encode()


def _through_asgi(
    app: FastAPI, path: str, headers: dict[str, str], body: Iterator[bytes]
) -> tuple[int, int, dict[str, Any]]:
    """POST ``body`` straight through the ASGI app; return its status, bytes read and answer."""
    read = 0
    answer: dict[str, Any] = {"status": None, "headers": {}, "body": b""}

    async def receive() -> dict[str, Any]:
        nonlocal read
        chunk = next(body, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        read += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            answer["status"] = message["status"]
            answer["headers"] = dict(message["headers"])
        elif message["type"] == "http.response.body":
            answer["body"] += message.get("body", b"")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return answer["status"], read, answer


def _preview_path() -> str:
    return f"{get_settings().api_prefix}/candidates/resume-preview"


def test_a_body_declared_over_the_cap_is_refused_before_it_is_read(app: FastAPI) -> None:
    headers = {"content-type": MULTIPART, "content-length": str(MAX_REQUEST_BYTES + 1)}

    status, read, answer = _through_asgi(
        app, _preview_path(), headers, _upload_chunks(MAX_REQUEST_BYTES)
    )

    assert (status, read) == (413, 0)
    assert json.loads(answer["body"]) == TOO_LARGE
    assert answer["headers"][b"connection"] == b"close"


def test_a_body_sent_without_a_length_is_cut_off_once_past_the_cap(app: FastAPI) -> None:
    headers = {"content-type": MULTIPART, "transfer-encoding": "chunked"}

    status, read, answer = _through_asgi(
        app, _preview_path(), headers, _upload_chunks(12 * 1024 * 1024)
    )

    assert status == 413
    assert MAX_REQUEST_BYTES < read <= MAX_REQUEST_BYTES + CHUNK
    assert json.loads(answer["body"]) == TOO_LARGE
    assert answer["headers"][b"connection"] == b"close"


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "chunked"])
def test_a_body_of_exactly_the_cap_is_read(app: FastAPI, declared: bool) -> None:
    framing = sum(len(chunk) for chunk in _upload_chunks(0))
    headers = {"content-type": MULTIPART}
    if declared:
        headers["content-length"] = str(MAX_REQUEST_BYTES)
    else:
        headers["transfer-encoding"] = "chunked"

    status, read, answer = _through_asgi(
        app, _preview_path(), headers, _upload_chunks(MAX_REQUEST_BYTES - framing)
    )

    assert (status, read) == (400, MAX_REQUEST_BYTES)
    assert json.loads(answer["body"])["detail"] == "Resume is larger than the 2 MB limit"


@pytest.mark.parametrize(
    ("file_bytes", "detail"),
    [
        (MAX_UPLOAD_BYTES, "The extracted text is 2,097,152 characters, over the 100,000"),
        (MAX_UPLOAD_BYTES + 1, "Resume is larger than the 2 MB limit"),
    ],
    ids=["at-the-upload-limit", "one-byte-over"],
)
def test_an_upload_near_the_limit_still_gets_the_endpoints_own_answer(
    client: TestClient, file_bytes: int, detail: str
) -> None:
    """The cap leaves room for the framing around the largest file, long name and all."""
    name = "r" * 251 + ".txt"

    response = client.post(
        "/candidates/resume-preview", files={"file": (name, b"a" * file_bytes, "text/plain")}
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith(detail)


def test_a_candidate_at_every_limit_fits_under_the_cap(client: TestClient) -> None:
    """Every text field at its limit, all but the email sent as surrogate-pair escapes."""
    wide = chr(0x1F600)
    payload = {
        "full_name": wide * 200,
        "email": "a" * 64 + "@" + ".".join(["b" * 63, "c" * 63, "d" * 57, "com"]),
        "phone": wide * 40,
        "location": wide * 150,
        "current_title": wide * 200,
        "resume_text": wide * MAX_RESUME_CHARS,
        "skills": [chr(0x1F600 + n) + wide * (MAX_SKILL_CHARS - 1) for n in range(100)],
    }
    assert 1_300_000 < len(json.dumps(payload, ensure_ascii=True)) < MAX_REQUEST_BYTES

    response = _post_ascii_json(client, "/candidates", payload)

    assert response.status_code == 201, response.text[:200]


def test_a_refused_body_still_carries_cors_headers(client: TestClient) -> None:
    origin = get_settings().cors_origin_list[0]

    response = client.post(
        "/candidates/resume-preview",
        content=b"a" * (MAX_REQUEST_BYTES + 1),
        headers={"origin": origin, "content-type": MULTIPART},
    )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == origin


# --- 11. timestamps written in two formats (found 2026-09-22) ---------------
#
# Was: a scoring response's scored_at came from Python, in UTC ("...Z"), while
# every timestamp read back from the database came in the database session's
# zone, "...-04:00" on the development machine: the same instant, but not the
# same text, so a client comparing them as strings disagreed. The listing test
# in test_matching_api.py compares scored_at across the two routes as text.


def test_the_apps_database_sessions_run_in_utc(app: FastAPI) -> None:
    """Pinned directly, since a server already configured for UTC would hide a regression."""
    sessions = app.dependency_overrides[get_db]()
    try:
        assert next(sessions).scalar(text("SHOW TimeZone")) == "UTC"
    finally:
        sessions.close()


def test_every_timestamp_the_api_returns_is_utc(
    client: TestClient, make_job: Factory, make_candidate: Factory, make_application: Factory
) -> None:
    job, candidate = make_job(), make_candidate()
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    events = client.get(f"/applications/{application['id']}/stage-events").json()

    stamps = [
        *(job[field] for field in ("created_at", "updated_at")),
        *(candidate[field] for field in ("created_at", "updated_at")),
        *(application[field] for field in ("applied_at", "created_at", "updated_at")),
        *(event["occurred_at"] for event in events),
    ]

    assert len(stamps) == 8
    assert [stamp for stamp in stamps if not stamp.endswith("Z")] == []
