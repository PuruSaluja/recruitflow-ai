"""Pins for the bugs found by probing the API with awkward input on 2026-09-10.

The pins for each bug fail when that bug's fix is reverted. Alongside them sit
guards against over-correcting -- nullable fields must still accept null -- which
are expected to pass either way. Kept together so the record of what broke
stays readable in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.api.v1 import applications as applications_api
from app.db.base import Base
from app.models import Candidate, Job
from app.models.application import Application
from app.models.enums import ApplicationStatus
from app.schemas.candidate import CandidateUpdate
from app.schemas.job import JobUpdate

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
    ],
)
def test_a_plausible_but_absent_id_is_still_a_404(
    client: TestClient, method: str, template: str
) -> None:
    """Guards against over-correcting: gt=0 must not swallow genuine 404s."""
    response = getattr(client, method)(template.format(id=999999))

    assert response.status_code == 404
