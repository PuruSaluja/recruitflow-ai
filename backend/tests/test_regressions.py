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

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.db.base import Base
from app.models import Candidate, Job
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
