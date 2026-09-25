"""Job requisition endpoints, end to end against the test database."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.models import Application, Job, MatchScore

Factory = Callable[..., dict[str, Any]]

VALID = {
    "title": "Data Analyst",
    "description": "Own the weekly metrics review and the dashboards behind it.",
}


def _titles(client: TestClient, **params: Any) -> list[str]:
    body = client.get("/jobs", params=params).json()
    return sorted(job["title"] for job in body["items"])


# --- create / read ------------------------------------------------------------


def test_create_applies_the_documented_defaults(make_job: Factory) -> None:
    job = make_job()

    assert job["status"] == "open"
    assert job["employment_type"] == "full_time"
    assert job["seniority"] == "mid"
    assert job["education_required"] == "bachelor"
    assert job["min_years_experience"] == 0
    assert job["required_skills"] == []
    assert job["preferred_skills"] == []
    assert job["salary_min"] is None
    assert job["salary_max"] is None


def test_every_field_round_trips_through_the_database(
    client: TestClient, make_job: Factory
) -> None:
    created = make_job(
        title="Senior Data Engineer",
        department="Data",
        location="Atlanta, GA",
        employment_type="contract",
        seniority="senior",
        education_required="master",
        min_years_experience=6,
        required_skills=["SQL", "Spark"],
        preferred_skills=["dbt"],
        salary_min=120000,
        salary_max=150000,
        status="draft",
    )

    response = client.get(f"/jobs/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created
    assert Decimal(created["salary_min"]) == Decimal("120000")
    assert Decimal(created["salary_max"]) == Decimal("150000")


def test_unknown_job_is_a_404(client: TestClient) -> None:
    response = client.get("/jobs/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job 999 not found"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"title": "X"}, "title", id="title-too-short"),
        pytest.param({"description": "too short"}, "description", id="description-too-short"),
        pytest.param({"min_years_experience": -1}, "min_years_experience", id="negative-years"),
        pytest.param({"min_years_experience": 51}, "min_years_experience", id="years-over-cap"),
        pytest.param({"seniority": "wizard"}, "seniority", id="unknown-enum"),
        pytest.param({"salary_min": -5}, "salary_min", id="negative-salary"),
    ],
)
def test_invalid_create_payloads_are_rejected(
    client: TestClient, overrides: dict[str, Any], field: str
) -> None:
    response = client.post("/jobs", json={**VALID, **overrides})

    assert response.status_code == 422
    assert field in {error["loc"][-1] for error in response.json()["detail"]}
    assert client.get("/jobs").json()["total"] == 0


def test_create_requires_title_and_description(client: TestClient) -> None:
    response = client.post("/jobs", json={})

    assert response.status_code == 422
    assert {"title", "description"} <= {error["loc"][-1] for error in response.json()["detail"]}


def test_salary_band_must_not_be_inverted_on_create(client: TestClient) -> None:
    response = client.post("/jobs", json={**VALID, "salary_min": 150000, "salary_max": 100000})

    assert response.status_code == 422
    assert "salary_max must be greater than or equal to salary_min" in response.text


# --- list -------------------------------------------------------------------------


def test_list_is_newest_first_and_total_ignores_pagination(
    client: TestClient, make_job: Factory
) -> None:
    ids = [make_job(title=f"Role {n}")["id"] for n in range(5)]

    page = client.get("/jobs", params={"limit": 2, "offset": 1}).json()

    assert page["total"] == 5
    assert (page["limit"], page["offset"]) == (2, 1)
    assert [job["id"] for job in page["items"]] == list(reversed(ids))[1:3]


def test_offset_past_the_end_returns_an_empty_page(client: TestClient, make_job: Factory) -> None:
    make_job()

    page = client.get("/jobs", params={"offset": 50}).json()

    assert page["items"] == []
    assert page["total"] == 1


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"status": "bogus"}],
    ids=["limit-zero", "limit-over-cap", "negative-offset", "unknown-status"],
)
def test_invalid_list_parameters_are_rejected(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get("/jobs", params=params).status_code == 422


def test_filters_narrow_the_list_and_combine(client: TestClient, make_job: Factory) -> None:
    make_job(title="Data Analyst", department="Data", seniority="junior")
    make_job(title="Senior Data Engineer", department="Data", seniority="senior")
    make_job(title="Senior Designer", department="Design", seniority="senior")
    make_job(title="Data Intern", department="Data", seniority="intern", status="draft")

    assert _titles(client, status="draft") == ["Data Intern"]
    assert _titles(client, seniority="senior") == ["Senior Data Engineer", "Senior Designer"]
    assert _titles(client, department="Data") == [
        "Data Analyst",
        "Data Intern",
        "Senior Data Engineer",
    ]
    assert _titles(client, department="Data", seniority="senior") == ["Senior Data Engineer"]
    assert _titles(client, employment_type="contract") == []


def test_department_filter_is_an_exact_match(client: TestClient, make_job: Factory) -> None:
    make_job(department="Data")

    assert _titles(client, department="data") == []
    assert _titles(client, department="Dat") == []


def test_title_search_is_a_case_insensitive_substring(
    client: TestClient, make_job: Factory
) -> None:
    make_job(title="Senior Data Engineer")
    make_job(title="Product Designer")

    assert _titles(client, q="data eng") == ["Senior Data Engineer"]
    assert _titles(client, q="DESIGN") == ["Product Designer"]


def test_filtered_total_counts_all_matches_not_just_the_page(
    client: TestClient, make_job: Factory
) -> None:
    for n in range(3):
        make_job(title=f"Data role {n}", department="Data")
    make_job(title="Designer", department="Design")

    page = client.get("/jobs", params={"department": "Data", "limit": 1}).json()

    assert len(page["items"]) == 1
    assert page["total"] == 3


# --- update -------------------------------------------------------------------------


def test_patch_changes_only_the_fields_sent(client: TestClient, make_job: Factory) -> None:
    job = make_job(title="Data Analyst", department="Data", required_skills=["SQL"])

    response = client.patch(f"/jobs/{job['id']}", json={"title": "Analytics Engineer"})

    assert response.status_code == 200
    updated = response.json()
    assert updated["title"] == "Analytics Engineer"
    assert updated["department"] == "Data"
    assert updated["required_skills"] == ["SQL"]
    assert datetime.fromisoformat(updated["updated_at"]) > datetime.fromisoformat(job["updated_at"])


def test_empty_patch_is_a_no_op(client: TestClient, make_job: Factory) -> None:
    job = make_job()

    response = client.patch(f"/jobs/{job['id']}", json={})

    assert response.status_code == 200
    assert response.json() == job


def test_patch_rejects_unknown_fields(client: TestClient, make_job: Factory) -> None:
    job = make_job()

    response = client.patch(f"/jobs/{job['id']}", json={"salary": 90000})

    assert response.status_code == 422
    assert client.get(f"/jobs/{job['id']}").json() == job


def test_patch_validates_values(client: TestClient, make_job: Factory) -> None:
    job = make_job()

    assert client.patch(f"/jobs/{job['id']}", json={"min_years_experience": 99}).status_code == 422


def test_patch_unknown_job_is_a_404(client: TestClient) -> None:
    assert client.patch("/jobs/999", json={"title": "Anything"}).status_code == 404


def test_patch_that_inverts_the_salary_band_is_a_validation_error(
    client: TestClient, make_job: Factory
) -> None:
    """Raising only the minimum can invert a band that was valid when created."""
    job = make_job(salary_min=100000, salary_max=150000)

    response = client.patch(f"/jobs/{job['id']}", json={"salary_min": 200000})

    assert response.status_code == 422
    assert Decimal(client.get(f"/jobs/{job['id']}").json()["salary_min"]) == Decimal("100000")


# --- close / delete ---------------------------------------------------------------


def test_close_is_idempotent(client: TestClient, make_job: Factory) -> None:
    job = make_job()

    first = client.post(f"/jobs/{job['id']}/close")
    second = client.post(f"/jobs/{job['id']}/close")

    assert first.status_code == 200
    assert first.json()["status"] == "closed"
    assert second.status_code == 200
    assert second.json() == first.json()


def test_close_unknown_job_is_a_404(client: TestClient) -> None:
    assert client.post("/jobs/999/close").status_code == 404


def test_delete_removes_only_that_job(client: TestClient, make_job: Factory) -> None:
    doomed, kept = make_job(title="Doomed"), make_job(title="Kept")

    assert client.delete(f"/jobs/{doomed['id']}").status_code == 204
    assert client.get(f"/jobs/{doomed['id']}").status_code == 404
    assert client.delete(f"/jobs/{doomed['id']}").status_code == 404
    assert client.get(f"/jobs/{kept['id']}").status_code == 200


def test_delete_cascades_to_applications_and_match_scores(
    client: TestClient, clean_db: Engine, make_job: Factory, make_candidate: Factory
) -> None:
    job, candidate = make_job(), make_candidate()
    with Session(clean_db) as session:
        session.add(Application(job_id=job["id"], candidate_id=candidate["id"]))
        session.add(
            MatchScore(
                job_id=job["id"],
                candidate_id=candidate["id"],
                score=0.8,
                features={},
                model_version="test",
            )
        )
        session.commit()

    assert client.delete(f"/jobs/{job['id']}").status_code == 204

    with Session(clean_db) as session:
        assert session.scalar(select(func.count()).select_from(Application)) == 0
        assert session.scalar(select(func.count()).select_from(MatchScore)) == 0
    assert client.get(f"/candidates/{candidate['id']}").status_code == 200


# --- input caps --------------------------------------------------------------
#
# The limits are written out rather than read from the constants, so changing a
# constant without updating these tests fails here.


@pytest.mark.parametrize(("length", "expected_status"), [(20_000, 201), (20_001, 422)])
def test_description_is_capped_at_20_000_characters_on_create(
    client: TestClient, length: int, expected_status: int
) -> None:
    response = client.post("/jobs", json={**VALID, "description": "d" * length})

    assert response.status_code == expected_status
    assert client.get("/jobs").json()["total"] == (1 if expected_status == 201 else 0)


@pytest.mark.parametrize("field", ["required_skills", "preferred_skills"])
@pytest.mark.parametrize(
    ("skill", "expected_status"),
    [
        pytest.param("s" * 100, 201, id="100"),
        pytest.param("s" * 101, 422, id="101"),
        pytest.param("s" * 50_000, 422, id="50000"),
    ],
)
def test_each_skill_is_capped_at_100_characters(
    client: TestClient, field: str, skill: str, expected_status: int
) -> None:
    response = client.post("/jobs", json={**VALID, field: ["SQL", skill]})

    assert response.status_code == expected_status
    if expected_status == 201:
        assert response.json()[field] == ["SQL", skill]
    else:
        assert response.json()["detail"][0]["loc"] == ["body", field, 1]


@pytest.mark.parametrize("field", ["required_skills", "preferred_skills"])
@pytest.mark.parametrize(("count", "expected_status"), [(100, 201), (101, 422)])
def test_each_skill_list_is_capped_at_100(
    client: TestClient, field: str, count: int, expected_status: int
) -> None:
    response = client.post("/jobs", json={**VALID, field: [f"skill-{n}" for n in range(count)]})

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        pytest.param({"description": "d" * 20_001}, "description", id="description"),
        pytest.param(
            {"required_skills": ["SQL", "s" * 101]}, "required_skills", id="required-skill"
        ),
        pytest.param(
            {"required_skills": [f"skill-{n}" for n in range(101)]},
            "required_skills",
            id="required-list",
        ),
        pytest.param(
            {"preferred_skills": ["SQL", "s" * 101]}, "preferred_skills", id="preferred-skill"
        ),
        pytest.param(
            {"preferred_skills": [f"skill-{n}" for n in range(101)]},
            "preferred_skills",
            id="preferred-list",
        ),
    ],
)
def test_patch_applies_the_same_caps_and_keeps_the_stored_row(
    client: TestClient, make_job: Factory, updates: dict[str, Any], field: str
) -> None:
    job = make_job(required_skills=["Python"])

    response = client.patch(f"/jobs/{job['id']}", json=updates)

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]
    assert client.get(f"/jobs/{job['id']}").json() == job


def test_patch_accepts_values_exactly_at_the_caps(client: TestClient, make_job: Factory) -> None:
    job = make_job()
    at_the_caps = ["s" * 100] + [f"skill-{n}" for n in range(99)]

    response = client.patch(
        f"/jobs/{job['id']}",
        json={
            "description": "d" * 20_000,
            "required_skills": at_the_caps,
            "preferred_skills": at_the_caps,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["description"]) == 20_000
    assert response.json()["required_skills"] == at_the_caps
    assert response.json()["preferred_skills"] == at_the_caps


def test_a_job_stored_before_the_caps_is_still_readable(
    client: TestClient, clean_db: Engine
) -> None:
    """The caps guard what comes in, so rows already stored over them still serialise."""
    with Session(clean_db) as session:
        row = Job(
            title="Legacy Row",
            description="d" * 20_001,
            required_skills=["s" * 101],
            preferred_skills=[f"skill-{n}" for n in range(101)],
        )
        session.add(row)
        session.commit()
        job_id = row.id

    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get("/jobs").json()["items"][0]["id"] == job_id
    # Editing another field leaves the oversized ones alone rather than refusing.
    response = client.patch(f"/jobs/{job_id}", json={"location": "Atlanta, GA"})
    assert response.status_code == 200
    assert len(response.json()["description"]) == 20_001
