"""Integration tests for the application stage-history audit trail."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

Factory = Callable[..., dict[str, Any]]


@pytest.fixture
def job(make_job: Factory) -> dict[str, Any]:
    return make_job()


@pytest.fixture
def candidate(make_candidate: Factory) -> dict[str, Any]:
    return make_candidate()


def _events(client: TestClient, application_id: int) -> list[dict[str, Any]]:
    response = client.get(f"/applications/{application_id}/stage-events")
    assert response.status_code == 200, response.text
    return response.json()


# --- opening event ----------------------------------------------------------


def test_creating_an_application_records_the_opening_event(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any]
) -> None:
    response = client.post(
        "/applications", json={"job_id": job["id"], "candidate_id": candidate["id"]}
    )
    application = response.json()

    events = _events(client, application["id"])
    assert len(events) == 1
    assert events[0]["from_status"] is None
    assert events[0]["to_status"] == "applied"
    assert events[0]["changed_by"] is None
    assert events[0]["occurred_at"] == application["applied_at"]


def test_create_records_who_opened_it(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any]
) -> None:
    response = client.post(
        "/applications",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "changed_by": "recruiter_a"},
    )
    application = response.json()

    events = _events(client, application["id"])
    assert events[0]["changed_by"] == "recruiter_a"


def test_changed_by_on_create_is_not_stored_on_the_application_itself(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any]
) -> None:
    response = client.post(
        "/applications",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "changed_by": "recruiter_a"},
    )
    assert "changed_by" not in response.json()


# --- moves --------------------------------------------------------------


def test_each_move_appends_an_event_with_from_and_to(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    application_id = application["id"]

    client.patch(f"/applications/{application_id}", json={"status": "screening"})
    client.patch(f"/applications/{application_id}", json={"status": "interview"})

    events = _events(client, application_id)
    assert [(e["from_status"], e["to_status"]) for e in events] == [
        (None, "applied"),
        ("applied", "screening"),
        ("screening", "interview"),
    ]


def test_events_are_ordered_oldest_first(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], advance: Factory
) -> None:
    response = client.post(
        "/applications", json={"job_id": job["id"], "candidate_id": candidate["id"]}
    )
    application_id = response.json()["id"]
    advance(application_id, "screening", "interview", "offer", "hired")

    events = _events(client, application_id)
    occurred = [e["occurred_at"] for e in events]
    assert occurred == sorted(occurred)
    assert [e["to_status"] for e in events] == [
        "applied",
        "screening",
        "interview",
        "offer",
        "hired",
    ]


def test_a_move_records_who_made_it(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    client.patch(
        f"/applications/{application['id']}",
        json={"status": "screening", "changed_by": "recruiter_b"},
    )

    events = _events(client, application["id"])
    assert events[-1]["changed_by"] == "recruiter_b"


def test_changed_by_on_a_patch_is_not_stored_on_the_application(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    response = client.patch(
        f"/applications/{application['id']}",
        json={"status": "screening", "changed_by": "recruiter_b"},
    )
    assert "changed_by" not in response.json()


# --- moves that do not happen record no event -------------------------------


def test_resending_the_current_status_records_no_new_event(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    client.patch(f"/applications/{application['id']}", json={"status": "applied"})

    assert len(_events(client, application["id"])) == 1


def test_an_illegal_move_records_no_event(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    response = client.patch(f"/applications/{application['id']}", json={"status": "hired"})
    assert response.status_code == 409

    assert len(_events(client, application["id"])) == 1


def test_a_field_only_patch_records_no_event(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    response = client.patch(f"/applications/{application['id']}", json={"notes": "Strong SQL"})
    assert response.status_code == 200

    assert len(_events(client, application["id"])) == 1


def test_changed_by_alone_with_no_status_move_is_a_no_op(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    """changed_by has nothing to attach to without an accompanying status move."""
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    response = client.patch(
        f"/applications/{application['id']}", json={"changed_by": "recruiter_c"}
    )
    assert response.status_code == 200
    assert response.json()["updated_at"] == application["updated_at"]

    assert len(_events(client, application["id"])) == 1


# --- lifecycle and errors ----------------------------------------------------


def test_deleting_an_application_removes_its_stage_events(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any], make_application: Factory
) -> None:
    application = make_application(job_id=job["id"], candidate_id=candidate["id"])
    assert client.delete(f"/applications/{application['id']}").status_code == 204
    assert client.get(f"/applications/{application['id']}/stage-events").status_code == 404


def test_stage_events_for_an_unknown_application_is_404(client: TestClient) -> None:
    response = client.get("/applications/999999/stage-events")
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]
