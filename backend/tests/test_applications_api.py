"""Integration tests for the application endpoints, against the test database."""

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


@pytest.fixture
def application(job: dict[str, Any], candidate: dict[str, Any], make_application: Factory):
    return make_application(job_id=job["id"], candidate_id=candidate["id"])


# --- creating -------------------------------------------------------------


def test_create_starts_the_application_at_applied(application: dict[str, Any]) -> None:
    assert application["status"] == "applied"
    assert application["source"] is None
    assert application["notes"] is None
    assert application["applied_at"]


def test_create_keeps_the_source_it_was_given(
    client: TestClient, job: dict[str, Any], candidate: dict[str, Any]
) -> None:
    response = client.post(
        "/applications",
        json={"job_id": job["id"], "candidate_id": candidate["id"], "source": "referral"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["source"] == "referral"


def test_the_same_candidate_cannot_apply_twice(
    client: TestClient, application: dict[str, Any]
) -> None:
    response = client.post(
        "/applications",
        json={"job_id": application["job_id"], "candidate_id": application["candidate_id"]},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert str(application["candidate_id"]) in detail
    assert str(application["job_id"]) in detail


def test_a_candidate_may_apply_to_a_second_job(
    client: TestClient, make_job: Factory, application: dict[str, Any]
) -> None:
    other = make_job(title="Second Role")
    response = client.post(
        "/applications",
        json={"job_id": other["id"], "candidate_id": application["candidate_id"]},
    )
    assert response.status_code == 201, response.text


def test_unknown_job_is_404(client: TestClient, candidate: dict[str, Any]) -> None:
    response = client.post(
        "/applications", json={"job_id": 999999, "candidate_id": candidate["id"]}
    )
    assert response.status_code == 404
    assert "Job 999999" in response.json()["detail"]


def test_unknown_candidate_is_404(client: TestClient, job: dict[str, Any]) -> None:
    response = client.post("/applications", json={"job_id": job["id"], "candidate_id": 999999})
    assert response.status_code == 404
    assert "Candidate 999999" in response.json()["detail"]


@pytest.mark.parametrize("bad_id", [0, -1])
def test_a_non_positive_id_is_rejected_as_a_payload_error(
    client: TestClient, candidate: dict[str, Any], bad_id: int
) -> None:
    """422 about the payload, not a 404 claiming the job does not exist."""
    response = client.post(
        "/applications", json={"job_id": bad_id, "candidate_id": candidate["id"]}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [{}, {"job_id": 1}, {"candidate_id": 1}])
def test_missing_ids_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/applications", json=payload).status_code == 422


@pytest.mark.parametrize("job_status", ["draft", "paused", "closed"])
def test_only_an_open_requisition_accepts_applications(
    client: TestClient, make_job: Factory, candidate: dict[str, Any], job_status: str
) -> None:
    closed_job = make_job(status=job_status)
    response = client.post(
        "/applications", json={"job_id": closed_job["id"], "candidate_id": candidate["id"]}
    )
    assert response.status_code == 409
    assert job_status in response.json()["detail"]


def test_closing_a_job_does_not_freeze_its_existing_applications(
    client: TestClient, application: dict[str, Any]
) -> None:
    assert client.post(f"/jobs/{application['job_id']}/close").status_code == 200
    response = client.patch(f"/applications/{application['id']}", json={"status": "screening"})
    assert response.status_code == 200, response.text


# --- pipeline moves -------------------------------------------------------


def test_the_full_path_to_hired(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    final = advance(application["id"], "screening", "interview", "offer", "hired")
    assert final["status"] == "hired"


@pytest.mark.parametrize(
    ("stages", "target"),
    [
        ((), "interview"),
        ((), "offer"),
        ((), "hired"),
        (("screening",), "offer"),
        (("screening", "interview"), "hired"),
    ],
)
def test_skipping_a_stage_is_refused(
    client: TestClient,
    application: dict[str, Any],
    advance: Factory,
    stages: tuple[str, ...],
    target: str,
) -> None:
    advance(application["id"], *stages)
    response = client.patch(f"/applications/{application['id']}", json={"status": target})
    assert response.status_code == 409
    assert target in response.json()["detail"]


def test_the_refusal_says_which_stages_are_reachable(
    client: TestClient, application: dict[str, Any]
) -> None:
    response = client.patch(f"/applications/{application['id']}", json={"status": "hired"})
    detail = response.json()["detail"]
    for reachable in ("screening", "rejected", "withdrawn"):
        assert reachable in detail


def test_moving_backwards_is_refused(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "screening", "interview")
    response = client.patch(f"/applications/{application['id']}", json={"status": "screening"})
    assert response.status_code == 409


@pytest.mark.parametrize("final_status", ["hired", "rejected", "withdrawn"])
def test_a_finished_application_cannot_move_again(
    client: TestClient, application: dict[str, Any], advance: Factory, final_status: str
) -> None:
    if final_status == "hired":
        advance(application["id"], "screening", "interview", "offer", "hired")
    else:
        advance(application["id"], final_status)
    response = client.patch(f"/applications/{application['id']}", json={"status": "screening"})
    assert response.status_code == 409
    assert "no longer change status" in response.json()["detail"]


@pytest.mark.parametrize("stages", [(), ("screening",), ("screening", "interview")])
@pytest.mark.parametrize("ending", ["rejected", "withdrawn"])
def test_rejection_and_withdrawal_work_from_any_live_stage(
    application: dict[str, Any], advance: Factory, stages: tuple[str, ...], ending: str
) -> None:
    advance(application["id"], *stages, ending)


def test_resending_the_current_status_changes_nothing(
    client: TestClient, application: dict[str, Any]
) -> None:
    """A retried request must not fail, and must not bump updated_at."""
    response = client.patch(f"/applications/{application['id']}", json={"status": "applied"})
    assert response.status_code == 200
    assert response.json()["updated_at"] == application["updated_at"]


# --- payload validation ---------------------------------------------------


def test_a_null_status_is_rejected(client: TestClient, application: dict[str, Any]) -> None:
    response = client.patch(f"/applications/{application['id']}", json={"status": None})
    assert response.status_code == 422
    assert "status" in str(response.json()["detail"])


def test_an_unknown_field_is_rejected(client: TestClient, application: dict[str, Any]) -> None:
    response = client.patch(f"/applications/{application['id']}", json={"stage": "x"})
    assert response.status_code == 422


def test_an_invalid_status_value_is_rejected(
    client: TestClient, application: dict[str, Any]
) -> None:
    assert (
        client.patch(f"/applications/{application['id']}", json={"status": "promoted"}).status_code
        == 422
    )


def test_notes_and_source_can_be_set_and_cleared(
    client: TestClient, application: dict[str, Any]
) -> None:
    url = f"/applications/{application['id']}"
    assert client.patch(url, json={"notes": "Strong SQL"}).json()["notes"] == "Strong SQL"
    assert client.patch(url, json={"notes": None}).json()["notes"] is None
    assert client.patch(url, json={"source": "careers page"}).json()["source"] == "careers page"
    assert client.patch(url, json={"source": None}).json()["source"] is None


def test_an_overlong_source_is_rejected(client: TestClient, application: dict[str, Any]) -> None:
    response = client.patch(f"/applications/{application['id']}", json={"source": "x" * 81})
    assert response.status_code == 422


def test_fetching_an_unknown_application_is_404(client: TestClient) -> None:
    assert client.get("/applications/999999").status_code == 404


def test_patching_an_unknown_application_is_404(client: TestClient) -> None:
    assert client.patch("/applications/999999", json={"notes": "x"}).status_code == 404


# --- listing --------------------------------------------------------------


def test_listing_is_newest_first(
    client: TestClient, job: dict[str, Any], make_candidate: Factory, make_application: Factory
) -> None:
    created = [
        make_application(job_id=job["id"], candidate_id=make_candidate()["id"])["id"]
        for _ in range(3)
    ]
    listed = [item["id"] for item in client.get("/applications").json()["items"]]
    assert listed == sorted(created, reverse=True)


def test_filters_narrow_the_list(
    client: TestClient,
    make_job: Factory,
    make_candidate: Factory,
    make_application: Factory,
) -> None:
    job_a, job_b = make_job(title="Role A"), make_job(title="Role B")
    one, two = make_candidate(), make_candidate()
    make_application(job_id=job_a["id"], candidate_id=one["id"], source="referral")
    make_application(job_id=job_a["id"], candidate_id=two["id"])
    make_application(job_id=job_b["id"], candidate_id=one["id"])

    def total(**params: Any) -> int:
        response = client.get("/applications", params=params)
        assert response.status_code == 200, response.text
        return response.json()["total"]

    assert total() == 3
    assert total(job_id=job_a["id"]) == 2
    assert total(job_id=job_b["id"]) == 1
    assert total(candidate_id=one["id"]) == 2
    assert total(status="applied") == 3
    assert total(status="hired") == 0
    assert total(source="referral") == 1
    assert total(job_id=job_a["id"], candidate_id=two["id"]) == 1


def test_the_status_filter_follows_a_move(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "screening")
    assert client.get("/applications", params={"status": "applied"}).json()["total"] == 0
    assert client.get("/applications", params={"status": "screening"}).json()["total"] == 1


def test_total_ignores_pagination_but_items_do_not(
    client: TestClient, job: dict[str, Any], make_candidate: Factory, make_application: Factory
) -> None:
    for _ in range(3):
        make_application(job_id=job["id"], candidate_id=make_candidate()["id"])
    page = client.get("/applications", params={"limit": 2}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["limit"] == 2


@pytest.mark.parametrize(
    "params",
    [{"status": "promoted"}, {"job_id": 0}, {"candidate_id": -1}, {"limit": 0}, {"offset": -1}],
)
def test_invalid_query_parameters_are_rejected(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get("/applications", params=params).status_code == 422


# --- deleting -------------------------------------------------------------


def test_delete_removes_the_application(client: TestClient, application: dict[str, Any]) -> None:
    assert client.delete(f"/applications/{application['id']}").status_code == 204
    assert client.get(f"/applications/{application['id']}").status_code == 404


def test_deleting_an_unknown_application_is_404(client: TestClient) -> None:
    assert client.delete("/applications/999999").status_code == 404


def test_deleting_the_candidate_removes_their_applications(
    client: TestClient, application: dict[str, Any]
) -> None:
    assert client.delete(f"/candidates/{application['candidate_id']}").status_code == 204
    assert client.get("/applications").json()["total"] == 0


def test_deleting_the_job_removes_its_applications(
    client: TestClient, application: dict[str, Any]
) -> None:
    assert client.delete(f"/jobs/{application['job_id']}").status_code == 204
    assert client.get("/applications").json()["total"] == 0


def test_deleting_an_application_leaves_the_job_and_candidate(
    client: TestClient, application: dict[str, Any]
) -> None:
    client.delete(f"/applications/{application['id']}")
    assert client.get(f"/jobs/{application['job_id']}").status_code == 200
    assert client.get(f"/candidates/{application['candidate_id']}").status_code == 200
