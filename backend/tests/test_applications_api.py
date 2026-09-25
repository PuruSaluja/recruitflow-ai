"""Integration tests for the application endpoints, against the test database."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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


def test_only_the_one_application_per_job_rule_answers_already_applied(
    client: TestClient,
    job: dict[str, Any],
    candidate: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any other constraint that refuses the row is left to the generic 409."""

    def refuse(self: Session, *args: Any, **kwargs: Any) -> None:
        raise IntegrityError(
            "INSERT INTO applications", {}, Exception('violates check constraint "ck_other"')
        )

    monkeypatch.setattr(Session, "flush", refuse)
    response = client.post(
        "/applications", json={"job_id": job["id"], "candidate_id": candidate["id"]}
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Resource conflicts with an existing record",
        "code": "conflict",
    }


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


# --- reopening ------------------------------------------------------------


def _events(client: TestClient, application_id: int) -> list[dict[str, Any]]:
    response = client.get(f"/applications/{application_id}/stage-events")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("ending", ["rejected", "withdrawn"])
def test_a_finished_application_can_be_reopened(
    client: TestClient, application: dict[str, Any], advance: Factory, ending: str
) -> None:
    advance(application["id"], "screening", ending)

    response = client.post(f"/applications/{application['id']}/reopen", json={})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "applied"
    assert client.get(f"/applications/{application['id']}").json()["status"] == "applied"


def test_reopen_records_the_real_from_status_and_who_did_it(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "screening", "interview", "rejected")

    client.post(f"/applications/{application['id']}/reopen", json={"changed_by": "  recruiter_a "})

    last = _events(client, application["id"])[-1]
    assert (last["from_status"], last["to_status"]) == ("rejected", "applied")
    assert last["changed_by"] == "recruiter_a"


def test_the_trail_stays_a_connected_chain_across_a_reopen(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    """The rejected stint is kept, not rewritten, so its duration stays answerable."""
    advance(application["id"], "screening", "withdrawn")
    client.post(f"/applications/{application['id']}/reopen")
    advance(application["id"], "screening", "interview")

    moves = [(e["from_status"], e["to_status"]) for e in _events(client, application["id"])]
    assert moves == [
        (None, "applied"),
        ("applied", "screening"),
        ("screening", "withdrawn"),
        ("withdrawn", "applied"),
        ("applied", "screening"),
        ("screening", "interview"),
    ]


def test_a_reopened_application_can_go_all_the_way_to_hired(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "rejected")
    client.post(f"/applications/{application['id']}/reopen")

    final = advance(application["id"], "screening", "interview", "offer", "hired")

    assert final["status"] == "hired"


def test_reopen_works_without_a_body(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "rejected")

    response = client.post(f"/applications/{application['id']}/reopen")

    assert response.status_code == 200, response.text
    assert _events(client, application["id"])[-1]["changed_by"] is None


def test_a_hired_application_cannot_be_reopened(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "screening", "interview", "offer", "hired")

    response = client.post(f"/applications/{application['id']}/reopen")

    assert response.status_code == 409
    assert "hired" in response.json()["detail"]


@pytest.mark.parametrize("stages", [(), ("screening",), ("screening", "interview", "offer")])
def test_a_live_application_cannot_be_reopened(
    client: TestClient, application: dict[str, Any], advance: Factory, stages: tuple[str, ...]
) -> None:
    advance(application["id"], *stages)

    response = client.post(f"/applications/{application['id']}/reopen")

    assert response.status_code == 409
    assert "has not finished" in response.json()["detail"]


def test_reopen_needs_the_requisition_to_be_open(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "rejected")
    assert client.post(f"/jobs/{application['job_id']}/close").status_code == 200

    response = client.post(f"/applications/{application['id']}/reopen")

    assert response.status_code == 409
    assert "closed" in response.json()["detail"]
    assert client.get(f"/applications/{application['id']}").json()["status"] == "rejected"


def test_a_refused_reopen_records_no_event(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "screening", "interview", "offer", "hired")
    before = len(_events(client, application["id"]))

    client.post(f"/applications/{application['id']}/reopen", json={"changed_by": "recruiter_a"})

    assert len(_events(client, application["id"])) == before


def test_the_patch_refusal_on_a_rejected_application_points_at_reopening(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "rejected")

    response = client.patch(f"/applications/{application['id']}", json={"status": "screening"})

    assert response.status_code == 409
    assert "Reopen it" in response.json()["detail"]


def test_reopening_an_unknown_application_is_404(client: TestClient) -> None:
    assert client.post("/applications/999999/reopen").status_code == 404


def test_an_unknown_field_in_the_reopen_body_is_rejected(
    client: TestClient, application: dict[str, Any], advance: Factory
) -> None:
    advance(application["id"], "rejected")

    response = client.post(f"/applications/{application['id']}/reopen", json={"status": "offer"})

    assert response.status_code == 422


@pytest.mark.parametrize(("length", "expected"), [(120, 200), (121, 422)])
def test_changed_by_on_a_reopen_is_capped_at_the_column_length(
    client: TestClient, application: dict[str, Any], advance: Factory, length: int, expected: int
) -> None:
    advance(application["id"], "rejected")

    response = client.post(
        f"/applications/{application['id']}/reopen", json={"changed_by": "x" * length}
    )

    assert response.status_code == expected, response.text


def test_a_reopen_that_lost_a_race_is_refused_and_leaves_no_event(
    client: TestClient,
    clean_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    application: dict[str, Any],
    advance: Factory,
) -> None:
    """Two recruiters reopening at once must not both succeed and double the event."""
    from sqlalchemy.orm import Session

    from app.api.v1 import applications as applications_api
    from app.models.application import Application
    from app.models.enums import ApplicationStatus

    advance(application["id"], "rejected")
    real_record = applications_api.record_stage_event

    def record_then_interfere(*args: Any, **kwargs: Any) -> Any:
        event = real_record(*args, **kwargs)
        with Session(clean_db) as other:
            other.get(Application, application["id"]).status = ApplicationStatus.APPLIED
            other.commit()
        return event

    monkeypatch.setattr(applications_api, "record_stage_event", record_then_interfere)

    response = client.post(f"/applications/{application['id']}/reopen")

    assert response.status_code == 409, response.text
    moves = [(e["from_status"], e["to_status"]) for e in _events(client, application["id"])]
    assert ("rejected", "applied") not in moves


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
