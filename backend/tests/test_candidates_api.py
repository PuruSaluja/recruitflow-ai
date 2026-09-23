"""Candidate endpoints and resume upload, end to end against the test database."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.models import Application, Candidate, MatchScore
from app.services.resume import MAX_RESUME_CHARS, MAX_UPLOAD_BYTES

Factory = Callable[..., dict[str, Any]]

RESUME = "Built reporting pipelines in SQL and Python for a retail team."
BULLET = "\u2022"


def _emails(client: TestClient, **params: Any) -> list[str]:
    body = client.get("/candidates", params=params).json()
    assert body["total"] == len(body["items"])
    return sorted(candidate["email"] for candidate in body["items"])


def _upload(filename: str, data: bytes, content_type: str = "text/plain") -> dict[str, Any]:
    return {"file": (filename, data, content_type)}


def _pdf(*lines: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    y = 720
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 16
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def pool(make_candidate: Factory) -> None:
    """Three candidates chosen so every filter separates them differently."""
    make_candidate(
        email="ana@example.com",
        full_name="Ana Diaz",
        current_title="Data Analyst",
        location="Atlanta, GA",
        years_experience=2,
        seniority="junior",
        education_level="bachelor",
        skills=["SQL", "Tableau"],
    )
    make_candidate(
        email="ben@example.com",
        full_name="Ben Okafor",
        current_title="Senior Data Engineer",
        location="Austin, TX",
        years_experience=8,
        seniority="senior",
        education_level="master",
        skills=["Python", "SQL", "Spark"],
    )
    make_candidate(
        email="cho@example.com",
        full_name="Cho Park",
        current_title="Product Designer",
        location="atlanta, ga",
        years_experience=5,
        seniority="mid",
        education_level="bachelor",
        skills=["Figma"],
    )


# --- create / read ------------------------------------------------------------


def test_create_applies_the_documented_defaults(make_candidate: Factory) -> None:
    candidate = make_candidate()

    assert candidate["years_experience"] == 0
    assert candidate["seniority"] == "mid"
    assert candidate["education_level"] == "bachelor"
    assert candidate["skills"] == []
    assert candidate["phone"] is None
    assert candidate["location"] is None
    assert candidate["current_title"] is None


def test_every_field_round_trips_through_the_database(
    client: TestClient, make_candidate: Factory
) -> None:
    created = make_candidate(
        full_name="Ada Lovelace",
        email="ada@example.com",
        phone="+1 404 555 0100",
        location="Atlanta, GA",
        current_title="Senior Analyst",
        years_experience=7,
        seniority="senior",
        education_level="master",
        resume_text=RESUME,
        skills=["SQL", "Python"],
    )

    response = client.get(f"/candidates/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_skills_are_trimmed_and_deduplicated_keeping_the_first_spelling(
    make_candidate: Factory,
) -> None:
    candidate = make_candidate(
        skills=[" Python ", "python", "PYTHON", "machine   learning", "", "   ", "SQL"]
    )

    assert candidate["skills"] == ["Python", "machine learning", "SQL"]


@pytest.mark.parametrize(("count", "expected_status"), [(100, 201), (101, 422)])
def test_skill_list_is_capped_at_100(client: TestClient, count: int, expected_status: int) -> None:
    payload = {
        "full_name": "Many Skills",
        "email": "many@example.com",
        "resume_text": RESUME,
        "skills": [f"skill-{n}" for n in range(count)],
    }

    assert client.post("/candidates", json=payload).status_code == expected_status


@pytest.mark.parametrize(
    ("length", "expected_status"), [(MAX_RESUME_CHARS, 201), (MAX_RESUME_CHARS + 1, 422)]
)
def test_resume_text_is_capped_at_100_000_characters_on_create(
    client: TestClient, length: int, expected_status: int
) -> None:
    payload = {"full_name": "Long Resume", "email": "long@example.com", "resume_text": "a" * length}

    response = client.post("/candidates", json=payload)

    assert response.status_code == expected_status
    assert client.get("/candidates").json()["total"] == (1 if expected_status == 201 else 0)


@pytest.mark.parametrize(
    ("skill", "expected_status"),
    [
        pytest.param("s" * 100, 201, id="100"),
        pytest.param("s" * 101, 422, id="101"),
        pytest.param("s" * 50_000, 422, id="50000"),
        pytest.param("  " + "s" * 100 + "  ", 201, id="padding-not-counted"),
        pytest.param("s" * 50 + "     " + "s" * 49, 201, id="inner-run-counted-once"),
    ],
)
def test_each_skill_is_capped_at_100_characters_after_tidying(
    client: TestClient, skill: str, expected_status: int
) -> None:
    payload = {
        "full_name": "Long Skill",
        "email": "skill@example.com",
        "resume_text": RESUME,
        "skills": ["SQL", skill],
    }

    response = client.post("/candidates", json=payload)

    assert response.status_code == expected_status
    if expected_status == 201:
        assert len(response.json()["skills"][1]) == 100
    else:
        assert response.json()["detail"][0]["loc"][-1] == "skills"
        assert "at most 100 characters" in response.json()["detail"][0]["msg"]
        assert client.get("/candidates").json()["total"] == 0


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        pytest.param({"resume_text": "a" * (MAX_RESUME_CHARS + 1)}, "resume_text", id="resume"),
        pytest.param({"skills": ["SQL", "s" * 101]}, "skills", id="skill"),
    ],
)
def test_patch_applies_the_same_length_caps_and_keeps_the_stored_row(
    client: TestClient, make_candidate: Factory, updates: dict[str, Any], field: str
) -> None:
    candidate = make_candidate(resume_text=RESUME, skills=["Python"])

    response = client.patch(f"/candidates/{candidate['id']}", json=updates)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    assert client.get(f"/candidates/{candidate['id']}").json() == candidate


def test_patch_accepts_values_exactly_at_the_caps(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate()

    response = client.patch(
        f"/candidates/{candidate['id']}",
        # PATCH checks before tidying, so the padding here must not count.
        json={"resume_text": "a" * MAX_RESUME_CHARS, "skills": ["  " + "s" * 100 + "  "]},
    )

    assert response.status_code == 200
    assert len(response.json()["resume_text"]) == MAX_RESUME_CHARS
    assert response.json()["skills"] == ["s" * 100]


def test_a_row_stored_before_the_caps_is_still_readable(
    client: TestClient, clean_db: Engine
) -> None:
    """The caps guard what comes in, so rows already stored over them still serialise."""
    with Session(clean_db) as session:
        row = Candidate(
            full_name="Legacy Row",
            email="legacy@example.com",
            resume_text="a" * (MAX_RESUME_CHARS + 1),
            skills=["s" * 101],
        )
        session.add(row)
        session.commit()
        candidate_id = row.id

    assert client.get(f"/candidates/{candidate_id}").status_code == 200
    assert client.get("/candidates").json()["items"][0]["id"] == candidate_id
    # Editing another field leaves the oversized ones alone rather than refusing.
    response = client.patch(f"/candidates/{candidate_id}", json={"location": "Atlanta, GA"})
    assert response.status_code == 200
    assert len(response.json()["resume_text"]) == MAX_RESUME_CHARS + 1


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"email": "not-an-email"}, "email", id="bad-email"),
        pytest.param({"full_name": "A"}, "full_name", id="name-too-short"),
        pytest.param({"resume_text": "too short"}, "resume_text", id="resume-too-short"),
        pytest.param({"years_experience": 61}, "years_experience", id="years-over-cap"),
        pytest.param({"seniority": "wizard"}, "seniority", id="unknown-enum"),
    ],
)
def test_invalid_create_payloads_are_rejected(
    client: TestClient, overrides: dict[str, Any], field: str
) -> None:
    payload = {
        "full_name": "Valid Name",
        "email": "valid@example.com",
        "resume_text": RESUME,
        **overrides,
    }

    response = client.post("/candidates", json=payload)

    assert response.status_code == 422
    assert field in {error["loc"][-1] for error in response.json()["detail"]}
    assert client.get("/candidates").json()["total"] == 0


def test_duplicate_email_is_a_409_not_a_500(client: TestClient, make_candidate: Factory) -> None:
    make_candidate(email="taken@example.com")

    response = client.post(
        "/candidates",
        json={"full_name": "Someone Else", "email": "taken@example.com", "resume_text": RESUME},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "conflict"
    assert client.get("/candidates").json()["total"] == 1


def test_unknown_candidate_is_a_404(client: TestClient) -> None:
    response = client.get("/candidates/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidate 999 not found"


# --- list ---------------------------------------------------------------------


def test_list_is_newest_first_and_total_ignores_pagination(
    client: TestClient, make_candidate: Factory
) -> None:
    ids = [make_candidate()["id"] for _ in range(5)]

    page = client.get("/candidates", params={"limit": 2, "offset": 2}).json()

    assert page["total"] == 5
    assert [candidate["id"] for candidate in page["items"]] == list(reversed(ids))[2:4]


@pytest.mark.usefixtures("pool")
def test_enum_filters(client: TestClient) -> None:
    assert _emails(client, seniority="senior") == ["ben@example.com"]
    assert _emails(client, education_level="bachelor") == ["ana@example.com", "cho@example.com"]


@pytest.mark.usefixtures("pool")
def test_experience_range_bounds_are_inclusive(client: TestClient) -> None:
    assert _emails(client, min_years_experience=5) == ["ben@example.com", "cho@example.com"]
    assert _emails(client, max_years_experience=5) == ["ana@example.com", "cho@example.com"]
    assert _emails(client, min_years_experience=5, max_years_experience=5) == ["cho@example.com"]


@pytest.mark.usefixtures("pool")
def test_location_is_a_case_insensitive_substring(client: TestClient) -> None:
    assert _emails(client, location="ATLANTA") == ["ana@example.com", "cho@example.com"]
    assert _emails(client, location="tx") == ["ben@example.com"]


@pytest.mark.usefixtures("pool")
def test_text_search_covers_name_email_and_title(client: TestClient) -> None:
    assert _emails(client, q="okafor") == ["ben@example.com"]
    assert _emails(client, q="cho@") == ["cho@example.com"]
    assert _emails(client, q="data") == ["ana@example.com", "ben@example.com"]


@pytest.mark.usefixtures("pool")
def test_skill_filter_is_case_insensitive_and_requires_every_listed_skill(
    client: TestClient,
) -> None:
    assert _emails(client, skill="sql") == ["ana@example.com", "ben@example.com"]
    assert _emails(client, skill=["SQL", "python"]) == ["ben@example.com"]
    assert _emails(client, skill=["SQL", "Figma"]) == []


@pytest.mark.usefixtures("pool")
def test_skill_filter_matches_whole_skills_not_substrings(client: TestClient) -> None:
    assert _emails(client, skill="SQ") == []


@pytest.mark.usefixtures("pool")
def test_blank_skill_parameters_are_ignored(client: TestClient) -> None:
    assert len(_emails(client, skill=["  "])) == 3


@pytest.mark.usefixtures("pool")
def test_filters_combine(client: TestClient) -> None:
    assert _emails(client, location="atlanta", skill="sql") == ["ana@example.com"]
    assert _emails(client, education_level="bachelor", min_years_experience=3) == [
        "cho@example.com"
    ]


@pytest.mark.usefixtures("pool")
def test_filtered_total_counts_all_matches_not_just_the_page(client: TestClient) -> None:
    page = client.get("/candidates", params={"location": "atlanta", "limit": 1}).json()

    assert len(page["items"]) == 1
    assert page["total"] == 2


@pytest.mark.parametrize(
    "params",
    [
        {"min_years_experience": 61},
        {"max_years_experience": -1},
        {"seniority": "bogus"},
        {"limit": 0},
    ],
    ids=["min-years-over-cap", "negative-max-years", "unknown-seniority", "limit-zero"],
)
def test_invalid_list_parameters_are_rejected(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get("/candidates", params=params).status_code == 422


# --- update -------------------------------------------------------------------


def test_patch_changes_only_the_fields_sent(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate(current_title="Analyst", skills=["SQL"])

    response = client.patch(f"/candidates/{candidate['id']}", json={"years_experience": 4})

    assert response.status_code == 200
    updated = response.json()
    assert updated["years_experience"] == 4
    assert updated["current_title"] == "Analyst"
    assert updated["skills"] == ["SQL"]


def test_patch_tidies_skills_the_same_way_create_does(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate()

    response = client.patch(
        f"/candidates/{candidate['id']}", json={"skills": ["dbt", " DBT ", "Looker"]}
    )

    assert response.json()["skills"] == ["dbt", "Looker"]


def test_patch_clears_a_nullable_field_with_null(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate(phone="+1 404 555 0100")

    response = client.patch(f"/candidates/{candidate['id']}", json={"phone": None})

    assert response.status_code == 200
    assert response.json()["phone"] is None


def test_patch_to_an_email_already_in_use_is_a_409(
    client: TestClient, make_candidate: Factory
) -> None:
    make_candidate(email="taken@example.com")
    candidate = make_candidate(email="free@example.com")

    response = client.patch(f"/candidates/{candidate['id']}", json={"email": "taken@example.com"})

    assert response.status_code == 409
    assert client.get(f"/candidates/{candidate['id']}").json()["email"] == "free@example.com"


def test_empty_patch_is_a_no_op(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate()

    response = client.patch(f"/candidates/{candidate['id']}", json={})

    assert response.status_code == 200
    assert response.json() == candidate


def test_patch_rejects_unknown_fields(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate()

    assert client.patch(f"/candidates/{candidate['id']}", json={"age": 30}).status_code == 422


def test_patch_unknown_candidate_is_a_404(client: TestClient) -> None:
    assert client.patch("/candidates/999", json={"full_name": "Anyone"}).status_code == 404


# --- delete -------------------------------------------------------------------


def test_delete_removes_only_that_candidate(client: TestClient, make_candidate: Factory) -> None:
    doomed, kept = make_candidate(), make_candidate()

    assert client.delete(f"/candidates/{doomed['id']}").status_code == 204
    assert client.get(f"/candidates/{doomed['id']}").status_code == 404
    assert client.delete(f"/candidates/{doomed['id']}").status_code == 404
    assert client.get(f"/candidates/{kept['id']}").status_code == 200


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

    assert client.delete(f"/candidates/{candidate['id']}").status_code == 204

    with Session(clean_db) as session:
        assert session.scalar(select(func.count()).select_from(Application)) == 0
        assert session.scalar(select(func.count()).select_from(MatchScore)) == 0
    assert client.get(f"/jobs/{job['id']}").status_code == 200


# --- resume upload ------------------------------------------------------------


def test_uploading_a_text_resume_replaces_the_stored_text_with_cleaned_text(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate(resume_text=RESUME)
    data = f"EXPERIENCE\n{BULLET} Built ETL jobs\nPage 1 of 1\nTook respon-\nsibility for BI."

    response = client.put(
        f"/candidates/{candidate['id']}/resume", files=_upload("cv.txt", data.encode())
    )

    assert response.status_code == 200
    assert response.json()["resume_text"] == (
        "EXPERIENCE\n- Built ETL jobs\n\nTook responsibility for BI."
    )
    assert response.json()["full_name"] == candidate["full_name"]


def test_uploading_a_pdf_resume(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate()
    pdf = _pdf("Led the analytics platform migration")

    response = client.put(
        f"/candidates/{candidate['id']}/resume",
        files=_upload("cv.pdf", pdf, "application/pdf"),
    )

    assert response.status_code == 200
    assert "Led the analytics platform migration" in response.json()["resume_text"]


def test_re_uploading_the_same_file_gives_the_same_result(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate()
    url = f"/candidates/{candidate['id']}/resume"
    files = _upload("cv.txt", b"Built   reporting pipelines in SQL for a retail team.")

    first = client.put(url, files=files)
    second = client.put(url, files=files)

    assert first.status_code == second.status_code == 200
    assert second.json()["resume_text"] == first.json()["resume_text"]


def test_stored_resume_is_served_as_plain_text(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate(resume_text=RESUME)

    response = client.get(f"/candidates/{candidate['id']}/resume")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == RESUME


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        pytest.param("cv.exe", b"MZ executable bytes here", "Unsupported resume format", id="exe"),
        pytest.param("cv.txt", b"", "empty", id="empty"),
        pytest.param("cv.txt", b"Too short", "No usable text", id="too-short"),
        pytest.param("cv.docx", b"PK\x03\x04zip-body", "DOCX", id="docx"),
        pytest.param(
            "cv.txt",
            b"a" * (MAX_RESUME_CHARS + 1),
            "over the 100,000-character limit",
            id="too-long",
        ),
    ],
)
def test_rejected_upload_is_a_400_and_keeps_the_stored_resume(
    client: TestClient, make_candidate: Factory, filename: str, data: bytes, message: str
) -> None:
    candidate = make_candidate(resume_text=RESUME)

    response = client.put(f"/candidates/{candidate['id']}/resume", files=_upload(filename, data))

    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert client.get(f"/candidates/{candidate['id']}/resume").text == RESUME


def test_oversized_upload_is_rejected(client: TestClient, make_candidate: Factory) -> None:
    candidate = make_candidate()
    data = b"a" * (MAX_UPLOAD_BYTES + 1)

    response = client.put(f"/candidates/{candidate['id']}/resume", files=_upload("cv.txt", data))

    assert response.status_code == 400
    assert "2 MB" in response.json()["detail"]


def test_upload_to_an_unknown_candidate_is_a_404(client: TestClient) -> None:
    response = client.put("/candidates/999/resume", files=_upload("cv.txt", RESUME.encode()))

    assert response.status_code == 404


def test_resume_of_an_unknown_candidate_is_a_404(client: TestClient) -> None:
    assert client.get("/candidates/999/resume").status_code == 404


# --- resume preview -----------------------------------------------------------


def test_preview_parses_without_storing_anything(
    client: TestClient, make_candidate: Factory
) -> None:
    candidate = make_candidate(resume_text=RESUME)
    expected_text = "Shipped a forecasting model\n\nin Python."

    response = client.post(
        "/candidates/resume-preview",
        files=_upload("draft.txt", b"Shipped   a  forecasting model\n\n\n\nin Python."),
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "draft.txt",
        "characters": len(expected_text),
        "words": 6,
        "text": expected_text,
    }
    assert client.get("/candidates").json()["total"] == 1
    assert client.get(f"/candidates/{candidate['id']}/resume").text == RESUME


def test_preview_rejects_unusable_files(client: TestClient) -> None:
    response = client.post("/candidates/resume-preview", files=_upload("x.exe", b"MZ nope"))

    assert response.status_code == 400
