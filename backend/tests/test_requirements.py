"""The check on a job's minimum experience, and where the API shows it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ml.corpus import JOB_MIN_YEARS
from app.models.enums import SeniorityLevel
from app.services.requirements import USUAL_MAX_MIN_YEARS, experience_requirement_note


@pytest.mark.parametrize("level", list(SeniorityLevel))
def test_a_minimum_is_flagged_only_once_it_passes_the_usual_for_its_level(
    level: SeniorityLevel,
) -> None:
    usual = USUAL_MAX_MIN_YEARS[level]

    assert experience_requirement_note(level, usual) is None
    note = experience_requirement_note(level, usual + 1)
    assert note is not None
    assert f"The minimum of {usual + 1} years' experience is above the {usual}" in note
    assert f"a {level.value} role" in note


def test_every_minimum_the_synthetic_jobs_are_drawn_with_passes_unflagged() -> None:
    """The usual ranges are the generator's own, so no generated job is flagged."""
    for level, (_, highest) in JOB_MIN_YEARS.items():
        assert USUAL_MAX_MIN_YEARS[level] == highest


def test_the_api_shows_the_note_on_a_job_whose_minimum_is_long(
    client: TestClient, make_job: Callable[..., dict[str, Any]]
) -> None:
    flagged = make_job(seniority="mid", min_years_experience=8)
    usual = make_job(seniority="mid", min_years_experience=3)

    assert len(flagged["requirement_notes"]) == 1
    assert "above the 3 a mid role usually asks for" in flagged["requirement_notes"][0]
    assert usual["requirement_notes"] == []
    listed = {job["id"]: job for job in client.get("/jobs").json()["items"]}
    assert listed[flagged["id"]]["requirement_notes"] == flagged["requirement_notes"]
    edited = client.patch(f"/jobs/{usual['id']}", json={"min_years_experience": 9}).json()
    assert len(edited["requirement_notes"]) == 1
