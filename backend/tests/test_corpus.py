"""Unit tests for the synthetic corpus generator. No database needed."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from collections import Counter
from statistics import mean

import pytest

from app.config import REPO_ROOT
from app.ml.corpus import (
    CANDIDATE_YEARS,
    COMMUNITY_RATE,
    FAMILIES,
    JOB_MIN_YEARS,
    SyntheticCandidate,
    SyntheticJob,
    generate_candidates,
    generate_jobs,
)
from app.ml.taxonomy import SKILLS, canonical_skill, normalise_skills
from app.models.enums import EmploymentType, SeniorityLevel
from app.schemas.candidate import CandidateCreate
from app.schemas.job import JobCreate

SEED = 7
FAMILY_BY_KEY = {family.key: family for family in FAMILIES}


@pytest.fixture(scope="module")
def jobs() -> list[SyntheticJob]:
    return generate_jobs(300, seed=SEED)


@pytest.fixture(scope="module")
def candidates() -> list[SyntheticCandidate]:
    return generate_candidates(3000, seed=SEED)


def _of_gender(candidates: list[SyntheticCandidate], gender: str) -> list[SyntheticCandidate]:
    return [candidate for candidate in candidates if candidate.gender == gender]


# --- role families ----------------------------------------------------------


def test_family_keys_are_unique() -> None:
    assert len(FAMILY_BY_KEY) == len(FAMILIES)


def test_every_family_skill_is_a_canonical_taxonomy_skill() -> None:
    names = {skill.name for skill in SKILLS}
    for family in FAMILIES:
        assert set(family.core_skills) | set(family.adjacent_skills) <= names, family.key


def test_no_family_lists_a_skill_as_both_core_and_adjacent() -> None:
    for family in FAMILIES:
        assert not set(family.core_skills) & set(family.adjacent_skills), family.key


# --- reproducibility --------------------------------------------------------


def test_the_same_seed_gives_the_same_corpus_and_another_seed_does_not() -> None:
    assert generate_jobs(20, seed=SEED) == generate_jobs(20, seed=SEED)
    assert generate_candidates(50, seed=SEED) == generate_candidates(50, seed=SEED)
    assert generate_jobs(20, seed=SEED) != generate_jobs(20, seed=SEED + 1)
    assert generate_candidates(50, seed=SEED) != generate_candidates(50, seed=SEED + 1)


def test_generation_does_not_depend_on_hash_randomisation() -> None:
    """Set iteration order changes from one process to the next; the corpus must not."""
    script = (
        "import dataclasses, json\n"
        "from app.ml.corpus import generate_candidates, generate_jobs\n"
        "rows = generate_jobs(30, seed=7) + generate_candidates(60, seed=7)\n"
        "print(json.dumps([dataclasses.asdict(row) for row in rows], default=sorted))\n"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT / "backend")}
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


# --- jobs -------------------------------------------------------------------


def test_every_job_is_a_valid_create_request(jobs: list[SyntheticJob]) -> None:
    for job in jobs:
        JobCreate.model_validate(job.api_payload())


def test_a_job_draws_its_skills_from_its_family(jobs: list[SyntheticJob]) -> None:
    for job in jobs:
        family = FAMILY_BY_KEY[job.family]
        assert 3 <= len(job.required_skills) <= 5
        assert set(job.required_skills) <= set(family.core_skills)
        assert 2 <= len(job.preferred_skills) <= 3
        assert set(job.preferred_skills) <= set(family.adjacent_skills)


def test_a_job_asks_for_experience_that_fits_its_level(jobs: list[SyntheticJob]) -> None:
    for job in jobs:
        low, high = JOB_MIN_YEARS[job.seniority]
        assert low <= job.min_years_experience <= high


def test_the_internships_are_exactly_the_intern_level_jobs(jobs: list[SyntheticJob]) -> None:
    for job in jobs:
        is_internship = job.employment_type is EmploymentType.INTERNSHIP
        assert is_internship == (job.seniority is SeniorityLevel.INTERN)


def test_every_family_and_level_appears_among_the_jobs(jobs: list[SyntheticJob]) -> None:
    assert {job.family for job in jobs} == set(FAMILY_BY_KEY)
    assert {job.seniority for job in jobs} == set(SeniorityLevel)


# --- candidates -------------------------------------------------------------


def test_every_candidate_is_a_valid_create_request(candidates: list[SyntheticCandidate]) -> None:
    for candidate in candidates:
        CandidateCreate.model_validate(candidate.api_payload())


def test_candidate_emails_are_unique(candidates: list[SyntheticCandidate]) -> None:
    assert len({candidate.email for candidate in candidates}) == len(candidates)


def test_demographics_and_true_skills_stay_out_of_the_api_payload(
    candidates: list[SyntheticCandidate],
) -> None:
    for candidate in candidates:
        assert (
            not {"gender", "age_band", "true_skills", "family", "ref"}
            & candidate.api_payload().keys()
        )


def test_a_candidates_experience_fits_their_level(candidates: list[SyntheticCandidate]) -> None:
    for candidate in candidates:
        low, high = CANDIDATE_YEARS[candidate.seniority]
        assert low <= candidate.years_experience <= high


def test_listed_skills_are_true_skills_with_some_left_out_and_some_aliased(
    candidates: list[SyntheticCandidate],
) -> None:
    left_out = aliased = 0
    for candidate in candidates:
        listed = normalise_skills(candidate.skills)
        assert len(listed) == len(candidate.skills), candidate.ref
        assert set(listed) <= candidate.true_skills, candidate.ref
        left_out += len(candidate.true_skills) - len(listed)
        aliased += sum(canonical_skill(spelling) != spelling for spelling in candidate.skills)
    assert left_out > 0
    assert aliased > 0


# --- demographics -----------------------------------------------------------


def test_the_pool_has_every_gender_and_age_band(candidates: list[SyntheticCandidate]) -> None:
    assert {candidate.gender for candidate in candidates} == {"woman", "man", "nonbinary"}
    assert {candidate.age_band for candidate in candidates} == {"under_40", "40_and_over"}


def test_qualifications_are_distributed_alike_across_genders(
    candidates: list[SyntheticCandidate],
) -> None:
    women, men = _of_gender(candidates, "woman"), _of_gender(candidates, "man")
    for attribute in ("family", "seniority", "education_level"):
        among_women = Counter(getattr(candidate, attribute) for candidate in women)
        among_men = Counter(getattr(candidate, attribute) for candidate in men)
        for value in among_women | among_men:
            gap = among_women[value] / len(women) - among_men[value] / len(men)
            assert abs(gap) < 0.06, (attribute, value, gap)
    skills_gap = mean(len(c.true_skills) for c in women) - mean(len(c.true_skills) for c in men)
    assert abs(skills_gap) < 0.25
    years_gap = mean(c.years_experience for c in women) - mean(c.years_experience for c in men)
    assert abs(years_gap) < 0.6


def test_age_band_follows_years_of_experience(candidates: list[SyntheticCandidate]) -> None:
    early = [candidate for candidate in candidates if candidate.years_experience <= 3]
    late = [candidate for candidate in candidates if candidate.years_experience >= 15]
    assert all(candidate.age_band == "under_40" for candidate in early)
    assert sum(candidate.age_band == "40_and_over" for candidate in late) > len(late) / 2


def test_the_women_in_tech_phrase_appears_only_in_womens_resumes(
    candidates: list[SyntheticCandidate],
) -> None:
    flagged = [c for c in candidates if "women" in c.resume_text.casefold()]
    assert flagged
    assert {candidate.gender for candidate in flagged} == {"woman"}


@pytest.mark.parametrize("gender", ["woman", "man"])
def test_a_community_line_is_as_common_whatever_the_gender(
    candidates: list[SyntheticCandidate], gender: str
) -> None:
    group = _of_gender(candidates, gender)
    rate = sum("\nCommunity: " in candidate.resume_text for candidate in group) / len(group)
    assert abs(rate - COMMUNITY_RATE) < 0.05


def test_proxy_share_changes_nothing_but_the_wording_of_womens_community_lines() -> None:
    planted = generate_candidates(500, seed=SEED, proxy_share=1.0)
    unplanted = generate_candidates(500, seed=SEED, proxy_share=0.0)
    for with_phrase, without_phrase in zip(planted, unplanted, strict=True):
        assert dataclasses.replace(with_phrase, resume_text="") == dataclasses.replace(
            without_phrase, resume_text=""
        )
        if with_phrase.gender != "woman":
            assert with_phrase.resume_text == without_phrase.resume_text
    assert not any("women" in candidate.resume_text.casefold() for candidate in unplanted)
    women_with_a_line = [
        c for c in planted if c.gender == "woman" and "\nCommunity: " in c.resume_text
    ]
    assert women_with_a_line
    assert all("women" in candidate.resume_text.casefold() for candidate in women_with_a_line)


@pytest.mark.parametrize("proxy_share", [-0.1, 1.5])
def test_an_out_of_range_proxy_share_is_refused(proxy_share: float) -> None:
    with pytest.raises(ValueError, match="proxy_share"):
        generate_candidates(1, seed=SEED, proxy_share=proxy_share)
