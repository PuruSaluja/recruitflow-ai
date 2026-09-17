"""Unit tests for the engineered matching features. No database needed."""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.ml import features as features_module
from app.ml.corpus import CorpusConfig, SyntheticCandidate, SyntheticJob, generate_corpus
from app.ml.features import (
    FEATURE_NAMES,
    YEARS_SCALE,
    _coverage,
    _jaccard,
    education_fit,
    experience_gap,
    experience_surplus,
    feature_dict,
    feature_vector,
    seniority_direction,
    seniority_distance,
    skill_keys,
)
from app.models.candidate import Candidate
from app.models.enums import EducationLevel, SeniorityLevel
from app.models.job import Job

_S = SeniorityLevel
_E = EducationLevel


def make_job(**changes) -> Job:
    defaults = dict(
        title="Data Engineer",
        description="x",
        seniority=_S.MID,
        education_required=_E.BACHELOR,
        min_years_experience=3,
        required_skills=["Python", "SQL"],
        preferred_skills=["AWS"],
    )
    return Job(**{**defaults, **changes})


def make_candidate(**changes) -> Candidate:
    defaults = dict(
        full_name="A B",
        email="a@b.com",
        resume_text="x",
        seniority=_S.MID,
        education_level=_E.BACHELOR,
        years_experience=3,
        skills=["Python", "SQL"],
    )
    return Candidate(**{**defaults, **changes})


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


# --- the vector contract ----------------------------------------------------


def test_the_names_and_the_dict_describe_the_same_features() -> None:
    assert set(FEATURE_NAMES) == feature_dict(make_job(), make_candidate()).keys()
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_the_vector_follows_the_declared_order() -> None:
    """A model trained on one order and scored on another fails silently."""
    job, candidate = make_job(), make_candidate(years_experience=9, skills=["Python"])
    features = feature_dict(job, candidate)

    vector = feature_vector(job, candidate)

    assert vector == tuple(features[name] for name in FEATURE_NAMES)
    assert len(vector) == len(FEATURE_NAMES)
    # Not all equal, or an order mistake could not show up here.
    assert len(set(vector)) > 1


def test_the_dict_is_built_in_the_declared_order() -> None:
    """Reordering the dict literal should be a test failure, not a silent vector change."""
    assert tuple(feature_dict(make_job(), make_candidate())) == FEATURE_NAMES


def test_the_vector_looks_features_up_by_name_not_by_dict_order(monkeypatch) -> None:
    """The literal happens to match FEATURE_NAMES, so only a reordered dict proves the lookup."""
    job, candidate = make_job(), make_candidate(years_experience=9, skills=["Python"])
    correct = feature_dict(job, candidate)
    monkeypatch.setattr(
        features_module, "feature_dict", lambda j, c: dict(reversed(list(correct.items())))
    )

    assert features_module.feature_vector(job, candidate) == tuple(
        correct[name] for name in FEATURE_NAMES
    )


def test_the_same_pair_always_gives_the_same_vector() -> None:
    job, candidate = make_job(), make_candidate()

    assert feature_vector(job, candidate) == feature_vector(job, candidate)


def _as_orm(job: SyntheticJob, candidate: SyntheticCandidate) -> tuple[Job, Candidate]:
    return (
        make_job(
            seniority=job.seniority,
            education_required=job.education_required,
            min_years_experience=job.min_years_experience,
            required_skills=list(job.required_skills),
            preferred_skills=list(job.preferred_skills),
        ),
        make_candidate(
            seniority=candidate.seniority,
            education_level=candidate.education_level,
            years_experience=candidate.years_experience,
            skills=list(candidate.skills),
        ),
    )


def test_the_orm_models_and_the_synthetic_ones_agree(corpus) -> None:
    """The protocols are structural, so both shapes must really produce the same numbers.

    Run over every pair rather than one: a disagreement could easily depend on
    a skill spelling or a level combination that one sample never reaches.
    """
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}

    for pair in corpus.pairs:
        job, candidate = jobs[pair.job_ref], people[pair.candidate_ref]
        orm_job, orm_candidate = _as_orm(job, candidate)
        assert feature_dict(orm_job, orm_candidate) == feature_dict(job, candidate), pair


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("required_skill_coverage", 0.0, 1.0),
        ("preferred_skill_coverage", 0.0, 1.0),
        ("skill_jaccard", 0.0, 1.0),
        ("experience_gap", 0.0, 1.0),
        ("experience_surplus", 0.0, 1.0),
        ("seniority_distance", 0.0, 1.0),
        ("seniority_direction", -1.0, 1.0),
        ("education_fit", 0.0, 1.0),
    ],
)
def test_every_feature_stays_in_its_range_across_the_corpus(corpus, name, low, high) -> None:
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}

    values = [feature_dict(jobs[p.job_ref], people[p.candidate_ref])[name] for p in corpus.pairs]

    assert all(math.isfinite(v) for v in values)
    assert low <= min(values) and max(values) <= high


# --- skills -----------------------------------------------------------------


def test_an_alias_counts_as_the_skill_it_names() -> None:
    job = make_job(required_skills=["PostgreSQL", "Python"])

    assert _coverage(job.required_skills, skill_keys(["postgres", "python3"])) == 1.0


def test_an_unknown_skill_matches_whatever_its_casing_and_spacing() -> None:
    """The taxonomy leaves unknown skills as written, so the key has to do the folding."""
    job = make_job(required_skills=["Rust  Lang"])

    assert _coverage(job.required_skills, skill_keys(["rust lang"])) == 1.0


def test_repeating_a_skill_does_not_inflate_coverage() -> None:
    job = make_job(required_skills=["Python", "Python", "python3", "SQL"])

    assert _coverage(job.required_skills, skill_keys(["Python"])) == pytest.approx(0.5)


def test_nothing_required_is_full_coverage_not_none() -> None:
    """An empty requirement must not push every candidate to the bottom."""
    assert _coverage([], skill_keys(["Python"])) == 1.0
    assert _coverage([], set()) == 1.0


def test_holding_none_of_what_is_required_is_zero() -> None:
    assert _coverage(["Python", "SQL"], skill_keys(["Cobol"])) == 0.0


def test_coverage_is_an_exact_fraction_not_a_rounded_one() -> None:
    """Halves and whole numbers survive rounding; two thirds does not."""
    assert _coverage(["Python", "SQL", "AWS"], skill_keys(["Python", "SQL"])) == pytest.approx(
        2 / 3
    )
    assert _coverage(["a", "b", "c", "d", "e", "f", "g"], skill_keys(["a"])) == pytest.approx(1 / 7)


def test_coverage_counts_only_what_is_asked_for() -> None:
    """Extra skills neither help nor hurt this feature."""
    held = skill_keys(["Python", "SQL", "Rust", "Go", "Excel"])

    assert _coverage(["Python", "SQL"], held) == 1.0


def test_required_and_preferred_coverage_are_not_interchangeable() -> None:
    """Two features over two different lists; swapping them must not go unnoticed."""
    job = make_job(required_skills=["Python", "SQL"], preferred_skills=["AWS", "Docker"])

    features = feature_dict(job, make_candidate(skills=["Python", "SQL"]))

    assert features["required_skill_coverage"] == 1.0
    assert features["preferred_skill_coverage"] == 0.0


def test_the_overlap_feature_counts_preferred_skills_too() -> None:
    """skill_jaccard is over every skill the job names, not the required list alone."""
    job = make_job(required_skills=["Python", "SQL"], preferred_skills=["AWS"])
    without = feature_dict(job, make_candidate(skills=["Python", "SQL"]))
    with_preferred = feature_dict(job, make_candidate(skills=["Python", "SQL", "AWS"]))

    assert with_preferred["skill_jaccard"] > without["skill_jaccard"]
    assert with_preferred["skill_jaccard"] == 1.0


def test_jaccard_is_overlap_over_union() -> None:
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_two_empty_skill_sets_score_no_overlap() -> None:
    """Conventionally 1.0, but that would hand a skill-less pair a perfect similarity."""
    assert _jaccard(set(), set()) == 0.0


def test_blank_skill_entries_are_ignored() -> None:
    assert skill_keys(["Python", "", "   ", "SQL"]) == skill_keys(["Python", "SQL"])


@pytest.mark.parametrize(
    "spellings",
    [
        ["postgres", "Python", "Rust  Lang", "  SQL ", "python3", "C++", "cpp"],
        ["Straße", "STRASSE", "Ünïcode Skill"],
        [],
    ],
)
def test_keying_an_already_keyed_list_changes_nothing(spellings: list[str]) -> None:
    """_coverage and _jaccard take keys, so keying twice has to be harmless."""
    once = skill_keys(spellings)

    assert skill_keys(once) == once


def test_raw_spellings_are_not_interchangeable_with_keys() -> None:
    """Why _coverage and _jaccard are private: raw input is wrong, not loud."""
    assert _coverage(["Python"], skill_keys(["python3"])) == 1.0
    assert _coverage(["Python"], {"python3"}) == 0.0
    assert _jaccard(skill_keys(["Python"]), skill_keys(["python3"])) == 1.0
    assert _jaccard({"Python"}, {"python3"}) == 0.0


# --- experience -------------------------------------------------------------


def test_meeting_the_minimum_exactly_is_neither_gap_nor_surplus() -> None:
    job, candidate = make_job(min_years_experience=3), make_candidate(years_experience=3)

    assert experience_gap(job, candidate) == 0.0
    assert experience_surplus(job, candidate) == 0.0


@pytest.mark.parametrize("years", [0, 1, 2, 3, 5, 8, 40])
def test_gap_and_surplus_are_never_both_non_zero(years: int) -> None:
    job, candidate = make_job(min_years_experience=3), make_candidate(years_experience=years)

    assert experience_gap(job, candidate) == 0.0 or experience_surplus(job, candidate) == 0.0


def test_the_gap_grows_with_the_shortfall_then_saturates() -> None:
    job = make_job(min_years_experience=10)
    gaps = [experience_gap(job, make_candidate(years_experience=y)) for y in (9, 8, 6, 5, 0)]

    assert gaps == sorted(gaps)
    assert gaps[0] == pytest.approx(1 / YEARS_SCALE)
    assert gaps[-2] == gaps[-1] == 1.0


def test_the_surplus_grows_with_the_years_then_saturates() -> None:
    """Mirrors the gap: the rate matters, not just that it reaches 1.0 eventually."""
    job = make_job(min_years_experience=0)
    surpluses = [
        experience_surplus(job, make_candidate(years_experience=y)) for y in (1, 2, 4, 5, 40)
    ]

    assert surpluses == sorted(surpluses)
    assert surpluses[0] == pytest.approx(1 / YEARS_SCALE)
    assert surpluses[-2] == surpluses[-1] == 1.0


# --- seniority --------------------------------------------------------------


def test_the_same_level_is_no_distance_and_no_direction() -> None:
    job, candidate = make_job(seniority=_S.SENIOR), make_candidate(seniority=_S.SENIOR)

    assert seniority_distance(job, candidate) == 0.0
    assert seniority_direction(job, candidate) == 0.0


def test_distance_ignores_direction_but_direction_does_not() -> None:
    job = make_job(seniority=_S.MID)
    junior, principal = make_candidate(seniority=_S.JUNIOR), make_candidate(seniority=_S.PRINCIPAL)

    assert seniority_distance(job, junior) > 0
    assert seniority_direction(job, junior) == -seniority_distance(job, junior)
    assert seniority_direction(job, principal) == seniority_distance(job, principal)


def test_the_two_ends_of_the_ladder_are_the_extremes() -> None:
    job = make_job(seniority=_S.INTERN)

    assert seniority_distance(job, make_candidate(seniority=_S.PRINCIPAL)) == 1.0
    assert seniority_direction(job, make_candidate(seniority=_S.PRINCIPAL)) == 1.0
    assert (
        seniority_direction(make_job(seniority=_S.PRINCIPAL), make_candidate(seniority=_S.INTERN))
        == -1.0
    )


# --- education --------------------------------------------------------------


def test_meeting_or_exceeding_the_requirement_both_fit_fully() -> None:
    """A PhD does not fit a bachelor's requirement better than a bachelor's does."""
    job = make_job(education_required=_E.BACHELOR)

    assert education_fit(job, make_candidate(education_level=_E.BACHELOR)) == 1.0
    assert education_fit(job, make_candidate(education_level=_E.DOCTORATE)) == 1.0


def test_falling_short_drops_the_fit_step_by_step() -> None:
    job = make_job(education_required=_E.DOCTORATE)
    fits = [
        education_fit(job, make_candidate(education_level=level))
        for level in (_E.HIGH_SCHOOL, _E.ASSOCIATE, _E.BACHELOR, _E.MASTER, _E.DOCTORATE)
    ]

    assert fits == sorted(fits)
    assert fits[0] == 0.0
    assert fits[-1] == 1.0


# --- does it carry the signal it is supposed to -----------------------------


def test_a_better_candidate_scores_better_on_the_features_that_should_move(corpus) -> None:
    job = make_job(
        seniority=_S.SENIOR,
        min_years_experience=6,
        education_required=_E.BACHELOR,
        required_skills=["Python", "SQL", "Apache Spark"],
        preferred_skills=["AWS", "Docker"],
    )
    strong = make_candidate(
        seniority=_S.SENIOR,
        years_experience=7,
        education_level=_E.MASTER,
        skills=["Python", "SQL", "Apache Spark", "AWS", "Docker"],
    )
    weak = make_candidate(
        seniority=_S.INTERN,
        years_experience=0,
        education_level=_E.HIGH_SCHOOL,
        skills=["Excel"],
    )

    good, bad = feature_dict(job, strong), feature_dict(job, weak)

    assert good["required_skill_coverage"] > bad["required_skill_coverage"]
    assert good["preferred_skill_coverage"] > bad["preferred_skill_coverage"]
    assert good["skill_jaccard"] > bad["skill_jaccard"]
    assert good["experience_gap"] < bad["experience_gap"]
    assert good["seniority_distance"] < bad["seniority_distance"]
    assert good["education_fit"] > bad["education_fit"]


def test_skill_coverage_tracks_the_generators_own_notion_of_fit(corpus) -> None:
    """Not a tuned threshold -- just that the feature points the way the labels do."""
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}
    covered = {True: [], False: []}
    for pair in corpus.pairs:
        features = feature_dict(jobs[pair.job_ref], people[pair.candidate_ref])
        covered[pair.relevance == 2].append(features["required_skill_coverage"])

    assert sum(covered[True]) / len(covered[True]) > 2 * sum(covered[False]) / len(covered[False])


def test_the_features_never_read_demographics(corpus) -> None:
    """Gender must not reach the matcher, so changing it may not move a single feature."""
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}

    for pair in corpus.pairs[:200]:
        job, person = jobs[pair.job_ref], people[pair.candidate_ref]
        flipped = dataclasses.replace(
            person,
            gender="nonbinary" if person.gender != "nonbinary" else "woman",
            age_band="40_and_over" if person.age_band == "under_40" else "under_40",
        )
        assert feature_dict(job, flipped) == feature_dict(job, person)


def test_the_features_never_read_the_hidden_true_skills(corpus) -> None:
    """The matcher sees the listed skills only; true_skills is the generator's secret."""
    jobs = {job.ref: job for job in corpus.jobs}
    people = {person.ref: person for person in corpus.candidates}

    for pair in corpus.pairs[:200]:
        job, person = jobs[pair.job_ref], people[pair.candidate_ref]
        blanked = dataclasses.replace(person, true_skills=frozenset())
        assert feature_dict(job, blanked) == feature_dict(job, person)
