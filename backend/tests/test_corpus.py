"""Unit tests for the synthetic corpus generator. No database needed."""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from statistics import mean, median

import pytest

from app.config import REPO_ROOT
from app.ml.corpus import (
    CANDIDATE_YEARS,
    COMMUNITY_RATE,
    FAMILIES,
    JOB_MIN_YEARS,
    PARTIAL_FIT,
    STRONG_FIT,
    Corpus,
    CorpusConfig,
    SyntheticCandidate,
    SyntheticJob,
    corpus_summary,
    generate_candidates,
    generate_corpus,
    generate_jobs,
    ground_truth_fit,
    label_pairs,
    relevance_grade,
)
from app.ml.taxonomy import SKILLS, canonical_skill, normalise_skills
from app.models.enums import EducationLevel, EmploymentType, SeniorityLevel
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
        "from app.ml.corpus import CorpusConfig, generate_corpus\n"
        "config = CorpusConfig(seed=7, n_jobs=20, n_candidates=400, slate_size=20)\n"
        "corpus = generate_corpus(config)\n"
        "rows = corpus.jobs + corpus.candidates + corpus.pairs\n"
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


# --- labelled pairs ---------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return generate_corpus(CorpusConfig(seed=SEED, n_jobs=120, n_candidates=3000))


def _slates(corpus: Corpus) -> dict[str, list[str]]:
    slates: dict[str, list[str]] = defaultdict(list)
    for pair in corpus.pairs:
        slates[pair.job_ref].append(pair.candidate_ref)
    return slates


def _genders(corpus: Corpus) -> dict[str, str]:
    return {candidate.ref: candidate.gender for candidate in corpus.candidates}


def test_the_same_config_gives_the_same_labels(corpus: Corpus) -> None:
    assert generate_corpus(corpus.config) == corpus


def test_every_job_gets_a_full_slate_of_distinct_candidates(corpus: Corpus) -> None:
    slates = _slates(corpus)
    assert set(slates) == {job.ref for job in corpus.jobs}
    for refs in slates.values():
        assert len(refs) == len(set(refs)) == corpus.config.slate_size


def test_a_slate_draws_its_family_share_mostly_from_near_the_jobs_level(corpus: Corpus) -> None:
    jobs = {job.ref: job for job in corpus.jobs}
    people = {candidate.ref: candidate for candidate in corpus.candidates}
    expected = round(corpus.config.slate_size * corpus.config.same_family_share)
    for job_ref, refs in _slates(corpus).items():
        job = jobs[job_ref]
        family = [people[ref] for ref in refs if people[ref].family == job.family]
        assert len(family) == expected, job_ref
        near_pool = [
            c
            for c in corpus.candidates
            if c.family == job.family and abs(c.seniority.rank - job.seniority.rank) <= 1
        ]
        near_picks = sum(abs(c.seniority.rank - job.seniority.rank) <= 1 for c in family)
        assert near_picks >= min(round(expected * 0.6), len(near_pool)), job_ref


def test_relevance_grades_fit_at_the_documented_thresholds(corpus: Corpus) -> None:
    assert all(pair.relevance == relevance_grade(pair.fit) for pair in corpus.pairs)
    assert {pair.relevance for pair in corpus.pairs} == {0, 1, 2}
    assert relevance_grade(STRONG_FIT) == 2
    assert relevance_grade(math.nextafter(STRONG_FIT, 0.0)) == 1
    assert relevance_grade(PARTIAL_FIT) == 1
    assert relevance_grade(math.nextafter(PARTIAL_FIT, 0.0)) == 0


def test_fit_ignores_demographics_and_the_visible_profile(corpus: Corpus) -> None:
    job = corpus.jobs[0]
    twins = (("woman", "40_and_over"), ("man", "under_40"), ("nonbinary", "under_40"))
    for candidate in corpus.candidates[:300]:
        expected = ground_truth_fit(job, candidate)
        for gender, age_band in twins:
            twin = dataclasses.replace(
                candidate, gender=gender, age_band=age_band, resume_text="", skills=()
            )
            assert ground_truth_fit(job, twin) == expected


def test_fit_rises_with_skills_and_falls_with_each_gap(
    jobs: list[SyntheticJob], candidates: list[SyntheticCandidate]
) -> None:
    job = dataclasses.replace(
        jobs[0],
        seniority=SeniorityLevel.MID,
        min_years_experience=3,
        education_required=EducationLevel.BACHELOR,
    )
    qualified = dataclasses.replace(
        candidates[0],
        true_skills=frozenset(job.required_skills + job.preferred_skills),
        years_experience=4,
        seniority=SeniorityLevel.MID,
        education_level=EducationLevel.BACHELOR,
    )
    best = ground_truth_fit(job, qualified)
    assert best >= STRONG_FIT
    for gap in (
        {"true_skills": frozenset(job.preferred_skills)},
        {"years_experience": 0},
        {"seniority": SeniorityLevel.PRINCIPAL},
        {"education_level": EducationLevel.HIGH_SCHOOL},
    ):
        assert ground_truth_fit(job, dataclasses.replace(qualified, **gap)) < best, gap
    assert ground_truth_fit(job, qualified, noise=-1.0) < best
    assert ground_truth_fit(job, qualified, noise=1.0) > best


def test_same_family_pairs_are_far_more_often_relevant(corpus: Corpus) -> None:
    jobs = {job.ref: job for job in corpus.jobs}
    people = {candidate.ref: candidate for candidate in corpus.candidates}
    relevant: dict[bool, list[bool]] = {True: [], False: []}
    for pair in corpus.pairs:
        same_family = people[pair.candidate_ref].family == jobs[pair.job_ref].family
        relevant[same_family].append(pair.relevance >= 1)
    assert mean(relevant[True]) > 5 * mean(relevant[False])


def test_advancement_follows_fit(corpus: Corpus) -> None:
    rates = [mean(p.advanced for p in corpus.pairs if p.relevance == grade) for grade in (0, 1, 2)]
    assert rates[0] < rates[1] < rates[2]


def test_ground_truth_relevance_is_alike_across_genders(corpus: Corpus) -> None:
    genders = _genders(corpus)
    relevance = {
        gender: mean(p.relevance for p in corpus.pairs if genders[p.candidate_ref] == gender)
        for gender in ("woman", "man")
    }
    assert abs(relevance["woman"] - relevance["man"]) < 0.06


def test_with_the_default_bias_women_are_advanced_less_often_than_men(corpus: Corpus) -> None:
    genders = _genders(corpus)
    advanced = {
        gender: mean(p.advanced for p in corpus.pairs if genders[p.candidate_ref] == gender)
        for gender in ("woman", "man")
    }
    assert advanced["woman"] < advanced["man"]


def test_historical_bias_only_ever_withholds_womens_advancement() -> None:
    config = CorpusConfig(seed=SEED, n_jobs=60, n_candidates=1500)
    biased = generate_corpus(config)
    unbiased = generate_corpus(dataclasses.replace(config, historical_bias=0.0))
    assert (biased.jobs, biased.candidates) == (unbiased.jobs, unbiased.candidates)
    genders = _genders(unbiased)
    withheld = 0
    for with_bias, without_bias in zip(biased.pairs, unbiased.pairs, strict=True):
        assert dataclasses.replace(with_bias, advanced=False) == dataclasses.replace(
            without_bias, advanced=False
        )
        if genders[with_bias.candidate_ref] == "woman":
            assert without_bias.advanced or not with_bias.advanced
            withheld += without_bias.advanced and not with_bias.advanced
        else:
            assert with_bias.advanced == without_bias.advanced
    assert withheld > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"n_jobs": 0},
        {"slate_size": 0},
        {"n_candidates": 10, "slate_size": 11},
        {"same_family_share": 1.5},
        {"historical_bias": -0.1},
        {"historical_bias": 1.1},
        {"proxy_share": -0.1},
        {"proxy_share": 1.1},
    ],
)
def test_an_impossible_config_is_refused(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CorpusConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"n_candidates": 50, "slate_size": 50},
        {"n_candidates": 200, "same_family_share": 1.0},
    ],
)
def test_a_slate_the_pool_cannot_fill_is_refused_with_the_reason(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="cannot fill a slate"):
        generate_corpus(CorpusConfig(seed=SEED, n_jobs=2, **changes))


# With these configs a slate of 10 takes 4 candidates from the job's family and 6 from others.
def _pool(
    job: SyntheticJob, candidates: list[SyntheticCandidate], same_family: int, other_families: int
) -> list[SyntheticCandidate]:
    family = [candidate for candidate in candidates if candidate.family == job.family]
    others = [candidate for candidate in candidates if candidate.family != job.family]
    return family[:same_family] + others[:other_families]


def test_a_pool_that_exactly_fills_a_slate_is_used_whole(
    jobs: list[SyntheticJob], candidates: list[SyntheticCandidate]
) -> None:
    config = CorpusConfig(seed=SEED, n_jobs=1, n_candidates=10, slate_size=10)
    pool = _pool(jobs[0], candidates, same_family=4, other_families=6)

    pairs = label_pairs([jobs[0]], pool, config)

    assert {pair.candidate_ref for pair in pairs} == {candidate.ref for candidate in pool}


@pytest.mark.parametrize(("same_family", "other_families"), [(3, 6), (4, 5)])
def test_a_pool_one_candidate_short_is_refused_with_the_reason(
    jobs: list[SyntheticJob],
    candidates: list[SyntheticCandidate],
    same_family: int,
    other_families: int,
) -> None:
    config = CorpusConfig(seed=SEED, n_jobs=1, n_candidates=10, slate_size=10)
    pool = _pool(jobs[0], candidates, same_family, other_families)

    with pytest.raises(ValueError, match="cannot fill a slate"):
        label_pairs([jobs[0]], pool, config)


# --- summary ----------------------------------------------------------------


def test_the_summary_counts_what_the_corpus_holds(corpus: Corpus) -> None:
    summary = corpus_summary(corpus)

    assert (summary.n_jobs, summary.n_candidates, summary.n_pairs) == (
        len(corpus.jobs),
        len(corpus.candidates),
        len(corpus.pairs),
    )
    assert summary.relevance_counts == Counter(pair.relevance for pair in corpus.pairs)
    assert sum(summary.relevance_counts.values()) == len(corpus.pairs)
    assert summary.gender_counts == Counter(c.gender for c in corpus.candidates)
    assert summary.age_band_counts == Counter(c.age_band for c in corpus.candidates)


def test_a_grade_nobody_scored_is_still_reported_as_zero() -> None:
    """All three grades are always keys, so a reader sees an empty class rather than a gap."""
    corpus = generate_corpus(CorpusConfig(seed=SEED, n_jobs=1, n_candidates=60, slate_size=1))

    assert set(corpus_summary(corpus).relevance_counts) == {0, 1, 2}


def test_the_summary_counts_the_jobs_left_without_a_relevant_or_strong_candidate(
    corpus: Corpus,
) -> None:
    graded: dict[str, list[int]] = defaultdict(list)
    for pair in corpus.pairs:
        graded[pair.job_ref].append(pair.relevance)
    summary = corpus_summary(corpus)

    assert summary.jobs_without_relevant == sum(max(g) < 1 for g in graded.values())
    assert summary.jobs_without_strong == sum(max(g) < 2 for g in graded.values())
    assert summary.median_relevant_per_job == median(
        [sum(grade >= 1 for grade in g) for g in graded.values()]
    )


def test_the_summary_rates_match_the_pairs_they_describe(corpus: Corpus) -> None:
    genders = _genders(corpus)
    summary = corpus_summary(corpus)

    assert summary.advance_rate == mean(pair.advanced for pair in corpus.pairs)
    assert summary.advance_rate_strong == mean(
        pair.advanced for pair in corpus.pairs if pair.relevance == 2
    )
    assert summary.advance_rate_other == mean(
        pair.advanced for pair in corpus.pairs if pair.relevance != 2
    )
    assert set(summary.advance_rate_by_gender) == set(genders.values())
    for gender, rate in summary.advance_rate_by_gender.items():
        assert rate == mean(
            pair.advanced for pair in corpus.pairs if genders[pair.candidate_ref] == gender
        )


def test_the_summary_separates_strong_pairs_from_the_rest(corpus: Corpus) -> None:
    summary = corpus_summary(corpus)

    assert summary.advance_rate_other < summary.advance_rate < summary.advance_rate_strong


def test_the_summary_reports_the_planted_gender_bias_and_its_absence() -> None:
    """Compared against the same seed without the bias, not against 1.0.

    The ratio on any one corpus is a noisy estimate -- across twelve seeds it
    runs 0.697 to 0.832 with the bias and 0.918 to 1.078 without it, so the two
    ranges nearly touch and a tolerance tight enough to mean anything would only
    be pinning a lucky draw. The same
    seed without the bias holds the people and the draws fixed, leaving a
    paired comparison that stays within 0.04 of the planted factor.
    """
    config = CorpusConfig(seed=SEED, n_jobs=60, n_candidates=1500)
    kept = 1 - config.historical_bias
    biased = corpus_summary(generate_corpus(config))
    unbiased = corpus_summary(generate_corpus(dataclasses.replace(config, historical_bias=0.0)))

    assert biased.advance_rate_by_gender["man"] == unbiased.advance_rate_by_gender["man"]
    assert biased.advance_rate_by_gender["woman"] == pytest.approx(
        unbiased.advance_rate_by_gender["woman"] * kept, abs=0.05 * kept
    )
    assert biased.women_to_men_advance_ratio == pytest.approx(
        unbiased.women_to_men_advance_ratio * kept, abs=0.05
    )
    assert biased.women_to_men_advance_ratio < unbiased.women_to_men_advance_ratio


def test_a_corpus_nobody_advanced_in_has_no_advance_ratio() -> None:
    """Nothing to divide by, so the ratio is absent rather than a made-up number."""
    corpus = generate_corpus(CorpusConfig(seed=SEED, n_jobs=5, n_candidates=300))
    nobody = dataclasses.replace(
        corpus, pairs=tuple(dataclasses.replace(pair, advanced=False) for pair in corpus.pairs)
    )

    summary = corpus_summary(nobody)

    assert summary.women_to_men_advance_ratio is None
    assert summary.advance_rate == 0.0
    assert set(summary.advance_rate_by_gender.values()) == {0.0}


def test_a_group_with_no_pairs_rates_zero_rather_than_one() -> None:
    """There is nothing to average; 1.0 would read as every strong pair advancing."""
    corpus = generate_corpus(CorpusConfig(seed=SEED, n_jobs=5, n_candidates=300))
    unranked = dataclasses.replace(
        corpus, pairs=tuple(dataclasses.replace(pair, relevance=0) for pair in corpus.pairs)
    )

    summary = corpus_summary(unranked)

    assert summary.relevance_counts[2] == 0
    assert summary.advance_rate_strong == 0.0
    assert summary.median_relevant_per_job == 0
    assert summary.jobs_without_relevant == summary.jobs_without_strong == summary.n_jobs


def test_the_summary_finds_the_planted_proxy_on_womens_resumes_alone() -> None:
    config = CorpusConfig(seed=SEED, n_jobs=5, n_candidates=2000)

    summary = corpus_summary(generate_corpus(config))

    assert summary.proxy_rate_women == pytest.approx(COMMUNITY_RATE * config.proxy_share, abs=0.05)
    assert summary.proxy_rate_others == 0.0


def test_dropping_the_proxy_share_clears_the_proxy_from_every_resume() -> None:
    config = CorpusConfig(seed=SEED, n_jobs=5, n_candidates=2000, proxy_share=0.0)

    summary = corpus_summary(generate_corpus(config))

    assert summary.proxy_rate_women == 0.0
    assert summary.proxy_rate_others == 0.0
