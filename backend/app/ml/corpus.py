"""Synthetic labelled corpus for training and auditing the matcher.

Everything here is generated; there is no real hiring data behind this project.
What the generator controls, so that nothing measured on it is read as a
finding about real people:

* A candidate's true profile -- role family, skills, experience, seniority,
  education -- is drawn at random. What the profile shows is noisier: the
  skills list leaves some true skills out, and skills are sometimes written
  under an alias.
* ``gender`` comes from its own random stream, independent of every
  qualification. ``age_band`` follows years of experience.
* Some women's resumes name a women-in-tech community group, a phrase a text
  model can learn as a proxy for gender. It is planted so the fairness audit
  has a known proxy to find. Every resume carries a community line at the same
  rate; only the wording differs.
* A labelled pair's ``fit`` is ground truth computed from the true profile
  alone, plus a random term for what no profile shows; it never sees
  demographics. ``relevance`` grades it for ranking metrics.
* ``advanced`` is a simulated past recruiter decision, the kind of label a real
  system learns from. It follows ``fit``, but women were advanced at
  ``1 - historical_bias`` times the rate of equally fit candidates -- a second
  planted bias, which ``historical_bias=0`` removes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from app.ml.taxonomy import SKILLS, get_skill
from app.models.enums import EducationLevel, EmploymentType, SeniorityLevel

_T = TypeVar("_T")
_S = SeniorityLevel
_E = EducationLevel


@dataclass(frozen=True)
class RoleFamily:
    key: str
    title: str
    department: str
    core_skills: tuple[str, ...]
    adjacent_skills: tuple[str, ...]
    #: The education a job in this family usually asks for.
    education: EducationLevel
    degree_fields: tuple[str, ...]
    duties: tuple[str, ...]


FAMILIES: tuple[RoleFamily, ...] = (
    RoleFamily(
        key="data_analysis",
        title="Data Analyst",
        department="Analytics",
        core_skills=("SQL", "Excel", "Tableau", "Power BI", "Statistics", "Python"),
        adjacent_skills=(
            "Looker",
            "A/B Testing",
            "Data Visualization",
            "Pandas",
            "BigQuery",
            "dbt",
        ),
        education=_E.BACHELOR,
        degree_fields=("Economics", "Statistics", "Business Analytics"),
        duties=(
            "build dashboards that track weekly business metrics",
            "answer ad-hoc questions from product and marketing teams",
            "define and document key performance indicators",
            "analyse experiment results and report on their impact",
        ),
    ),
    RoleFamily(
        key="data_engineering",
        title="Data Engineer",
        department="Data Platform",
        core_skills=("Python", "SQL", "Apache Spark", "Apache Airflow", "ETL", "Data Modeling"),
        adjacent_skills=("Kafka", "dbt", "Snowflake", "AWS", "Docker", "Scala"),
        education=_E.BACHELOR,
        degree_fields=("Computer Science", "Information Systems", "Mathematics"),
        duties=(
            "design and maintain batch and streaming data pipelines",
            "model warehouse tables for analytics",
            "monitor pipeline reliability and data quality",
            "move legacy jobs onto an orchestrated platform",
        ),
    ),
    RoleFamily(
        key="data_science",
        title="Data Scientist",
        department="Data Science",
        core_skills=("Python", "Statistics", "Machine Learning", "scikit-learn", "Pandas", "SQL"),
        adjacent_skills=(
            "A/B Testing",
            "R",
            "XGBoost",
            "Feature Engineering",
            "NLP",
            "Deep Learning",
        ),
        education=_E.MASTER,
        degree_fields=("Statistics", "Mathematics", "Computer Science"),
        duties=(
            "build predictive models of customer behaviour",
            "design experiments and analyse their results",
            "turn exploratory analysis into production features",
            "explain model findings to non-technical stakeholders",
        ),
    ),
    RoleFamily(
        key="ml_engineering",
        title="Machine Learning Engineer",
        department="Machine Learning",
        core_skills=(
            "Python",
            "Machine Learning",
            "PyTorch",
            "MLOps",
            "Docker",
            "Feature Engineering",
        ),
        adjacent_skills=(
            "TensorFlow",
            "Kubernetes",
            "AWS",
            "NLP",
            "Computer Vision",
            "Apache Spark",
        ),
        education=_E.BACHELOR,
        degree_fields=("Computer Science", "Electrical Engineering", "Mathematics"),
        duties=(
            "deploy and monitor machine learning models in production",
            "build training pipelines and feature stores",
            "cut model inference latency and cost",
            "help data scientists productionise research code",
        ),
    ),
    RoleFamily(
        key="backend",
        title="Backend Engineer",
        department="Engineering",
        core_skills=("Python", "REST APIs", "PostgreSQL", "Docker", "Git", "System Design"),
        adjacent_skills=("FastAPI", "Django", "Redis", "Kubernetes", "Go", "Microservices"),
        education=_E.BACHELOR,
        degree_fields=("Computer Science", "Software Engineering", "Information Technology"),
        duties=(
            "design and build REST services",
            "own database schemas and migrations",
            "improve service reliability and performance",
            "review code and mentor other engineers",
        ),
    ),
    RoleFamily(
        key="frontend",
        title="Frontend Engineer",
        department="Engineering",
        core_skills=("JavaScript", "TypeScript", "React", "HTML", "CSS", "Git"),
        adjacent_skills=("Redux", "Jest", "Accessibility", "GraphQL", "Vue.js", "Cypress"),
        education=_E.ASSOCIATE,
        degree_fields=("Computer Science", "Interaction Design", "Information Technology"),
        duties=(
            "build accessible, responsive user interfaces",
            "turn designs into reusable components",
            "improve page load performance",
            "write unit and end-to-end tests for the UI",
        ),
    ),
    RoleFamily(
        key="devops",
        title="DevOps Engineer",
        department="Infrastructure",
        core_skills=("Linux", "Docker", "Kubernetes", "Terraform", "CI/CD", "AWS"),
        adjacent_skills=("Prometheus", "Ansible", "Bash", "Go", "GCP", "Azure"),
        education=_E.BACHELOR,
        degree_fields=("Computer Science", "Information Technology", "Network Engineering"),
        duties=(
            "run and automate cloud infrastructure",
            "maintain CI/CD pipelines for dozens of services",
            "improve monitoring, alerting and incident response",
            "manage infrastructure as code",
        ),
    ),
    RoleFamily(
        key="qa",
        title="QA Automation Engineer",
        department="Quality",
        core_skills=("Test Automation", "Selenium", "Python", "CI/CD", "Git", "pytest"),
        adjacent_skills=("Cypress", "Jest", "JavaScript", "REST APIs", "Docker", "Agile"),
        education=_E.ASSOCIATE,
        degree_fields=("Computer Science", "Information Technology", "Software Engineering"),
        duties=(
            "build and maintain automated regression suites",
            "run tests as part of the CI pipeline",
            "reproduce and triage defects with developers",
            "write test plans for new features",
        ),
    ),
)

LOCATIONS = (
    "Atlanta, GA",
    "Austin, TX",
    "Boston, MA",
    "Chicago, IL",
    "Denver, CO",
    "New York, NY",
    "Raleigh, NC",
    "San Francisco, CA",
    "Seattle, WA",
    "Remote",
)
FIRST_NAMES = (
    "Alex", "Avery", "Casey", "Dakota", "Drew", "Emerson", "Hayden", "Jamie", "Jordan", "Kai",
    "Morgan", "Parker", "Quinn", "Reese", "Riley", "Robin", "Rowan", "Sam", "Skyler", "Taylor",
)  # fmt: skip
LAST_NAMES = (
    "Ali", "Brown", "Chen", "Fischer", "Garcia", "Haddad", "Johnson", "Kim", "Kowalski", "Lopez",
    "Mensah", "Murphy", "Nguyen", "Novak", "Okafor", "Patel", "Rossi", "Silva", "Smith", "Tanaka",
)  # fmt: skip

#: Relative share of each gender in the candidate pool.
GENDER_WEIGHTS = {"woman": 40, "man": 55, "nonbinary": 5}
#: Share of resumes, in every group alike, that end with a community line.
COMMUNITY_RATE = 0.4

#: Years of experience a candidate at each level has, inclusive.
CANDIDATE_YEARS: dict[SeniorityLevel, tuple[int, int]] = {
    _S.INTERN: (0, 1),
    _S.JUNIOR: (0, 3),
    _S.MID: (2, 6),
    _S.SENIOR: (5, 11),
    _S.STAFF: (8, 16),
    _S.PRINCIPAL: (12, 25),
}
#: Minimum years of experience a job at each level asks for, inclusive.
JOB_MIN_YEARS: dict[SeniorityLevel, tuple[int, int]] = {
    _S.INTERN: (0, 0),
    _S.JUNIOR: (0, 1),
    _S.MID: (2, 3),
    _S.SENIOR: (4, 6),
    _S.STAFF: (7, 9),
    _S.PRINCIPAL: (10, 12),
}

_JOB_SENIORITY = {
    _S.INTERN: 5,
    _S.JUNIOR: 20,
    _S.MID: 30,
    _S.SENIOR: 30,
    _S.STAFF: 10,
    _S.PRINCIPAL: 5,
}
_CANDIDATE_SENIORITY = {
    _S.INTERN: 5,
    _S.JUNIOR: 22,
    _S.MID: 33,
    _S.SENIOR: 27,
    _S.STAFF: 9,
    _S.PRINCIPAL: 4,
}
_EMPLOYMENT = {
    EmploymentType.FULL_TIME: 85,
    EmploymentType.CONTRACT: 10,
    EmploymentType.PART_TIME: 5,
}
_EDUCATION = {_E.HIGH_SCHOOL: 6, _E.ASSOCIATE: 12, _E.BACHELOR: 55, _E.MASTER: 22, _E.DOCTORATE: 5}
#: Data science and ML engineering see more advanced degrees than other families.
_RESEARCH_EDUCATION = {
    _E.HIGH_SCHOOL: 2,
    _E.ASSOCIATE: 3,
    _E.BACHELOR: 45,
    _E.MASTER: 38,
    _E.DOCTORATE: 12,
}
_RESEARCH_FAMILIES = frozenset({"data_science", "ml_engineering"})
_EDUCATION_BY_RANK = sorted(EducationLevel, key=lambda level: level.rank)
_START_AGE = {
    _E.HIGH_SCHOOL: 19,
    _E.ASSOCIATE: 20,
    _E.BACHELOR: 22,
    _E.MASTER: 24,
    _E.DOCTORATE: 27,
}
_BASE_SALARY = {
    _S.INTERN: 45_000,
    _S.JUNIOR: 70_000,
    _S.MID: 95_000,
    _S.SENIOR: 130_000,
    _S.STAFF: 165_000,
    _S.PRINCIPAL: 195_000,
}

_TITLE_PATTERNS = {
    _S.INTERN: "{title} Intern",
    _S.JUNIOR: "Junior {title}",
    _S.MID: "{title}",
    _S.SENIOR: "Senior {title}",
    _S.STAFF: "Staff {title}",
    _S.PRINCIPAL: "Principal {title}",
}
_DEGREES = {
    _E.ASSOCIATE: "Associate degree",
    _E.BACHELOR: "Bachelor's degree",
    _E.MASTER: "Master's degree",
    _E.DOCTORATE: "PhD",
}
_DEGREE_REQUIREMENTS = {
    _E.HIGH_SCHOOL: "a high school diploma or higher",
    _E.ASSOCIATE: "an associate degree or higher",
    _E.BACHELOR: "a bachelor's degree or higher",
    _E.MASTER: "a master's degree or higher",
    _E.DOCTORATE: "a PhD",
}

#: Chance of holding each core skill of one's family at intern level; each level adds to it.
_CORE_RATE = 0.5
_CORE_RATE_PER_LEVEL = 0.08
_ADJACENT_RATE = 0.3
#: Share of true skills a profile's skills list includes.
_LISTED_RATE = 0.8
#: Share of true skills the resume text mentions.
_MENTION_RATE = 0.7
#: Chance a written skill uses an alias rather than its canonical name.
_ALIAS_RATE = 0.2

_NEUTRAL_COMMUNITY = (
    "Organiser of a local open-source meetup",
    "Volunteer tutor at a weekend coding club",
    "Captain of a chess club",
    "Mentor at student hackathons",
)
_WOMEN_COMMUNITY = (
    "Organiser of a women in data meetup",
    "Mentor in a women's coding bootcamp",
    "Member of a women in engineering network",
)


@dataclass(frozen=True)
class SyntheticJob:
    ref: str
    family: str
    title: str
    department: str
    location: str
    employment_type: EmploymentType
    seniority: SeniorityLevel
    education_required: EducationLevel
    min_years_experience: int
    description: str
    required_skills: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    salary_min: int
    salary_max: int

    def api_payload(self) -> dict[str, Any]:
        """The body ``POST /jobs`` accepts for this requisition."""
        return {
            "title": self.title,
            "department": self.department,
            "location": self.location,
            "employment_type": self.employment_type.value,
            "seniority": self.seniority.value,
            "education_required": self.education_required.value,
            "min_years_experience": self.min_years_experience,
            "description": self.description,
            "required_skills": list(self.required_skills),
            "preferred_skills": list(self.preferred_skills),
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
        }


@dataclass(frozen=True)
class SyntheticCandidate:
    ref: str
    family: str
    full_name: str
    email: str
    location: str
    current_title: str
    years_experience: int
    seniority: SeniorityLevel
    education_level: EducationLevel
    resume_text: str
    #: The skills list as the profile shows it: some true skills left out, some under aliases.
    skills: tuple[str, ...]
    #: Canonical names of the skills the candidate really has.
    true_skills: frozenset[str]
    gender: str
    age_band: str

    def api_payload(self) -> dict[str, Any]:
        """The body ``POST /candidates`` accepts. Demographics and true skills stay out of it."""
        return {
            "full_name": self.full_name,
            "email": self.email,
            "location": self.location,
            "current_title": self.current_title,
            "years_experience": self.years_experience,
            "seniority": self.seniority.value,
            "education_level": self.education_level.value,
            "resume_text": self.resume_text,
            "skills": list(self.skills),
        }


def title_for(family: RoleFamily, seniority: SeniorityLevel) -> str:
    return _TITLE_PATTERNS[seniority].format(title=family.title)


def generate_jobs(n: int, *, seed: int) -> list[SyntheticJob]:
    """Generate ``n`` requisitions; the same arguments always give the same jobs."""
    rng = random.Random(f"{seed}:jobs")
    jobs: list[SyntheticJob] = []
    for index in range(n):
        family = rng.choice(FAMILIES)
        seniority = _weighted(rng, _JOB_SENIORITY)
        title = title_for(family, seniority)
        required = tuple(rng.sample(family.core_skills, rng.randint(3, 5)))
        preferred = tuple(rng.sample(family.adjacent_skills, rng.randint(2, 3)))
        min_years = rng.randint(*JOB_MIN_YEARS[seniority])
        education = _job_education(rng, family)
        employment = (
            EmploymentType.INTERNSHIP if seniority is _S.INTERN else _weighted(rng, _EMPLOYMENT)
        )
        salary_min, salary_max = _salary_band(rng, seniority)
        location = rng.choice(LOCATIONS)
        description = _job_description(
            rng, family, title, required, preferred, min_years, education
        )
        jobs.append(
            SyntheticJob(
                ref=f"J{index + 1:04d}",
                family=family.key,
                title=title,
                department=family.department,
                location=location,
                employment_type=employment,
                seniority=seniority,
                education_required=education,
                min_years_experience=min_years,
                description=description,
                required_skills=required,
                preferred_skills=preferred,
                salary_min=salary_min,
                salary_max=salary_max,
            )
        )
    return jobs


def generate_candidates(n: int, *, seed: int, proxy_share: float = 0.6) -> list[SyntheticCandidate]:
    """Generate ``n`` candidates; the same arguments always give the same people.

    ``proxy_share`` is the share of women's community lines that name a
    women-in-tech group. Profiles and demographics come from separate random
    streams, so neither it nor a candidate's gender can change a qualification.
    """
    if not 0.0 <= proxy_share <= 1.0:
        raise ValueError(f"proxy_share must be between 0 and 1, got {proxy_share}")

    profiles = random.Random(f"{seed}:candidates")
    people = random.Random(f"{seed}:demographics")
    candidates: list[SyntheticCandidate] = []
    for index in range(n):
        family = profiles.choice(FAMILIES)
        seniority = _weighted(profiles, _CANDIDATE_SENIORITY)
        years = profiles.randint(*CANDIDATE_YEARS[seniority])
        education = _weighted(
            profiles, _RESEARCH_EDUCATION if family.key in _RESEARCH_FAMILIES else _EDUCATION
        )
        true_skills = _true_skills(profiles, family, seniority)
        skills = _listed_skills(profiles, true_skills)
        title = title_for(family, seniority)
        location = profiles.choice(LOCATIONS)
        resume = _resume_text(profiles, family, title, years, education, true_skills)

        gender = _weighted(people, GENDER_WEIGHTS)
        age = _START_AGE[education] + years + people.randint(0, 8)
        first, last = people.choice(FIRST_NAMES), people.choice(LAST_NAMES)
        community = _community_line(people, gender, proxy_share)
        if community:
            resume = f"{resume}\n{community}"

        candidates.append(
            SyntheticCandidate(
                ref=f"C{index + 1:05d}",
                family=family.key,
                full_name=f"{first} {last}",
                email=f"{first}.{last}.{index + 1}@example.com".lower(),
                location=location,
                current_title=title,
                years_experience=years,
                seniority=seniority,
                education_level=education,
                resume_text=resume,
                skills=skills,
                true_skills=true_skills,
                gender=gender,
                age_band="40_and_over" if age >= 40 else "under_40",
            )
        )
    return candidates


def _weighted(rng: random.Random, weights: Mapping[_T, int]) -> _T:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _spell(rng: random.Random, name: str) -> str:
    aliases = get_skill(name).aliases
    if aliases and rng.random() < _ALIAS_RATE:
        return rng.choice(aliases)
    return name


def _join(items: Sequence[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _job_education(rng: random.Random, family: RoleFamily) -> EducationLevel:
    shift = _weighted(rng, {0: 70, -1: 20, 1: 10})
    rank = min(max(family.education.rank + shift, 0), len(_EDUCATION_BY_RANK) - 1)
    return _EDUCATION_BY_RANK[rank]


def _salary_band(rng: random.Random, seniority: SeniorityLevel) -> tuple[int, int]:
    low = round(_BASE_SALARY[seniority] * rng.uniform(0.9, 1.1), -3)
    high = round(low * rng.uniform(1.15, 1.35), -3)
    return int(low), int(high)


def _job_description(
    rng: random.Random,
    family: RoleFamily,
    title: str,
    required: Sequence[str],
    preferred: Sequence[str],
    min_years: int,
    education: EducationLevel,
) -> str:
    duties = rng.sample(family.duties, 3)
    if min_years == 0:
        experience = "none required"
    else:
        experience = f"at least {min_years} year{'' if min_years == 1 else 's'}"
    return "\n".join(
        [
            f"We are hiring a {title} to join the {family.department} team.",
            f"You will {duties[0]}, {duties[1]} and {duties[2]}.",
            f"Required: {_join([_spell(rng, name) for name in required])}.",
            f"Nice to have: {_join([_spell(rng, name) for name in preferred])}.",
            f"Experience: {experience}.",
            f"Education: {_DEGREE_REQUIREMENTS[education]}.",
        ]
    )


def _true_skills(
    rng: random.Random, family: RoleFamily, seniority: SeniorityLevel
) -> frozenset[str]:
    core_rate = _CORE_RATE + _CORE_RATE_PER_LEVEL * seniority.rank
    skills = {name for name in family.core_skills if rng.random() < core_rate}
    if len(skills) < 2:
        skills.update(rng.sample(family.core_skills, 2))
    skills.update(name for name in family.adjacent_skills if rng.random() < _ADJACENT_RATE)
    family_pool = set(family.core_skills) | set(family.adjacent_skills)
    outside = [skill.name for skill in SKILLS if skill.name not in family_pool]
    skills.update(rng.sample(outside, rng.randint(0, 3)))
    return frozenset(skills)


def _listed_skills(rng: random.Random, true_skills: frozenset[str]) -> tuple[str, ...]:
    # Sorted before any draw: set order varies between processes, the corpus must not.
    listed = [_spell(rng, name) for name in sorted(true_skills) if rng.random() < _LISTED_RATE]
    rng.shuffle(listed)
    return tuple(listed)


def _resume_text(
    rng: random.Random,
    family: RoleFamily,
    title: str,
    years: int,
    education: EducationLevel,
    true_skills: frozenset[str],
) -> str:
    mentioned = [name for name in sorted(true_skills) if rng.random() < _MENTION_RATE]
    rng.shuffle(mentioned)
    duties = rng.sample(family.duties, 3)
    lines = [f"{title} with {_years_phrase(years)} of experience."]
    for position, duty in enumerate(duties):
        bullet = f"- {duty[0].upper()}{duty[1:]}"
        skills = mentioned[position :: len(duties)]
        if skills:
            bullet += " using " + _join([_spell(rng, name) for name in skills])
        lines.append(f"{bullet}.")
    if education is _E.HIGH_SCHOOL:
        lines.append("Education: high school diploma.")
    else:
        lines.append(f"Education: {_DEGREES[education]} in {rng.choice(family.degree_fields)}.")
    return "\n".join(lines)


def _years_phrase(years: int) -> str:
    if years == 0:
        return "less than a year"
    return f"{years} year{'' if years == 1 else 's'}"


def _community_line(rng: random.Random, gender: str, proxy_share: float) -> str | None:
    # Every draw happens whatever the outcome, so proxy_share and gender can
    # change this line's wording but never what the stream produces next.
    has_line = rng.random() < COMMUNITY_RATE
    names_group = rng.random() < proxy_share
    neutral = rng.choice(_NEUTRAL_COMMUNITY)
    womens = rng.choice(_WOMEN_COMMUNITY)
    if not has_line:
        return None
    return f"Community: {womens if gender == 'woman' and names_group else neutral}."


#: ``fit`` at or above these grades a pair 2 (strong) or 1 (partial); below both it is 0.
STRONG_FIT = 0.6
PARTIAL_FIT = 0.35
#: Spread of the random term in ``fit`` that stands in for what no profile shows.
_FIT_NOISE = 0.5
#: Share of a slate's same-family picks drawn from within one level of the job.
_NEAR_LEVEL_SHARE = 0.6


@dataclass(frozen=True)
class LabelledPair:
    job_ref: str
    candidate_ref: str
    #: Ground truth in (0, 1), from the true profile and never from demographics.
    fit: float
    #: ``fit`` graded for ranking metrics: 2 strong, 1 partial, 0 not a fit.
    relevance: int
    #: Whether a past recruiter moved the candidate forward; carries the planted bias.
    advanced: bool


@dataclass(frozen=True)
class CorpusConfig:
    seed: int = 42
    n_jobs: int = 200
    n_candidates: int = 4000
    slate_size: int = 50
    #: Share of each slate drawn from the job's own role family.
    same_family_share: float = 0.4
    #: How much less often women were advanced than equally fit candidates.
    historical_bias: float = 0.25
    #: Share of women's community lines that name a women-in-tech group.
    proxy_share: float = 0.6

    def __post_init__(self) -> None:
        if min(self.n_jobs, self.n_candidates, self.slate_size) < 1:
            raise ValueError("n_jobs, n_candidates and slate_size must all be at least 1")
        if self.slate_size > self.n_candidates:
            raise ValueError("slate_size cannot be larger than n_candidates")
        if not 0.0 <= self.same_family_share <= 1.0:
            raise ValueError("same_family_share must be between 0 and 1")
        if not 0.0 <= self.historical_bias <= 1.0:
            raise ValueError("historical_bias must be between 0 and 1")


@dataclass(frozen=True)
class Corpus:
    config: CorpusConfig
    jobs: tuple[SyntheticJob, ...]
    candidates: tuple[SyntheticCandidate, ...]
    pairs: tuple[LabelledPair, ...]


def generate_corpus(config: CorpusConfig | None = None) -> Corpus:
    """Jobs, candidates and labelled pairs; the same config always gives the same corpus."""
    config = config or CorpusConfig()
    jobs = generate_jobs(config.n_jobs, seed=config.seed)
    candidates = generate_candidates(
        config.n_candidates, seed=config.seed, proxy_share=config.proxy_share
    )
    pairs = label_pairs(jobs, candidates, config)
    return Corpus(config, tuple(jobs), tuple(candidates), tuple(pairs))


def ground_truth_fit(
    job: SyntheticJob, candidate: SyntheticCandidate, *, noise: float = 0.0
) -> float:
    """How well ``candidate`` truly meets ``job``, in (0, 1).

    Reads only the true profile -- skill coverage, a shortfall in years, distance
    in seniority and missing education -- while ``noise`` stands in for what no
    profile shows.
    """
    skills = candidate.true_skills
    required = sum(name in skills for name in job.required_skills) / len(job.required_skills)
    preferred = sum(name in skills for name in job.preferred_skills) / len(job.preferred_skills)
    shortfall = min(max(job.min_years_experience - candidate.years_experience, 0) / 5, 1.0)
    seniority_gap = abs(job.seniority.rank - candidate.seniority.rank)
    education_gap = max(job.education_required.rank - candidate.education_level.rank, 0)
    score = (
        4.0 * required
        + 1.0 * preferred
        - 2.0 * shortfall
        - 0.8 * seniority_gap
        - 0.7 * education_gap
        + noise
        - 3.0
    )
    return 1.0 / (1.0 + math.exp(-score))


def relevance_grade(fit: float) -> int:
    """Grade ``fit`` for ranking metrics: 2 strong, 1 partial, 0 not a fit."""
    if fit >= STRONG_FIT:
        return 2
    return 1 if fit >= PARTIAL_FIT else 0


def label_pairs(
    jobs: Sequence[SyntheticJob], candidates: Sequence[SyntheticCandidate], config: CorpusConfig
) -> list[LabelledPair]:
    """Label every job against a slate of candidates, sourced the way a recruiter would.

    A slate takes ``same_family_share`` of its candidates from the job's role
    family, most of those from within one level of the job, and the rest from
    other families.
    """
    slates = random.Random(f"{config.seed}:slates")
    outcomes = random.Random(f"{config.seed}:outcomes")
    by_family: dict[str, list[SyntheticCandidate]] = {family.key: [] for family in FAMILIES}
    for candidate in candidates:
        by_family[candidate.family].append(candidate)
    outsiders = {key: [c for c in candidates if c.family != key] for key in by_family}
    n_same = round(config.slate_size * config.same_family_share)
    n_other = config.slate_size - n_same

    pairs: list[LabelledPair] = []
    for job in jobs:
        family = by_family[job.family]
        if len(family) < n_same or len(outsiders[job.family]) < n_other:
            raise ValueError(
                f"The pool cannot fill a slate of {config.slate_size} for {job.ref}: it needs "
                f"{n_same} {job.family} candidates and {n_other} from other families but has "
                f"{len(family)} and {len(outsiders[job.family])}. Generate more candidates or "
                "use a smaller slate."
            )
        near = [c for c in family if abs(c.seniority.rank - job.seniority.rank) <= 1]
        slate = slates.sample(near, min(round(n_same * _NEAR_LEVEL_SHARE), len(near)))
        picked = {c.ref for c in slate}
        slate += slates.sample([c for c in family if c.ref not in picked], n_same - len(slate))
        slate += slates.sample(outsiders[job.family], n_other)
        for candidate in slate:
            # Both draws happen for every pair, so historical_bias changes no other label.
            fit = ground_truth_fit(job, candidate, noise=outcomes.gauss(0.0, _FIT_NOISE))
            draw = outcomes.random()
            chance = _advance_chance(fit)
            if candidate.gender == "woman":
                chance *= 1.0 - config.historical_bias
            pairs.append(
                LabelledPair(job.ref, candidate.ref, fit, relevance_grade(fit), draw < chance)
            )
    return pairs


def _advance_chance(fit: float) -> float:
    return 1.0 / (1.0 + math.exp(-6.0 * (fit - 0.5)))
