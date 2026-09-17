"""Engineered features describing how well one candidate fits one job.

These are the structured half of the matcher's input -- skills, experience,
seniority and education, all read from fields a recruiter can see. The text
half (TF-IDF over the resume and the job description) is separate, so that a
feature that looks wrong can be traced to a field rather than to a vocabulary.

Two rules hold for everything here:

* **Every feature is a finite float, and every one of them is defined for every
  pair.** A job with no required skills or a candidate with no skills still
  produces a full vector; nothing returns None and nothing divides by zero.
* **Order is fixed by** :data:`FEATURE_NAMES`. A model trained on one order and
  scored on another fails silently, so the vector is always built from that
  tuple rather than from a dict's iteration order.

Nothing here reads demographics, and nothing reads a synthetic candidate's
``true_skills``: the matcher only ever sees the skills a profile actually
lists, misspellings, omissions and all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from app.ml.taxonomy import normalise_skills, spelling_key
from app.models.enums import EducationLevel, SeniorityLevel


class JobLike(Protocol):
    """The job fields the matcher reads. Both the ORM model and the synthetic job satisfy it."""

    seniority: SeniorityLevel
    education_required: EducationLevel
    min_years_experience: int
    required_skills: Sequence[str]
    preferred_skills: Sequence[str]


class CandidateLike(Protocol):
    """The candidate fields the matcher reads."""

    seniority: SeniorityLevel
    education_level: EducationLevel
    years_experience: int
    skills: Sequence[str]


#: The feature vector's order. Changing it invalidates any persisted model.
FEATURE_NAMES: tuple[str, ...] = (
    "required_skill_coverage",
    "preferred_skill_coverage",
    "skill_jaccard",
    "experience_gap",
    "experience_surplus",
    "seniority_distance",
    "seniority_direction",
    "education_fit",
)

#: Years of difference at which experience features saturate. Matches the scale
#: the corpus generator uses for its own shortfall term.
YEARS_SCALE = 5.0

_MAX_SENIORITY_STEPS = len(SeniorityLevel) - 1
_MAX_EDUCATION_STEPS = len(EducationLevel) - 1


def skill_keys(spellings: Iterable[str]) -> set[str]:
    """The comparable form of a skill list: canonicalised, then keyed for comparison.

    Canonicalising alone is not enough. A spelling the taxonomy knows becomes
    its canonical name, so ``"postgres"`` and ``"PostgreSQL"`` already agree --
    but an unknown skill is kept as written, so ``"Rust Lang"`` and
    ``"rust lang"`` would not. Keying both collapses that last difference.
    """
    return {spelling_key(name) for name in normalise_skills(spellings)}


def coverage(wanted: Iterable[str], held: set[str]) -> float:
    """Share of ``wanted`` the candidate holds; 1.0 when nothing is wanted.

    Nothing asked for means nothing missing, so an empty requirement is full
    coverage rather than zero -- a job that lists no required skills must not
    push every candidate to the bottom.
    """
    keys = skill_keys(wanted)
    if not keys:
        return 1.0
    return len(keys & held) / len(keys)


def jaccard(left: set[str], right: set[str]) -> float:
    """Overlap of two skill sets over their union; 0.0 when both are empty.

    Two empty sets are conventionally identical, but scoring that 1.0 would
    hand a candidate who lists no skills a perfect similarity against a job
    that lists none either. There is no evidence of overlap, so this reports
    none.
    """
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def experience_gap(job: JobLike, candidate: CandidateLike) -> float:
    """How far short of the job's minimum the candidate falls, 0.0 to 1.0."""
    shortfall = max(job.min_years_experience - candidate.years_experience, 0)
    return min(shortfall / YEARS_SCALE, 1.0)


def experience_surplus(job: JobLike, candidate: CandidateLike) -> float:
    """How far past the job's minimum the candidate goes, 0.0 to 1.0.

    Kept apart from the gap so the model can price being over-experienced
    differently from being under-experienced; at most one of the two is ever
    non-zero.
    """
    surplus = max(candidate.years_experience - job.min_years_experience, 0)
    return min(surplus / YEARS_SCALE, 1.0)


def seniority_distance(job: JobLike, candidate: CandidateLike) -> float:
    """Levels between the two, ignoring direction, 0.0 to 1.0."""
    return abs(candidate.seniority.rank - job.seniority.rank) / _MAX_SENIORITY_STEPS


def seniority_direction(job: JobLike, candidate: CandidateLike) -> float:
    """Which side of the job's level the candidate sits, -1.0 to 1.0.

    Negative is under-levelled, positive over-levelled. Paired with the
    distance, this lets a model treat a senior applying to a junior role
    differently from the reverse.
    """
    return (candidate.seniority.rank - job.seniority.rank) / _MAX_SENIORITY_STEPS


def education_fit(job: JobLike, candidate: CandidateLike) -> float:
    """1.0 when the candidate meets or exceeds the requirement, falling to 0.0.

    Exceeding it is not scored above meeting it -- a PhD does not fit a
    bachelor's requirement better than a bachelor's does.
    """
    shortfall = max(job.education_required.rank - candidate.education_level.rank, 0)
    return 1.0 - shortfall / _MAX_EDUCATION_STEPS


def feature_dict(job: JobLike, candidate: CandidateLike) -> dict[str, float]:
    """Every feature for one pair, keyed by name.

    This is the shape ``match_scores.features`` stores, so a ranking can be
    explained field by field.
    """
    held = skill_keys(candidate.skills)
    job_skills = skill_keys([*job.required_skills, *job.preferred_skills])
    return {
        "required_skill_coverage": coverage(job.required_skills, held),
        "preferred_skill_coverage": coverage(job.preferred_skills, held),
        "skill_jaccard": jaccard(job_skills, held),
        "experience_gap": experience_gap(job, candidate),
        "experience_surplus": experience_surplus(job, candidate),
        "seniority_distance": seniority_distance(job, candidate),
        "seniority_direction": seniority_direction(job, candidate),
        "education_fit": education_fit(job, candidate),
    }


def feature_vector(job: JobLike, candidate: CandidateLike) -> tuple[float, ...]:
    """The same features as a tuple ordered by :data:`FEATURE_NAMES`."""
    features = feature_dict(job, candidate)
    return tuple(features[name] for name in FEATURE_NAMES)
