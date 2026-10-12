"""Checks on what a job asks of candidates, shown to the recruiter rather than enforced."""

from __future__ import annotations

from app.models.enums import SeniorityLevel

#: The longest minimum experience a job at each level usually asks for, in years.
#: A longer one is flagged for review: it shuts out candidates who could do the job,
#: and it falls hardest on younger ones, so where the law protects every age (in the
#: UK and the EU, for instance) it has to be justified. These are the top of the
#: ranges the synthetic jobs are drawn from; a deployment should set its own.
USUAL_MAX_MIN_YEARS: dict[SeniorityLevel, int] = {
    SeniorityLevel.INTERN: 0,
    SeniorityLevel.JUNIOR: 1,
    SeniorityLevel.MID: 3,
    SeniorityLevel.SENIOR: 6,
    SeniorityLevel.STAFF: 9,
    SeniorityLevel.PRINCIPAL: 12,
}


def experience_requirement_note(seniority: SeniorityLevel, min_years: int) -> str | None:
    """A note for the recruiter when ``min_years`` is long for ``seniority``, else ``None``."""
    usual = USUAL_MAX_MIN_YEARS[seniority]
    if min_years <= usual:
        return None
    level = seniority.value.replace("_", " ")
    return (
        f"The minimum of {min_years} years' experience is above the {usual} a {level} "
        "role usually asks for. A long minimum rules out candidates who could do the job "
        "and falls hardest on younger ones; check that it is needed."
    )
