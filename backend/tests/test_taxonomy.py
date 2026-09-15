"""Unit tests for the skills taxonomy. No database needed."""

from __future__ import annotations

import pytest

from app.ml.taxonomy import (
    CATEGORIES,
    SKILLS,
    Skill,
    build_lookup,
    canonical_skill,
    get_skill,
    normalise_skills,
)


def test_canonical_names_are_unique() -> None:
    names = [skill.name for skill in SKILLS]
    assert len(names) == len(set(names))


def test_every_skill_has_a_known_category() -> None:
    assert {skill.category for skill in SKILLS} <= CATEGORIES


def test_every_spelling_resolves_to_its_own_skill_whatever_its_case_or_spacing() -> None:
    for skill in SKILLS:
        for spelling in (skill.name, *skill.aliases):
            variants = (
                spelling,
                spelling.upper(),
                f"  {spelling.lower()}\t",
                spelling.replace(" ", "  "),
            )
            for variant in variants:
                assert canonical_skill(variant) == skill.name, variant


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("k8s", "Kubernetes"),
        ("postgres", "PostgreSQL"),
        ("sklearn", "scikit-learn"),
        ("golang", "Go"),
        ("  POWER   bi ", "Power BI"),
        ("Data\tModelling", "Data Modeling"),
    ],
)
def test_spellings_match_whatever_their_case_and_spacing(spelling: str, expected: str) -> None:
    assert canonical_skill(spelling) == expected


@pytest.mark.parametrize("spelling", ["COBOL", "", "   "])
def test_an_unknown_or_blank_spelling_has_no_canonical_name(spelling: str) -> None:
    assert canonical_skill(spelling) is None


def test_a_spelling_claimed_by_two_skills_is_refused() -> None:
    skills = [Skill("Go", "language", ("golang",)), Skill("Go Tooling", "practice", ("GoLang",))]
    with pytest.raises(ValueError, match="claimed by both"):
        build_lookup(skills)


def test_get_skill_takes_canonical_names_only() -> None:
    assert get_skill("Kubernetes").aliases == ("k8s",)
    with pytest.raises(KeyError):
        get_skill("k8s")


def test_normalising_canonicalises_and_drops_repeats_in_first_seen_order() -> None:
    assert normalise_skills(["Postgres", "SQL", "postgresql", " sql "]) == ["PostgreSQL", "SQL"]


def test_normalising_keeps_an_unknown_skill_as_first_written() -> None:
    assert normalise_skills(["  COBOL   batch ", "cobol batch"]) == ["COBOL batch"]


def test_normalising_drops_blank_entries() -> None:
    assert normalise_skills(["", "   ", "k8s"]) == ["Kubernetes"]
