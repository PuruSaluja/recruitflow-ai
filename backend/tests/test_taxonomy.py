"""Unit tests for the skills taxonomy. No database needed."""

from __future__ import annotations

import time

import pytest

from app.ml.taxonomy import (
    CATEGORIES,
    SKILLS,
    WRITTEN_AS,
    Skill,
    build_lookup,
    canonical_skill,
    get_skill,
    normalise_skills,
    skills_in_text,
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


# --- skills named in free text -------------------------------------------------


def test_every_case_bound_spelling_is_one_the_taxonomy_has() -> None:
    spellings = {
        spelling.casefold() for skill in SKILLS for spelling in (skill.name, *skill.aliases)
    }
    assert set(WRITTEN_AS) <= spellings
    for key, forms in WRITTEN_AS.items():
        assert all(form.casefold() == key for form in forms), key


def test_every_spelling_is_found_in_a_sentence_as_written() -> None:
    for skill in SKILLS:
        for spelling in (skill.name, *skill.aliases):
            for form in WRITTEN_AS.get(spelling.casefold(), (spelling,)):
                assert skills_in_text(f"Five years of {form}, daily.") == [skill.name], form


def test_case_does_not_matter_for_spellings_that_are_not_ordinary_words() -> None:
    assert skills_in_text("POSTGRESQL, kubernetes and Scikit-Learn") == [
        "PostgreSQL",
        "Kubernetes",
        "scikit-learn",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Quick to react to incidents",
        "We excel at keeping things agile",
        "Let the rest of the team go",
        "Led R&D at a startup",
        "Edited index.ts and app.js",
        "Worked at Google on Gopher tooling",
        "A spark of an idea; a flask of tea; 5 ml of torch oil",
        "In spring every node was busy",
    ],
)
def test_ordinary_words_are_not_read_as_skills(text: str) -> None:
    assert skills_in_text(text) == []


def test_a_capitalised_word_starting_a_sentence_is_read_as_the_skill() -> None:
    """The limit of going by case: "Go" opening a sentence reads as the language,
    as "React" opening a line of skills has to."""
    assert skills_in_text("Go to the rest of the team.") == ["Go"]
    assert skills_in_text("React, Redux and Jest") == ["React", "Redux", "Jest"]


def test_names_inside_longer_names_are_not_found_twice() -> None:
    # Java inside JavaScript, SQL inside PostgreSQL and NoSQL, Go inside Google.
    assert skills_in_text("JavaScript, PostgreSQL, NoSQL and Google") == [
        "JavaScript",
        "PostgreSQL",
    ]


def test_the_longest_spelling_wins_where_spellings_overlap() -> None:
    assert skills_in_text("ML Ops on Spring Boot with Apache Spark") == [
        "MLOps",
        "Spring Boot",
        "Apache Spark",
    ]


def test_symbols_in_names_are_matched_whole() -> None:
    assert skills_in_text("C++, C#, CI/CD, A/B Testing and Node.js") == [
        "C++",
        "C#",
        "CI/CD",
        "A/B Testing",
        "Node.js",
    ]


def test_skills_come_once_each_in_order_of_first_mention() -> None:
    text = "Docker and Python.\nMore Python, postgres, then Docker again; Postgres."
    assert skills_in_text(text) == ["Docker", "Python", "PostgreSQL"]


def test_a_spelling_split_across_lines_is_found() -> None:
    assert skills_in_text("Built with Machine\nLearning and Power  BI") == [
        "Machine Learning",
        "Power BI",
    ]


def test_text_with_no_skills_gives_none() -> None:
    assert skills_in_text("") == []
    assert skills_in_text("Managed a team of twelve.") == []


def test_a_resume_at_the_length_cap_is_read_quickly() -> None:
    """100,000 characters naming two skills thousands of times. Resolving overlaps
    by comparing each match with every kept one took 5.6 s here; the time is
    now linear in the matches."""
    text = ("Python developer with SQL. " * 4000)[:100_000]

    started = time.perf_counter()
    found = skills_in_text(text)
    elapsed = time.perf_counter() - started

    assert found == ["Python", "SQL"]
    assert elapsed < 2.0
