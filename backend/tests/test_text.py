"""Unit tests for the text documents and the vectoriser settings. No database needed."""

from __future__ import annotations

import dataclasses

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.text import candidate_text, job_text, make_vectoriser
from app.models.candidate import Candidate
from app.models.enums import EducationLevel, SeniorityLevel
from app.models.job import Job


def tokens(text: str) -> list[str]:
    return make_vectoriser().build_analyzer()(text)


def make_job(**changes) -> Job:
    defaults = dict(
        title="Data Engineer",
        description="Build and run batch pipelines.",
        seniority=SeniorityLevel.MID,
        education_required=EducationLevel.BACHELOR,
        min_years_experience=3,
        required_skills=["Python", "Apache Spark"],
        preferred_skills=["dbt"],
    )
    return Job(**{**defaults, **changes})


def make_candidate(**changes) -> Candidate:
    defaults = dict(
        full_name="A B",
        email="a@b.com",
        current_title="Analyst",
        resume_text="Built dashboards in Tableau.",
        seniority=SeniorityLevel.MID,
        education_level=EducationLevel.BACHELOR,
        years_experience=3,
        skills=["SQL", "Tableau"],
    )
    return Candidate(**{**defaults, **changes})


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


# --- tokens -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C++", ["c++"]),
        ("C#", ["c#"]),
        ("F#", ["f#"]),
        ("R", ["r"]),
        ("Node.js", ["node.js"]),
        ("Vue.js", ["vue.js"]),
    ],
)
def test_skill_names_come_through_whole(text: str, expected: list[str]) -> None:
    assert tokens(text) == expected


def test_the_default_pattern_would_have_lost_them() -> None:
    """The reason the module sets its own pattern, as its docstring claims."""
    default = TfidfVectorizer().build_analyzer()

    assert default("C++ C# R") == []
    assert default("Node.js") == ["node", "js"]


def test_punctuation_around_a_word_is_dropped() -> None:
    words = tokens("Python, SQL; (Airflow) experience.")

    assert words == ["python", "sql", "airflow", "experience"]


def test_slashes_and_hyphens_still_split() -> None:
    """Python/SQL is two skills, not one token that no job will ever contain."""
    assert tokens("Python/SQL") == ["python", "sql"]
    assert tokens("CI/CD") == ["ci", "cd"]
    assert tokens("scikit-learn") == ["scikit", "learn"]


def test_a_dot_joins_only_between_word_characters() -> None:
    assert tokens("version 3.14.6 of it.") == ["version", "3.14.6", "of", "it"]


def test_tokens_are_lower_cased() -> None:
    assert tokens("PYTHON Python python") == ["python", "python", "python"]


def test_words_outside_ascii_are_kept() -> None:
    assert tokens("Ünïcode Café") == ["ünïcode", "café"]


@pytest.mark.parametrize(
    ("variant", "plain"),
    [
        ("café", "café"),
        ("certiﬁed", "certified"),
        ("Ｐｙｔｈｏｎ", "Python"),
        ("Py​thon", "Python"),
        ("Py­thon", "Python"),
        ("Py﻿thon", "Python"),
        ("STRASSE", "Straße"),
    ],
    ids=["decomposed", "ligature", "full-width", "zero-width", "soft-hyphen", "bom", "casefold"],
)
def test_one_word_is_one_term_however_it_is_encoded(variant: str, plain: str) -> None:
    """Only uploaded resumes are normalised on the way in; JSON bodies and job text are not."""
    assert tokens(variant) == tokens(plain)
    assert len(tokens(plain)) == 1


def test_a_possessive_keeps_its_word() -> None:
    """The planted proxy is written both "women's" and "women"; both must give one term."""
    assert "women" in tokens("Mentor in a women's coding bootcamp")
    assert "women" in tokens("Organiser of a women in data meetup")


def test_skills_that_are_also_english_stop_words_survive() -> None:
    """scikit-learn's English list would remove every one of these."""
    words = ["go", "system", "front", "back", "full"]

    assert set(words) <= ENGLISH_STOP_WORDS
    assert tokens(" ".join(words)) == words


# --- documents --------------------------------------------------------------


def test_the_job_document_is_title_description_then_both_skill_lists() -> None:
    job = make_job()

    assert job_text(job).split("\n") == [
        "Data Engineer",
        "Build and run batch pipelines.",
        "Python Apache Spark",
        "dbt",
    ]


def test_the_candidate_document_is_title_resume_then_skills() -> None:
    candidate = make_candidate()

    assert candidate_text(candidate).split("\n") == [
        "Analyst",
        "Built dashboards in Tableau.",
        "SQL Tableau",
    ]


def test_a_candidate_without_a_title_does_not_become_the_word_none() -> None:
    text = candidate_text(make_candidate(current_title=None))

    assert "none" not in tokens(text)
    assert tokens(text) == tokens("Built dashboards in Tableau. SQL Tableau")


def test_the_orm_models_and_the_synthetic_ones_give_the_same_documents(corpus) -> None:
    for job in corpus.jobs:
        orm = make_job(
            title=job.title,
            description=job.description,
            required_skills=list(job.required_skills),
            preferred_skills=list(job.preferred_skills),
        )
        assert job_text(orm) == job_text(job), job.ref

    for person in corpus.candidates[:300]:
        orm = make_candidate(
            current_title=person.current_title,
            resume_text=person.resume_text,
            skills=list(person.skills),
        )
        assert candidate_text(orm) == candidate_text(person), person.ref


def test_the_documents_never_read_demographics_or_true_skills(corpus) -> None:
    for person in corpus.candidates[:200]:
        changed = dataclasses.replace(
            person,
            gender="nonbinary" if person.gender != "nonbinary" else "woman",
            age_band="40_and_over" if person.age_band == "under_40" else "under_40",
            true_skills=frozenset(),
        )
        assert candidate_text(changed) == candidate_text(person)


# --- vectoriser settings ----------------------------------------------------


def test_a_repeated_term_counts_for_more_than_once_but_less_than_twice() -> None:
    """sublinear_tf: a skill in both the resume and the skills list must not count double."""
    vectoriser = make_vectoriser().fit(["python sql", "python sql"])
    row = vectoriser.transform(["python python sql"])
    weight = {term: row[0, column] for term, column in vectoriser.vocabulary_.items()}

    # Both terms have the same idf, so the ratio is the term-frequency weighting alone.
    assert weight["python"] / weight["sql"] == pytest.approx(1.6931, abs=1e-4)


def test_a_term_found_in_one_document_only_is_not_learned() -> None:
    vectoriser = make_vectoriser().fit(["python sql", "python rust"])

    assert set(vectoriser.vocabulary_) == {"python"}


def test_documents_are_scaled_to_unit_length() -> None:
    """The matrix takes a dot product as the cosine, which needs this."""
    vectoriser = make_vectoriser().fit(["python sql airflow", "python sql airflow"])
    row = vectoriser.transform(["python python python sql airflow"])

    assert row.multiply(row).sum() == pytest.approx(1.0)


def test_each_vectoriser_is_new_so_fitted_state_is_never_shared() -> None:
    make_vectoriser().fit(["python sql", "python sql"])

    with pytest.raises(NotFittedError):
        make_vectoriser().transform(["python"])


def test_the_planted_proxy_reaches_the_vocabulary(corpus) -> None:
    """Deliberately not stripped: the fairness audit needs to see what a model does with it."""
    vectoriser = make_vectoriser().fit([candidate_text(c) for c in corpus.candidates])

    assert "women" in vectoriser.vocabulary_


def test_the_proxy_comes_only_from_the_planted_phrase() -> None:
    corpus = generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200, proxy_share=0.0))
    vectoriser = make_vectoriser().fit([candidate_text(c) for c in corpus.candidates])

    assert "women" not in vectoriser.vocabulary_
