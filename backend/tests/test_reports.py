"""Unit tests for the PDF report builders. No database: jobs and candidates are built in memory."""

from __future__ import annotations

import io
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pypdf import PdfReader
from reportlab.lib.fonts import tt2ps
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.features import FEATURE_NAMES, feature_dict
from app.models import Candidate, Job
from app.models.enums import EducationLevel, EmploymentType, SeniorityLevel
from app.schemas.candidate import CandidateSummary
from app.schemas.match import MatchFeatures, RankedCandidate
from app.services.reports import (
    FONT,
    FONT_BOLD,
    SHORTLIST_HEADER,
    SHORTLIST_WIDTHS_MM,
    drawable,
    scorecard_pdf,
    shortlist_pdf,
    skill_breakdown,
)

SCORED_AT = datetime(2026, 10, 10, 6, 30, tzinfo=UTC)
GENERATED_AT = datetime(2026, 10, 10, 7, 0, tzinfo=UTC)


def _job(**overrides: object) -> Job:
    fields: dict[str, object] = {
        "id": 7,
        "title": "Senior Data Engineer",
        "department": "Data",
        "location": "Toronto",
        "employment_type": EmploymentType.FULL_TIME,
        "seniority": SeniorityLevel.SENIOR,
        "education_required": EducationLevel.BACHELOR,
        "min_years_experience": 5,
        "description": "Build pipelines.",
        "required_skills": ["Python", "postgres", "Airflow"],
        "preferred_skills": ["dbt", "Kafka"],
        "updated_at": SCORED_AT - timedelta(days=1),
    }
    return Job(**(fields | overrides))


def _candidate(candidate_id: int = 3, **overrides: object) -> Candidate:
    fields: dict[str, object] = {
        "id": candidate_id,
        "full_name": f"Candidate {candidate_id}",
        "email": f"c{candidate_id}@example.com",
        "current_title": "Data Engineer",
        "years_experience": 6,
        "seniority": SeniorityLevel.MID,
        "education_level": EducationLevel.MASTER,
        "resume_text": "Pipelines.",
        "skills": ["python", "PostgreSQL", "Kafka"],
        "updated_at": SCORED_AT - timedelta(hours=1),
    }
    return Candidate(**(fields | overrides))


def _ranked(candidate: Candidate, job: Job, rank: int, score: float) -> RankedCandidate:
    return RankedCandidate(
        rank=rank,
        score=score,
        scored_at=SCORED_AT,
        candidate=CandidateSummary.model_validate(candidate),
        features=MatchFeatures(**feature_dict(job, candidate)),
    )


def _text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


# --- skills ----------------------------------------------------------------


def test_the_breakdown_canonicalises_both_sides_and_keeps_the_jobs_order() -> None:
    breakdown = skill_breakdown(["Python", "postgres", "Airflow"], ["PostgreSQL", "python"])

    assert breakdown.met == ["Python", "PostgreSQL"]
    assert breakdown.missing == ["Apache Airflow"]


def test_nothing_asked_for_is_full_coverage() -> None:
    assert skill_breakdown([], ["Python"]).coverage == 1.0


def test_the_breakdown_agrees_with_the_coverage_features_on_every_corpus_pair() -> None:
    corpus = generate_corpus(CorpusConfig(seed=11, n_jobs=10, n_candidates=1200))
    for job in corpus.jobs:
        for candidate in corpus.candidates:
            features = feature_dict(job, candidate)
            required = skill_breakdown(job.required_skills, candidate.skills)
            preferred = skill_breakdown(job.preferred_skills, candidate.skills)
            assert required.coverage == pytest.approx(features["required_skill_coverage"])
            assert preferred.coverage == pytest.approx(features["preferred_skill_coverage"])


# --- text ------------------------------------------------------------------


def test_markup_in_a_field_is_drawn_as_text() -> None:
    job = _job(title="R&D <lead")
    candidate = _candidate(full_name="Ann <b>Lee</b> & Co")
    text = _text(
        shortlist_pdf(
            job,
            [_ranked(candidate, job, 1, 0.5)],
            total=1,
            model_version="abc",
            generated_at=GENERATED_AT,
        )
    )

    assert "R&D <lead" in text
    assert "Ann <b>Lee</b> & Co" in text


@pytest.mark.parametrize(
    "name",
    [
        "Zoë Müller",
        "Łukasz Šimić",  # Polish and Croatian letters Windows-1252 lacks
        "Ğül Şahin",  # Turkish
        "Nguyễn Văn Đức",  # Vietnamese, with stacked accents
        "Ελένη Παπαδοπούλου",  # Greek
        "Дмитрий Иванов",  # Cyrillic
        "Արամ",  # Armenian
        "ნინო",  # Georgian
    ],
)
def test_names_in_the_scripts_the_font_has_are_drawn_as_typed(name: str) -> None:
    """Read back from the PDF, not just from drawable: the embedded font has to carry each
    letter's text mapping too, or the name could not be copied or searched."""
    assert drawable(name) == name
    job = _job()
    pdf = shortlist_pdf(
        job,
        [_ranked(_candidate(full_name=name), job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    assert name in _text(pdf)


@pytest.mark.parametrize(
    ("name", "drawn"),
    [
        ("李雷", '<font face="DroidSansFallback">李雷</font>'),  # Chinese, simplified
        ("陳淑芬", '<font face="DroidSansFallback">陳淑芬</font>'),  # and traditional
        ("山田さくら", '<font face="DroidSansFallback">山田さくら</font>'),  # Japanese kanji, kana
        ("김민준", '<font face="NanumSquare">김민준</font>'),  # Korean Hangul
        ("Anna 王", 'Anna <font face="DroidSansFallback">王</font>'),  # only what DejaVu lacks
    ],
)
def test_chinese_japanese_and_korean_are_drawn_in_the_fallback_faces(name: str, drawn: str) -> None:
    """Read back from both reports, the scorecard's bold title included, and drawn in the
    face that has the characters rather than as "?"."""
    assert drawable(name) == drawn
    job = _job()
    candidate = _candidate(full_name=name)
    ranked = _ranked(candidate, job, 1, 0.5)
    shortlist = shortlist_pdf(job, [ranked], total=1, model_version="v", generated_at=GENERATED_AT)
    scorecard = scorecard_pdf(
        job, candidate, ranked, total=1, model_version="v", generated_at=GENERATED_AT
    )

    assert name in _text(shortlist)
    assert f"Scorecard: {name}" in _text(scorecard)
    face = drawn.split('face="')[1].split('"')[0]
    regular, bold = (_embedded_name(face), _embedded_name(_bold_of(face)))
    assert regular in _fonts(shortlist)  # the table cell
    assert bold in _fonts(scorecard)  # the bold title


def _embedded_name(registered: str) -> str:
    """The name a registered font is embedded under: its own, not the registered one."""
    return pdfmetrics.getFont(registered).face.name.decode()


def _bold_of(family: str) -> str:
    return tt2ps(family, 1, 0)


@pytest.mark.parametrize(
    ("name", "drawn"),
    [
        # Each right-to-left run reversed into the order it is seen in.
        ("יעל כהן", "ןהכ לעי"),
        # Arabic in the joined forms its letters take beside their neighbours.
        ("محمد علي", "\ufef2\ufee0\ufecb \ufeaa\ufee4\ufea4\ufee3"),
        # Left-to-right text around it keeps its place and order.
        ("Anna محمد Lee", "Anna \ufeaa\ufee4\ufea4\ufee3 Lee"),
        ("Dr. יעל", "Dr. לעי"),
    ],
)
def test_hebrew_and_arabic_are_drawn_in_the_order_they_are_read(name: str, drawn: str) -> None:
    """What the page holds, read off it by position. ReportLab lays each line out left to
    right, so the value is put in display order first; copying it out of the PDF can give
    the right-to-left letters reversed, which the README says."""
    assert drawable(name) == drawn
    job = _job()
    pdf = shortlist_pdf(
        job,
        [_ranked(_candidate(full_name=name), job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    page = PdfReader(io.BytesIO(pdf)).pages[0].extract_text(extraction_mode="layout")
    assert drawn in page
    assert _fonts(pdf) == {"DejaVuSans", "DejaVuSans-Bold"}


@pytest.mark.parametrize(
    ("name", "face"),
    [
        # Vowel signs and the virama are combining marks; they are kept, not dropped.
        ("प्रिया कृष्णन", "NotoSansDevanagari"),
        ("ณัฐวุฒิ ศรีสุข", "NotoSansThai"),
    ],
)
def test_devanagari_and_thai_are_drawn_in_their_own_faces(name: str, face: str) -> None:
    words = name.split()
    assert drawable(name) == " ".join(f'<font face="{face}">{word}</font>' for word in words)
    assert drawable(name, bold=True).startswith(f'<font face="{face}-Bold">')

    job = _job()
    pdf = shortlist_pdf(
        job,
        [_ranked(_candidate(full_name=name), job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    assert f"{face}-Regular" in _fonts(pdf)


def test_devanagari_is_shaped_into_its_conjuncts() -> None:
    """Shaped, a conjunct or a reph is a glyph of its own, which the PDF can only map
    back to a private-use character; drawn letter by letter, every one would map back
    to itself. Copying such a name out of the PDF does not give it back exactly."""
    job = _job()
    pdf = shortlist_pdf(
        job,
        [_ranked(_candidate(full_name="शर्मा"), job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    text = _text(pdf)

    assert any("" <= char <= "" for char in text)


@pytest.mark.parametrize(
    ("font", "shaped"),
    [
        ("NotoSansDevanagari", True),
        ("NotoSansDevanagari-Bold", True),
        ("NotoSansThai", True),  # Thai marks sit right only when positioned by HarfBuzz
        ("NotoSansThai-Bold", True),
        ("DejaVuSans", False),
        ("DejaVuSans-Bold", False),
        ("DroidSansFallback", False),
        ("NanumSquare", False),
        ("NanumSquare-Bold", False),
    ],
)
def test_only_the_scripts_that_need_it_are_shaped(font: str, shaped: bool) -> None:
    registered = pdfmetrics.getFont(font)

    if shaped:
        assert registered.hbFont(9) is not None
    else:
        with pytest.raises(AttributeError, match="is not shaped"):
            registered.hbFont(9)


def test_latin_beside_a_shaped_script_is_not_joined_into_ligatures() -> None:
    """A shaped paragraph shapes every word; DejaVu's are left as they are, so
    "Shortlist" is not drawn, or copied out, as "Shortliﬆ", nor "Rafiq" as "Raﬁq"."""
    job = _job(title="Data विश्लेषक")
    pdf = shortlist_pdf(
        job,
        [_ranked(_candidate(full_name="Rafiq สมชาย"), job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    text = _text(pdf)

    assert "Shortlist: Data" in text
    assert "Rafiq" in text
    assert not any("ﬀ" <= char <= "ﬆ" for char in text)


def test_without_harfbuzz_devanagari_and_thai_are_marked_missing() -> None:
    """ReportLab shapes only when uharfbuzz imports; unshaped, these scripts would be
    drawn wrong, so they are drawn as "?" instead. Run in a fresh interpreter, as
    the fonts are registered when the module is imported."""
    code = (
        "import sys; sys.modules['uharfbuzz'] = None\n"
        "from app.services.reports import drawable, _FACES\n"
        "print([face.family for face in _FACES]); print(drawable('राम ไทย').encode())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )

    families, drawn = result.stdout.splitlines()
    assert families == "['DejaVuSans', 'DroidSansFallback', 'NanumSquare']"
    assert drawn == "b'??? ???'"


@pytest.mark.parametrize(
    ("value", "drawn"),
    [
        ("ሰላም", "???"),  # Ethiopic: no embedded face has it
        ("Sam \U0001f600", "Sam ?"),  # in the font, but past U+FFFF it cannot be mapped
        ("\U0001d400lpha", "Alpha"),  # mathematical bold A folds to A
        ("Jose\u0301 Garci\u0301a", "Jos\u00e9 Garc\u00eda"),  # accents composed first
        ("q\u0301", "q"),  # an accent with no composed letter is dropped, not drawn as ?
        ("A\tB\nC", "A B C"),  # whitespace controls become spaces
        ("bell\x07x", "bellx"),  # other controls have nothing to draw
        ("zw\u200bsp \u202eabc", "zwsp abc"),  # nor zero-width or direction marks
    ],
)
def test_characters_the_font_lacks_are_folded_or_replaced(value: str, drawn: str) -> None:
    assert drawable(value) == drawn
    job = _job()
    candidate = _candidate(full_name=value)
    pdf = shortlist_pdf(
        job,
        [_ranked(candidate, job, 1, 0.5)],
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )
    assert drawn in _text(pdf)


# --- shortlist -------------------------------------------------------------


def test_the_shortlist_lists_candidates_in_rank_order_with_their_scores() -> None:
    job = _job()
    people = [_candidate(i, full_name=f"Person {name}") for i, name in enumerate("BCA", start=1)]
    rows = [
        _ranked(p, job, rank, score)
        for rank, (p, score) in enumerate(zip(people, (0.91, 0.62, 0.40), strict=True), start=1)
    ]
    text = _text(
        shortlist_pdf(job, rows, total=9, model_version="0123abcd", generated_at=GENERATED_AT)
    )

    assert "Shortlist: Senior Data Engineer" in text
    assert "top 3 of 9" in text
    positions = [text.index(f"Person {name}") for name in "BCA"]
    assert positions == sorted(positions)
    for score in ("0.910", "0.620", "0.400"):
        assert score in text
    assert "Matcher 0123abcd" in text
    assert "2026-10-10 07:00 UTC" in text


def test_the_shortlist_says_when_the_jobs_minimum_is_long_for_its_level() -> None:
    def shortlist(min_years: int) -> str:
        job = _job(seniority=SeniorityLevel.MID, min_years_experience=min_years)
        ranked = [_ranked(_candidate(), job, 1, 0.5)]
        return _text(
            shortlist_pdf(job, ranked, total=1, model_version="v", generated_at=GENERATED_AT)
        )

    flagged = " ".join(shortlist(8).split())
    assert "Check: The minimum of 8 years' experience is above the 3 a mid role" in flagged
    assert "Check:" not in shortlist(3)


def test_a_long_shortlist_runs_over_pages_and_repeats_its_header() -> None:
    job = _job()
    rows = [_ranked(_candidate(i), job, i, 0.5) for i in range(1, 201)]
    reader = PdfReader(
        io.BytesIO(
            shortlist_pdf(job, rows, total=200, model_version="v", generated_at=GENERATED_AT)
        )
    )

    assert len(reader.pages) > 1
    assert all("Current title" in page.extract_text() for page in reader.pages)
    assert "Candidate 200" in reader.pages[-1].extract_text()


def test_an_empty_shortlist_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        shortlist_pdf(_job(), [], total=0, model_version="v", generated_at=GENERATED_AT)


# --- scorecard -------------------------------------------------------------


def _scorecard(job: Job, candidate: Candidate) -> str:
    return _text(
        scorecard_pdf(
            job,
            candidate,
            _ranked(candidate, job, 3, 0.734),
            total=12,
            model_version="feedbeef",
            generated_at=GENERATED_AT,
        )
    )


def _fonts(pdf: bytes) -> set[str]:
    found = set()
    for page in PdfReader(io.BytesIO(pdf)).pages:
        for font in (page["/Resources"].get("/Font") or {}).values():
            found.add(str(font.get_object()["/BaseFont"]).split("+")[-1])
    return found


def test_the_shortlist_columns_fit_their_headings_and_widest_values() -> None:
    """DejaVu Sans is wider than Helvetica; ReportLab does not wrap a plain cell, it lets the
    text run into the padding and past the border. Default padding is 6 pt a side."""
    widest = ("200", "Candidate", "Current title", "60", "0.999", "100%", "100%")
    assert sum(SHORTLIST_WIDTHS_MM) == 178  # A4 less the two 16 mm margins
    for heading, value, width in zip(SHORTLIST_HEADER, widest, SHORTLIST_WIDTHS_MM, strict=True):
        room = width * mm - 12
        assert stringWidth(heading, FONT_BOLD, 8) <= room, heading
        assert stringWidth(value, FONT, 8) <= room, value


def test_both_reports_draw_every_word_in_the_embedded_font() -> None:
    """Table cells the code fills (ranks, scores, levels) included, not only names."""
    job = _job()
    candidate = _candidate()
    ranked = _ranked(candidate, job, 1, 0.5)
    shortlist = shortlist_pdf(job, [ranked], total=1, model_version="v", generated_at=GENERATED_AT)
    scorecard = scorecard_pdf(
        job, candidate, ranked, total=1, model_version="v", generated_at=GENERATED_AT
    )

    assert _fonts(shortlist) == {"DejaVuSans", "DejaVuSans-Bold"}
    assert _fonts(scorecard) == {"DejaVuSans", "DejaVuSans-Bold"}


def test_the_scorecard_shows_rank_skills_profile_and_every_feature() -> None:
    text = _scorecard(_job(), _candidate())

    assert "Scorecard: Candidate 3" in text
    assert "Rank 3 of 12" in text
    assert "score 0.734" in text
    assert "Lists 2 of 3 required and 1 of 2 preferred skills" in text
    assert "Airflow" in text and "dbt" in text
    assert "5+ years" in text and "6 years" in text
    for name in FEATURE_NAMES:
        assert name.replace("_", " ") in text
    assert "changed after this score" not in text


@pytest.mark.parametrize(
    ("job_edited", "candidate_edited", "said"),
    [
        (True, False, "The job changed"),
        (False, True, "The candidate changed"),
        (True, True, "The job and the candidate changed"),
    ],
)
def test_the_scorecard_says_when_the_score_predates_an_edit(
    job_edited: bool, candidate_edited: bool, said: str
) -> None:
    later = SCORED_AT + timedelta(minutes=5)
    job = _job(**({"updated_at": later} if job_edited else {}))
    candidate = _candidate(**({"updated_at": later} if candidate_edited else {}))

    assert said in _scorecard(job, candidate)


def test_a_scorecard_runs_onto_more_pages_for_the_longest_skill_lists_allowed() -> None:
    """100 skills of 100 characters per list, as the API accepts: no part may need a whole page."""
    required = [f"required skill {i:03d} " + "x" * 81 for i in range(100)]
    preferred = [f"preferred skill {i:03d} " + "y" * 80 for i in range(100)]
    assert {len(name) for name in required + preferred} == {100}
    job = _job(required_skills=required, preferred_skills=preferred)
    candidate = _candidate(skills=required[:50])

    pdf = scorecard_pdf(
        job,
        candidate,
        _ranked(candidate, job, 1, 0.5),
        total=1,
        model_version="v",
        generated_at=GENERATED_AT,
    )

    assert len(PdfReader(io.BytesIO(pdf)).pages) > 1
    text = "".join(_text(pdf).split())
    assert all("".join(name.split()) in text for name in required + preferred)


def test_a_scorecard_for_someone_else_s_ranking_is_refused() -> None:
    job = _job()
    with pytest.raises(ValueError, match="candidate 3, not 4"):
        scorecard_pdf(
            job,
            _candidate(4),
            _ranked(_candidate(3), job, 1, 0.5),
            total=1,
            model_version="v",
            generated_at=GENERATED_AT,
        )


def test_times_are_shown_in_utc_whatever_offset_they_carry() -> None:
    """The label says UTC, so a time given with another offset is converted, not relabelled."""
    eastern = timezone(timedelta(hours=-4))
    job = _job()
    candidate = _candidate()
    ranked = _ranked(candidate, job, 1, 0.5).model_copy(
        update={"scored_at": datetime(2026, 10, 10, 2, 30, tzinfo=eastern)}
    )

    text = _text(
        scorecard_pdf(
            job,
            candidate,
            ranked,
            total=1,
            model_version="v",
            generated_at=datetime(2026, 10, 10, 3, 0, tzinfo=eastern),
        )
    )

    assert "scored 2026-10-10 06:30 UTC" in text
    assert "generated 2026-10-10 07:00 UTC" in text
