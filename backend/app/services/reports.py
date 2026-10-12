"""PDF reports over a job's cached ranking: the shortlist, and one candidate's scorecard.

Nothing is scored here. Both reports read rows that ``POST /jobs/{id}/score``
cached, under the matcher currently on disk, so a report says what the
ranking already says -- the same ranks, scores and features -- and each one
names the matcher version and when its scores were taken.

The skill breakdown on a scorecard is worked out from the job and candidate
as they are on file when the report is made, with the same canonicalisation
the matcher's features use. If either was edited after the score was cached,
the breakdown can disagree with the stored features, and the scorecard says
that the score predates the edit rather than leaving the two to contradict
each other silently. That compares the rows' ``updated_at``, stamped by the
database's clock, with ``scored_at``, stamped by the app's, so on separate
hosts the two clocks have to agree.

Text is drawn in fonts embedded in each PDF (``app/assets/fonts``): DejaVu
Sans for Latin with its accents (Polish, Turkish, Vietnamese...), Greek,
Cyrillic, Armenian, Georgian, Hebrew and Arabic; Droid Sans Fallback for
Chinese characters and Japanese kana; Nanum Square for Korean. A character no
face has is drawn without its accents where that leaves one a face has, and as
``"?"`` otherwise (Devanagari, Thai, emoji). ReportLab lays every line out left
to right and does not join Arabic letters, so a value with Hebrew or Arabic in
it is put into the order it is seen in, Arabic in its joined forms, before it is
drawn; copied out of the PDF, those letters can come out reversed. Accents typed
as separate combining marks are composed onto their letter first. Zero-width
spaces, direction marks and control characters are left out, except that
whitespace controls read as spaces. Every field is also escaped, since ReportLab
reads ``<`` and ``&`` in a paragraph as markup and fails on an unclosed tag.
"""

from __future__ import annotations

import io
import itertools
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import arabic_reshaper
from bidi import get_display
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.ml.features import FEATURE_NAMES, skill_keys
from app.ml.taxonomy import normalise_skills, spelling_key
from app.models.candidate import Candidate
from app.models.job import Job
from app.schemas.match import MatchFeatures, RankedCandidate
from app.services.requirements import experience_requirement_note

#: Most rows one shortlist report lists.
MAX_SHORTLIST = 200
#: The shortlist's columns, and their widths across the page's 178 mm, wide enough
#: for each heading and for "200" or "100%" in the embedded font at 8 pt.
SHORTLIST_HEADER = ("#", "Candidate", "Current title", "Years", "Score", "Required", "Preferred")
SHORTLIST_WIDTHS_MM = (10, 48, 51, 14, 15, 20, 20)

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"


@dataclass(frozen=True)
class _Face:
    """A registered font family and the code points every one of its weights draws."""

    family: str
    bold: str
    glyphs: frozenset[int]


def _register(family: str, regular_file: str, bold_file: str | None = None) -> _Face:
    """Register ``family`` and its bold face with ReportLab.

    A family with one weight draws bold text in that weight. The reports use no italics.
    """
    regular = TTFont(family, str(_FONT_DIR / regular_file))
    pdfmetrics.registerFont(regular)
    glyphs = frozenset(regular.face.charToGlyph)
    bold_name = family
    if bold_file is not None:
        bold_name = f"{family}-Bold"
        bold = TTFont(bold_name, str(_FONT_DIR / bold_file))
        pdfmetrics.registerFont(bold)
        glyphs &= frozenset(bold.face.charToGlyph)
    pdfmetrics.registerFontFamily(
        family, normal=family, bold=bold_name, italic=family, boldItalic=bold_name
    )
    return _Face(family, bold_name, glyphs)


#: Tried in order for each character; the first that has it draws it. DejaVu Sans
#: draws the reports' own text, Droid Sans Fallback Chinese characters and Japanese
#: kana, Nanum Square Korean Hangul.
_FACES = (
    _register(FONT, "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    _register("DroidSansFallback", "DroidSansFallbackFull.ttf"),
    _register("NanumSquare", "NanumSquareR.ttf", "NanumSquareB.ttf"),
)
#: Right-to-left letters. ReportLab lays every line out left to right, so a value
#: holding any is put into the order it is seen in before it is drawn.
_RIGHT_TO_LEFT = {"R", "AL"}
#: ReportLab writes a code point past U+FFFF into the PDF's text map as five hex
#: digits, which readers reject, so it could not be copied or searched: emoji and
#: the like are treated as missing, though the font has some.
_LAST_MAPPABLE = 0xFFFF
_STYLES = getSampleStyleSheet()
_BODY = ParagraphStyle("body", parent=_STYLES["BodyText"], fontName=FONT, fontSize=9, leading=11)
_SMALL = ParagraphStyle("small", parent=_BODY, fontSize=7.5, leading=9, textColor=colors.grey)
_CELL = ParagraphStyle("cell", parent=_BODY, fontSize=8, leading=10)
_TITLE = ParagraphStyle("title", parent=_STYLES["Title"], fontName=FONT_BOLD)
_HEADING = ParagraphStyle("heading", parent=_STYLES["Heading2"], fontName=FONT_BOLD)

_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF2")),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#C5CCD6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8FA")]),
    ]
)


#: Format characters (zero-width spaces, direction marks) and combining marks still
#: loose after composing: nothing to draw, so they are left out rather than drawn as "?".
_INVISIBLE = {"Cf", "Mn", "Me"}


def drawable(value: object, *, bold: bool = False) -> str:
    """``value`` as text the embedded fonts can draw, escaped for a ReportLab paragraph.

    Characters the first face lacks are wrapped in ``<font face=...>`` for the face
    that has them. That tag sets the weight too, overriding ``<b>`` and a bold style,
    so text drawn in bold has to say so with ``bold``.
    """
    runs: list[tuple[_Face, str]] = []
    # Composed first, so a letter typed as "e" plus a combining accent is drawn as
    # the one letter the font has rather than as "e?".
    text = unicodedata.normalize("NFC", str(value))
    if any(unicodedata.bidirectional(char) in _RIGHT_TO_LEFT for char in text):
        # Arabic letters into the joined forms they take beside their neighbours,
        # which ReportLab would not choose, then every right-to-left run reversed
        # into the order it is seen in, around left-to-right text.
        text = get_display(arabic_reshaper.reshape(text), base_dir="L")
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc":  # controls: whitespace reads as a space, the rest draw a box
            if char.isspace():
                runs.append((_FACES[0], " "))
            continue
        if category in _INVISIBLE:
            continue
        face = _face_for(char)
        if face is not None:
            runs.append((face, char))
            continue
        folded = "".join(
            c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c)
        )
        faces = [_face_for(c) for c in folded]
        if folded and None not in faces:
            runs.extend(zip(faces, folded, strict=True))
        else:
            runs.append((_FACES[0], "?"))
    out = []
    for face, group in itertools.groupby(runs, key=lambda run: run[0]):
        text = escape("".join(char for _, char in group))
        if face is not _FACES[0]:
            text = f'<font face="{face.bold if bold else face.family}">{text}</font>'
        out.append(text)
    return "".join(out)


def _face_for(char: str) -> _Face | None:
    """The first face that can draw ``char``, or ``None``."""
    if ord(char) > _LAST_MAPPABLE:
        return None
    for face in _FACES:
        if ord(char) in face.glyphs:
            return face
    return None


@dataclass(frozen=True)
class SkillBreakdown:
    """Which of a job's skills a candidate lists, in the job's own order."""

    met: list[str]
    missing: list[str]

    @property
    def coverage(self) -> float:
        """Share met; 1.0 when nothing is asked for, as the coverage features define it."""
        wanted = len(self.met) + len(self.missing)
        return len(self.met) / wanted if wanted else 1.0


def skill_breakdown(wanted: Iterable[str], held: Iterable[str]) -> SkillBreakdown:
    """Split ``wanted`` into the skills ``held`` covers and the rest, compared as features do."""
    held_keys = skill_keys(held)
    met: list[str] = []
    missing: list[str] = []
    for name in normalise_skills(wanted):
        (met if spelling_key(name) in held_keys else missing).append(name)
    return SkillBreakdown(met=met, missing=missing)


def _para(text: str, style: ParagraphStyle = _BODY) -> Paragraph:
    return Paragraph(text, style)


def _requirement_notes(job: Job) -> list[Paragraph]:
    """The job's requirement note under its line, when it has one (see ``requirements``)."""
    note = experience_requirement_note(job.seniority, job.min_years_experience)
    return [_para(f"<b>Check:</b> {drawable(note)}")] if note else []


def _stamp(moment: datetime) -> str:
    """``moment`` in UTC, converted rather than relabelled whatever offset it carries."""
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _job_line(job: Job) -> str:
    parts = [job.seniority.value, job.employment_type.value.replace("_", " ")]
    parts += [part for part in (job.department, job.location) if part]
    return drawable(" · ".join(parts))


def _footer(model_version: str, generated_at: datetime) -> Paragraph:
    return _para(
        f"Matcher {drawable(model_version)} · generated {_stamp(generated_at)}. "
        "A score is the matcher's calibrated probability that the candidate would be "
        "advanced for this job; it supports a recruiter's review and does not replace it.",
        _SMALL,
    )


def _render(story: list, title: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
        author="RecruitFlow AI",
        # The canvas otherwise selects Helvetica at the top of each page, which puts
        # an unembedded font in the file even though no text is drawn in it.
        initialFontName=FONT,
    )
    doc.build(story)
    return buffer.getvalue()


def shortlist_pdf(
    job: Job,
    rows: Sequence[RankedCandidate],
    *,
    total: int,
    model_version: str,
    generated_at: datetime,
) -> bytes:
    """The job's ranked shortlist: the best ``rows`` of ``total`` cached, best first."""
    if not rows:
        raise ValueError("a shortlist needs at least one ranked candidate")
    table = [list(SHORTLIST_HEADER)]
    for row in rows:
        features = row.features
        table.append(
            [
                str(row.rank),
                _para(drawable(row.candidate.full_name), _CELL),
                _para(drawable(row.candidate.current_title or "-"), _CELL),
                str(row.candidate.years_experience),
                f"{row.score:.3f}",
                f"{features.required_skill_coverage:.0%}",
                f"{features.preferred_skill_coverage:.0%}",
            ]
        )
    grid = Table(table, colWidths=[w * mm for w in SHORTLIST_WIDTHS_MM], repeatRows=1)
    grid.setStyle(_TABLE_STYLE)

    oldest = min(row.scored_at for row in rows)
    story = [
        _para(f"Shortlist: {drawable(job.title, bold=True)}", _TITLE),
        _para(f"Job {job.id} · {_job_line(job)}"),
        *_requirement_notes(job),
        _para(
            f"The top {len(rows)} of {total} candidates scored for this job, best first. "
            f"Required and Preferred are the shares of the job's required and preferred "
            f"skills each candidate lists. Scores taken from {_stamp(oldest)}."
        ),
        Spacer(1, 4 * mm),
        grid,
        Spacer(1, 6 * mm),
        _footer(model_version, generated_at),
    ]
    return _render(story, f"Shortlist - {job.title}")


def _skill_lines(label: str, breakdown: SkillBreakdown) -> list[Paragraph]:
    """Paragraphs rather than table cells: a table row cannot break across pages, so a
    long enough skill list in one cell would not fit on any page and fail the report."""
    return [
        _para(f"<b>{label}, {heading}:</b> {drawable(', '.join(names)) if names else '-'}")
        for heading, names in (("listed", breakdown.met), ("not listed", breakdown.missing))
    ]


def scorecard_pdf(
    job: Job,
    candidate: Candidate,
    ranked: RankedCandidate,
    *,
    total: int,
    model_version: str,
    generated_at: datetime,
) -> bytes:
    """One candidate's scorecard for one job: rank, score, skills and the features behind it."""
    if ranked.candidate.id != candidate.id:
        raise ValueError(f"the ranking is for candidate {ranked.candidate.id}, not {candidate.id}")
    required = skill_breakdown(job.required_skills, candidate.skills)
    preferred = skill_breakdown(job.preferred_skills, candidate.skills)

    profile = Table(
        [
            ["", "Job asks for", "Candidate"],
            [
                "Experience",
                f"{job.min_years_experience}+ years",
                f"{candidate.years_experience} years",
            ],
            ["Seniority", job.seniority.value, candidate.seniority.value],
            [
                "Education",
                job.education_required.value.replace("_", " "),
                candidate.education_level.value.replace("_", " "),
            ],
        ],
        colWidths=[22 * mm, 78 * mm, 78 * mm],
    )
    profile.setStyle(_TABLE_STYLE)

    values = ranked.features.model_dump()
    features = Table(
        [["Feature", "Value", "Meaning"]]
        + [
            [
                name.replace("_", " "),
                f"{values[name]:.3f}",
                _para(drawable(MatchFeatures.model_fields[name].description or ""), _CELL),
            ]
            for name in FEATURE_NAMES
        ],
        colWidths=[42 * mm, 16 * mm, 120 * mm],
        repeatRows=1,
    )
    features.setStyle(_TABLE_STYLE)

    story = [
        _para(f"Scorecard: {drawable(candidate.full_name, bold=True)}", _TITLE),
        _para(
            f"For job {job.id}, {drawable(job.title)} ({_job_line(job)}). "
            f"Candidate {candidate.id}"
            + (f", {drawable(candidate.current_title)}" if candidate.current_title else "")
            + "."
        ),
        Spacer(1, 3 * mm),
        _para(
            f"<b>Rank {ranked.rank} of {total}</b> · score <b>{ranked.score:.3f}</b> · "
            f"scored {_stamp(ranked.scored_at)}"
        ),
    ]
    edited = [
        what
        for what, row in (("job", job), ("candidate", candidate))
        if row.updated_at is not None and row.updated_at > ranked.scored_at
    ]
    if edited:
        story.append(
            _para(
                f"The {' and the '.join(edited)} changed after this score was taken; "
                "the skills and profile below are as on file now, the score and features "
                "as they were then. Score the job again to bring them together."
            )
        )
    story += [
        _para("Skills", _HEADING),
        _para(
            f"Lists {len(required.met)} of {len(required.met) + len(required.missing)} "
            f"required and {len(preferred.met)} of "
            f"{len(preferred.met) + len(preferred.missing)} preferred skills."
        ),
        *_skill_lines("Required", required),
        *_skill_lines("Preferred", preferred),
        _para("Profile", _HEADING),
        profile,
        _para("Features behind the score", _HEADING),
        _para(
            "The engineered half of the matcher's input. The other half, TF-IDF over the "
            "resume and the job description, is not broken out per score."
        ),
        features,
        Spacer(1, 6 * mm),
        _footer(model_version, generated_at),
    ]
    return _render(story, f"Scorecard - {candidate.full_name} - {job.title}")
