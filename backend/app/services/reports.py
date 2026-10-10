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

Text is drawn in the PDF's built-in Helvetica, which only covers Windows-1252.
Characters outside it are drawn without their accents where that leaves a
character the font has (``"ś"`` as ``"s"``), and as ``"?"`` otherwise
(``"Ł"``, any CJK). Accents typed as separate combining marks are composed
onto their letter first; control characters other than whitespace,
zero-width and direction marks are left out. Every field is also escaped,
since ReportLab reads ``<`` and ``&`` in a paragraph as markup and fails on
an unclosed tag.
"""

from __future__ import annotations

import io
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.ml.features import FEATURE_NAMES, skill_keys
from app.ml.taxonomy import normalise_skills, spelling_key
from app.models.candidate import Candidate
from app.models.job import Job
from app.schemas.match import MatchFeatures, RankedCandidate

#: Most rows one shortlist report lists.
MAX_SHORTLIST = 200

_STYLES = getSampleStyleSheet()
_BODY = ParagraphStyle("body", parent=_STYLES["BodyText"], fontSize=9, leading=11)
_SMALL = ParagraphStyle("small", parent=_BODY, fontSize=7.5, leading=9, textColor=colors.grey)
_CELL = ParagraphStyle("cell", parent=_BODY, fontSize=8, leading=10)
_TITLE = _STYLES["Title"]
_HEADING = _STYLES["Heading2"]

_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#C5CCD6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8FA")]),
    ]
)


#: Format characters (zero-width spaces, direction marks) and combining marks still
#: loose after composing: nothing to draw, so they are left out rather than drawn as "?".
_INVISIBLE = {"Cf", "Mn", "Me"}


def drawable(value: object) -> str:
    """``value`` as text Helvetica can draw, escaped for a ReportLab paragraph."""
    out = []
    # Composed first, so a letter typed as "e" plus a combining accent is drawn as
    # the one letter the font has rather than as "e?".
    for char in unicodedata.normalize("NFC", str(value)):
        category = unicodedata.category(char)
        if category == "Cc":  # controls: whitespace reads as a space, the rest draw a box
            if char.isspace():
                out.append(" ")
            continue
        if category in _INVISIBLE:
            continue
        if _encodable(char):
            out.append(char)
            continue
        folded = "".join(
            c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c)
        )
        out.append(folded if folded and all(map(_encodable, folded)) else "?")
    return escape("".join(out))


def _encodable(char: str) -> bool:
    try:
        char.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


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
    header = ["#", "Candidate", "Current title", "Years", "Score", "Required", "Preferred"]
    table = [header]
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
    widths = [w * mm for w in (9, 48, 55, 13, 15, 19, 19)]
    grid = Table(table, colWidths=widths, repeatRows=1)
    grid.setStyle(_TABLE_STYLE)

    oldest = min(row.scored_at for row in rows)
    story = [
        _para(f"Shortlist: {drawable(job.title)}", _TITLE),
        _para(f"Job {job.id} · {_job_line(job)}"),
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
        _para(f"Scorecard: {drawable(candidate.full_name)}", _TITLE),
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
