"""Unit tests for resume ingestion. No database needed.

Non-ASCII inputs are written as escapes: several are invisible, and a test for
"strip the zero-width space" whose zero-width space was lost in an editor would
compare "Python" to "Python" and pass without testing anything.
"""

from __future__ import annotations

import io

import pytest

from app.services.resume import (
    MAX_RESUME_CHARS,
    MAX_UPLOAD_BYTES,
    ResumeIngestError,
    extract_text,
    normalise_resume_text,
)

FI_LIGATURE = "\ufb01"
NBSP = "\u00a0"
ZERO_WIDTH_SPACE = "\u200b"
SOFT_HYPHEN = "\u00ad"
BOM = "\ufeff"
WORD_SYMBOL_BULLET = "\uf0b7"
MIDDLE_DOT = "\u00b7"
EN_DASH = "\u2013"


def _pdf(*pages: list[str]) -> bytes:
    """Render a real PDF, one list of lines per page."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    for lines in pages:
        y = 720
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- normalise_resume_text ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(f"ef{FI_LIGATURE}cient teams", "efficient teams", id="fi-ligature"),
        pytest.param("\uff30\uff59\uff54\uff48\uff4f\uff4e", "Python", id="full-width"),
        pytest.param(f"New{NBSP}York", "New York", id="non-breaking-space"),
        pytest.param(f"Py{ZERO_WIDTH_SPACE}thon", "Python", id="zero-width-space"),
        pytest.param(f"Java{SOFT_HYPHEN}Script", "JavaScript", id="soft-hyphen"),
        pytest.param(f"{BOM}Summary", "Summary", id="byte-order-mark"),
    ],
)
def test_unicode_artefacts_are_folded_or_removed(raw: str, expected: str) -> None:
    assert raw != expected  # guards against the input degenerating into the output
    assert normalise_resume_text(raw) == expected


def test_control_bytes_are_stripped_but_line_structure_survives() -> None:
    raw = "Skills:\x00 SQL\x07\nTools:\x1b Tableau"

    assert normalise_resume_text(raw) == "Skills: SQL\nTools: Tableau"


@pytest.mark.parametrize(
    "bullet",
    ["\u2022", "\u25aa", "\u25a0", "\u25cf", "\u25e6", EN_DASH, "\u2014", "\u27a2"],
    ids=[
        "bullet",
        "small-square",
        "black-square",
        "circle",
        "white-bullet",
        "en-dash",
        "em-dash",
        "arrowhead",
    ],
)
def test_leading_bullet_glyphs_become_a_plain_dash(bullet: str) -> None:
    assert normalise_resume_text(f"{bullet} Built ETL jobs") == "- Built ETL jobs"


def test_word_symbol_font_bullet_is_normalised() -> None:
    """U+F0B7 is what Word's Symbol-font bullet extracts as; NFKC leaves it alone."""
    assert normalise_resume_text(f"{WORD_SYMBOL_BULLET} Built ETL jobs") == "- Built ETL jobs"


def test_mid_line_punctuation_is_not_mistaken_for_a_bullet() -> None:
    raw = f"Python {MIDDLE_DOT} SQL {MIDDLE_DOT} dbt\nAcme Corp, 2019 {EN_DASH} 2021"

    assert normalise_resume_text(raw) == raw


def test_words_split_by_line_wrapping_are_rejoined() -> None:
    assert normalise_resume_text("Took respon-\nsibility for reporting") == (
        "Took responsibility for reporting"
    )


def test_hyphen_before_a_capitalised_line_is_left_alone() -> None:
    """Only lowercase-to-lowercase breaks are wrap artefacts; "ETL-\\nJobs" may be intended."""
    assert normalise_resume_text("Owned ETL-\nJobs and dashboards") == (
        "Owned ETL-\nJobs and dashboards"
    )


@pytest.mark.parametrize(
    "footer",
    ["Page 2", "page 2 of 4", "PAGE 3/4", "2 of 4", "  2 / 4  "],
)
def test_page_footers_on_their_own_line_are_dropped(footer: str) -> None:
    assert normalise_resume_text(f"Experience\n{footer}\nEducation") == "Experience\n\nEducation"


def test_a_bare_year_is_kept_because_it_is_usually_a_date_not_a_page_number() -> None:
    assert normalise_resume_text("Acme Corp\n2019\nAnalyst") == "Acme Corp\n2019\nAnalyst"


def test_page_mentioned_inside_a_sentence_is_kept() -> None:
    raw = "Shipped the landing page 2 weeks early"

    assert normalise_resume_text(raw) == raw


def test_form_feed_becomes_a_paragraph_break() -> None:
    assert normalise_resume_text("end of page one\fstart of page two") == (
        "end of page one\n\nstart of page two"
    )


def test_whitespace_is_collapsed_and_trimmed() -> None:
    raw = "\r\n\r\n  Senior   Analyst \t\r\n\n\n\n\nLed\t\tthe team   \r\n"

    assert normalise_resume_text(raw) == "Senior Analyst\n\nLed the team"


def test_normalising_is_idempotent() -> None:
    raw = (
        f"{BOM}\u2022 Took respon-\nsibility\nPage 1 of 2\n\n\n\n"
        f"for Py{ZERO_WIDTH_SPACE}thon {MIDDLE_DOT} SQL"
    )
    once = normalise_resume_text(raw)

    assert once != raw
    assert normalise_resume_text(once) == once


# --- extract_text: plain text -----------------------------------------------


def test_utf8_text_with_a_byte_order_mark_is_decoded() -> None:
    data = f"{BOM}Built reporting pipelines in SQL.".encode()

    assert data.startswith(b"\xef\xbb\xbf")
    assert extract_text("resume.txt", data) == "Built reporting pipelines in SQL."


def test_windows_1252_text_is_decoded() -> None:
    text = "Caf\u00e9 manager, r\u00e9sum\u00e9 attached"
    data = text.encode("cp1252")

    assert extract_text("resume.txt", data) == text


def test_utf16_text_is_decoded() -> None:
    data = "Built reporting pipelines in SQL.".encode("utf-16")

    assert extract_text("resume.txt", data) == "Built reporting pipelines in SQL."


def test_a_file_with_no_extension_is_treated_as_text() -> None:
    assert extract_text("resume", b"Built reporting pipelines in SQL.") == (
        "Built reporting pipelines in SQL."
    )


# --- extract_text: PDF --------------------------------------------------------


def test_pdf_text_is_extracted_across_pages_with_footers_dropped() -> None:
    data = _pdf(
        ["EXPERIENCE", "Built reporting pipelines in SQL", "Page 1 of 2"],
        ["EDUCATION", "B.S. Computer Science", "Page 2 of 2"],
    )

    text = extract_text("resume.pdf", data)

    assert "Built reporting pipelines in SQL" in text
    assert "B.S. Computer Science" in text
    assert "Page 1 of 2" not in text
    assert "Page 2 of 2" not in text
    assert text.index("EXPERIENCE") < text.index("EDUCATION")


def test_a_pdf_misnamed_as_txt_is_still_read_as_a_pdf() -> None:
    data = _pdf(["Built reporting pipelines in SQL"])

    assert "Built reporting pipelines in SQL" in extract_text("resume.txt", data)


def test_a_password_protected_pdf_is_rejected_with_a_clear_message() -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf(["Confidential resume text"]))))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(ResumeIngestError, match="password-protected"):
        extract_text("resume.pdf", buffer.getvalue())


def test_a_corrupt_pdf_is_rejected_rather_than_crashing() -> None:
    with pytest.raises(ResumeIngestError, match=r"^The PDF"):
        extract_text("resume.pdf", b"%PDF-1.7\nthis is not really a pdf body at all")


def test_a_pdf_needing_an_uninstalled_crypto_backend_is_rejected_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pypdf raises DependencyError, which subclasses Exception directly.

    It sits outside the PyPdfError tree, so neither PdfReadError nor PyPdfError
    catches it and it has to be named. Reachable whenever an AES-encrypted PDF
    arrives on an install without `cryptography`, where it escaped as a 500.
    """
    from pypdf.errors import DependencyError

    def missing_backend(*args: object, **kwargs: object) -> None:
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr("pypdf.PdfReader", missing_backend)

    with pytest.raises(ResumeIngestError, match="could not be read"):
        extract_text("resume.pdf", _pdf(["Built reporting pipelines in SQL"]))


# --- extract_text: rejections -------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        pytest.param("resume.txt", b"", "empty", id="empty"),
        pytest.param(
            "resume.txt", b"a" * (MAX_UPLOAD_BYTES + 1), "larger than the 2 MB limit", id="too-big"
        ),
        pytest.param("resume.docx", b"PK\x03\x04rest-of-zip", "DOCX", id="docx"),
        pytest.param(
            "resume.pdf", b"plain text pretending", "does not contain a PDF header", id="fake-pdf"
        ),
        pytest.param(
            "resume.exe",
            b"MZ executable bytes here",
            "Unsupported resume format '.exe'",
            id="unsupported-extension",
        ),
        pytest.param(
            "resume.txt",
            b"\x00\x01\x02\x03" * 50 + b"hello",
            "does not look like text",
            id="binary",
        ),
        pytest.param("resume.txt", b"Too short", "No usable text", id="too-short"),
        pytest.param(
            "resume.txt",
            f"\u2022 {BOM} {ZERO_WIDTH_SPACE} Page 1".encode(),
            "No usable text",
            id="only-artefacts",
        ),
    ],
)
def test_unusable_uploads_are_rejected(filename: str, data: bytes, message: str) -> None:
    with pytest.raises(ResumeIngestError, match=message):
        extract_text(filename, data)


def test_a_pdf_with_no_pages_is_rejected_by_name() -> None:
    from pypdf import PdfWriter

    buffer = io.BytesIO()
    PdfWriter().write(buffer)

    with pytest.raises(ResumeIngestError, match="The PDF has no pages"):
        extract_text("resume.pdf", buffer.getvalue())


def test_bytes_no_accepted_encoding_can_decode_are_rejected_naming_all_three() -> None:
    # 0x81 starts no UTF-8 sequence and is one of the five bytes Windows-1252 leaves
    # undefined, so neither decoder takes the file.
    data = b"Senior data engineer, eight years \x81 Python and SQL pipelines."

    with pytest.raises(ResumeIngestError, match="not valid UTF-8, UTF-16 or Windows-1252"):
        extract_text("resume.txt", data)


def test_upload_exactly_at_the_size_limit_is_accepted() -> None:
    # Padded with trailing spaces, which normalising strips, so the file sits
    # on the byte limit while its text sits on the character limit.
    data = b"a" * MAX_RESUME_CHARS + b" " * (MAX_UPLOAD_BYTES - MAX_RESUME_CHARS)

    assert len(data) == MAX_UPLOAD_BYTES
    assert len(extract_text("resume.txt", data)) == MAX_RESUME_CHARS


def test_text_over_the_character_limit_is_rejected_and_the_message_gives_both_lengths() -> None:
    data = b"a" * (MAX_RESUME_CHARS + 1)

    with pytest.raises(ResumeIngestError) as excinfo:
        extract_text("resume.txt", data)

    assert str(excinfo.value) == (
        "The extracted text is 100,001 characters, over the 100,000-character limit for a resume"
    )


def test_the_character_limit_applies_to_the_text_after_normalising() -> None:
    # 100,000 letters with a soft hyphen inside: 100,001 characters raw,
    # 100,000 once the invisible character is removed.
    data = ("a" * 50_000 + SOFT_HYPHEN + "a" * 50_000).encode()

    assert len(extract_text("resume.txt", data)) == MAX_RESUME_CHARS


def test_the_character_limit_applies_to_pdf_text_too() -> None:
    """A small PDF can hold far more text than a resume: the byte cap is no guard."""
    line = "Built reporting pipelines in SQL and Python. " * 2
    data = _pdf(*[[line] * 40 for _ in range(30)])  # 30 pages of 40 lines of 90 characters

    assert len(data) < MAX_UPLOAD_BYTES // 20
    with pytest.raises(ResumeIngestError, match="100,000-character limit"):
        extract_text("resume.pdf", data)
