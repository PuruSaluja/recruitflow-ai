"""Resume ingestion: turn an uploaded file into clean, matchable text.

Whatever the matcher eventually does with a resume, it sees this module's
output first. Extraction artefacts -- soft hyphens, "fi" ligatures, Word's
private-use bullet glyph, page footers repeated on every page -- all become
tokens downstream, so they are cheaper to remove once, here, than to work
around in the feature pipeline later.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

#: Resumes are a page or two. Anything larger is a mistake or an attack.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

#: Extensions accepted by the upload endpoints. A file with no extension is
#: treated as plain text; content sniffing overrides the extension for PDFs.
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".text", ".pdf"})

#: Mirrors ``CandidateBase.resume_text``'s ``min_length``. Below this a PDF is
#: almost certainly a scan with no text layer.
MIN_EXTRACTED_CHARS = 20

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"

# Zero-width and soft-hyphen characters survive NFKC and glue tokens together:
# soft hyphen, ZWSP, ZWNJ, ZWJ, word joiner, BOM.
_INVISIBLE_RE = re.compile("[­​‌‍⁠﻿]")

# C0/C1 control characters other than tab and newline. PDF text extraction
# emits these wherever a glyph is missing from the exporting font's encoding,
# and they would otherwise be stored verbatim and fed to the vectoriser.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")

# Bullet glyphs are rewritten only at the start of a line: a mid-sentence "·"
# is punctuation, but a leading one is list structure. The list covers what Word,
# LaTeX and Google Docs actually emit -- including U+F0B7, the private-use
# codepoint Word uses for Symbol-font bullets, and U+25A0, which is what a small
# square bullet degrades to when the exporting font has no U+25AA glyph.
_BULLET_LINE_RE = re.compile(
    "^[ \t]*[•‣⁃▪▫▬■□▸▶●○◦·∙–—➢➤]+[ \t]*",
    re.MULTILINE,
)

# "Page 2", "Page 2 of 4", "2 of 4", "2 / 4" -- footer noise repeated on every
# page. A bare number on its own line is deliberately *not* matched: in a resume
# that is far more often a date column ("2019") than a page number.
_PAGE_MARKER_RE = re.compile(
    r"^[ \t]*(?:page[ \t]*\d+(?:[ \t]*(?:of|/)[ \t]*\d+)?|\d+[ \t]*(?:of|/)[ \t]*\d+)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


class ResumeIngestError(ValueError):
    """Raised when an upload cannot be turned into usable resume text.

    The API surfaces the message verbatim as a 400, so every message here is
    written for the person who uploaded the file, not for a log reader.
    """


def normalise_resume_text(text: str) -> str:
    """Collapse extraction artefacts into plain, consistently spaced text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # NFKC folds ligatures (U+FB01 becomes "fi"), full-width characters and
    # non-breaking spaces into their ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    # A form feed is a PDF page boundary; keep it as a paragraph break.
    text = text.replace("\f", "\n\n")
    text = _PAGE_MARKER_RE.sub("", text)
    text = _BULLET_LINE_RE.sub("- ", text)
    # Re-join words split across a line break. Dropping the hyphen is right for
    # justified wrapping ("respon-\nsibility") and wrong for genuine compounds
    # ("full-\nstack"); the first is far more common in extracted PDF text, and
    # the sklearn token pattern splits on hyphens either way.
    text = _HYPHEN_BREAK_RE.sub("", text)
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def extract_text(filename: str | None, data: bytes) -> str:
    """Extract normalised resume text from an uploaded file's bytes.

    Raises:
        ResumeIngestError: the file is empty, too large, in an unsupported
            format, or yields too little text to be a resume.
    """
    if not data:
        raise ResumeIngestError("The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ResumeIngestError(
            f"Resume is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )

    suffix = Path(filename or "").suffix.lower()

    # Sniff before trusting the extension: a resume mailed as "resume.txt" that
    # is really a PDF is common enough to handle rather than reject.
    if data.startswith(_PDF_MAGIC):
        raw = _extract_pdf(data)
    elif data.startswith(_ZIP_MAGIC):
        raise ResumeIngestError(
            "DOCX and other zip-based formats are not supported -- "
            "export the resume as PDF or plain text first"
        )
    elif suffix == ".pdf":
        raise ResumeIngestError("File is named .pdf but does not contain a PDF header")
    elif suffix in SUPPORTED_EXTENSIONS or not suffix:
        raw = _decode(data)
    else:
        raise ResumeIngestError(
            f"Unsupported resume format '{suffix}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    text = normalise_resume_text(raw)
    if len(text) < MIN_EXTRACTED_CHARS:
        raise ResumeIngestError(
            "No usable text could be extracted "
            f"({len(text)} characters, minimum {MIN_EXTRACTED_CHARS}). "
            "If this is a scanned PDF it has no text layer and needs OCR first."
        )
    return text


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        # An empty owner password covers PDFs encrypted only to restrict
        # printing, which is a common export default.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeIngestError("The PDF is password-protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ResumeIngestError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ResumeIngestError(f"The PDF could not be read: {exc}") from exc

    if not pages:
        raise ResumeIngestError("The PDF has no pages")
    return "\f".join(pages)


def _decode(data: bytes) -> str:
    """Decode text bytes, trying the encodings resumes actually arrive in."""
    encodings = ["utf-8-sig", "cp1252"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.insert(0, "utf-16")

    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(text):
            return text
        raise ResumeIngestError(
            "The file does not look like text -- upload a PDF or a plain-text resume"
        )

    raise ResumeIngestError("The file is not valid UTF-8, UTF-16 or Windows-1252 text")


def _looks_like_text(text: str, threshold: float = 0.02) -> bool:
    """Reject binary that happens to decode, e.g. a renamed image."""
    if not text:
        return False
    control = sum(1 for ch in text if ch not in "\n\r\t\f" and unicodedata.category(ch) == "Cc")
    return control / len(text) <= threshold
