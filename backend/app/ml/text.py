"""The text half of the matcher's input: what a job and a candidate say, as TF-IDF.

The engineered features in :mod:`app.ml.features` read structured fields; this
reads the words. Each side contributes one document -- a job its title,
description and skill lists, a candidate their current title, resume and listed
skills -- and both are vectorised with one vocabulary so they can be compared.

Three settings are deliberate:

* **Text is folded before it is tokenised.** Only uploaded resumes are
  normalised on the way in (:mod:`app.services.resume`); a resume sent as JSON
  and every job description arrive as typed. Without folding, one word could
  become two terms: ``certiﬁed`` with a ligature, ``Ｐｙｔｈｏｎ`` in full
  width, ``café`` in its other Unicode form, or ``Py`` and ``thon`` either side
  of a zero-width space.
* **The token pattern keeps skill names whole.** scikit-learn's default drops
  one-character tokens and punctuation, so ``C++``, ``C#`` and ``R`` vanish and
  ``Node.js`` splits in two. This pattern keeps ``+`` and ``#`` inside a token
  and a ``.`` between two word characters, but still splits on ``/`` and ``-``,
  so ``Python/SQL`` stays two skills. A ``+`` between two words joins them as
  well, so ``Python+SQL`` stays one token.
* **There is no stop-word list.** scikit-learn's English list contains ``go``,
  ``system``, ``front``, ``back`` and ``full`` -- a language, and half of
  "System Design", "front-end", "back-end" and "full-stack". Inverse document
  frequency already discounts words that every document uses.

Nothing here reads demographics, but the resume is read as written: a phrase in
it that stands in for gender reaches the vocabulary like any other word.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer


class JobText(Protocol):
    """The job fields the text features read."""

    title: str
    description: str
    required_skills: Sequence[str]
    preferred_skills: Sequence[str]


class CandidateText(Protocol):
    """The candidate fields the text features read."""

    current_title: str | None
    resume_text: str
    skills: Sequence[str]


#: Word characters, with ``+`` and ``#`` allowed after the first and ``.`` between two.
TOKEN_PATTERN = r"(?u)\b\w[\w+#]*(?:\.\w[\w+#]*)*"

# The characters the resume normaliser strips for the same reason: they survive
# NFKC and split a word in two. Soft hyphen, ZWSP, ZWNJ, ZWJ, word joiner, BOM.
_INVISIBLE_RE = re.compile("[­​‌‍⁠﻿]")


def fold_text(text: str) -> str:
    """Fold ``text`` so one word has one spelling: NFKC, no invisible characters, casefolded."""
    return _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def job_text(job: JobText) -> str:
    """The job as one document: title, description, then both skill lists."""
    return "\n".join(
        [job.title, job.description, " ".join(job.required_skills), " ".join(job.preferred_skills)]
    )


def candidate_text(candidate: CandidateText) -> str:
    """The candidate as one document: current title, resume, then the listed skills."""
    return "\n".join(
        [candidate.current_title or "", candidate.resume_text, " ".join(candidate.skills)]
    )


def make_vectoriser() -> TfidfVectorizer:
    """A fresh, unfitted vectoriser with the settings every text feature uses.

    ``sublinear_tf`` because a candidate's document names most skills twice,
    once in the resume and once in the skills list; a second mention should
    count for less than the first. ``min_df=2`` because a term that appears in
    one resume only can never describe anyone else.
    """
    return TfidfVectorizer(
        preprocessor=fold_text, token_pattern=TOKEN_PATTERN, sublinear_tf=True, min_df=2
    )
