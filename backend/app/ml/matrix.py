"""The full feature matrix: one row per job-candidate pair, ready for a model.

A row is three blocks, always in this order:

1. the engineered features, named and ordered by
   :data:`app.ml.features.FEATURE_NAMES`;
2. ``text_similarity``, the cosine between the job's and the candidate's TF-IDF
   vectors;
3. the candidate's own TF-IDF vector, one column per term, named
   ``resume:<term>``.

The vocabulary is learned from the candidates' text alone. A term no resume
uses can never match one, so job-only wording ("we are hiring", "join the
team") would add nothing to any similarity -- it would only lengthen each job's
vector by a different amount, pulling that job's similarities down by a factor
the model cannot see, and add columns that are zero in every row.

The job's own terms are not a block: they are the same for every candidate on a
job's slate, so they could only shift one job's scores against another's,
never reorder a slate.

The resume terms are where a model learns from wording directly, and so where
it can pick up a proxy for gender. The synthetic corpus plants one -- a
women-in-tech community line on some women's resumes -- and ``resume:women`` is
a column like any other, on purpose: the fairness audit exists to measure what a
model does with it, and removing the term here would hide what it is meant to
find.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from app.ml.features import FEATURE_NAMES, CandidateLike, JobLike, feature_vector
from app.ml.text import CandidateText, JobText, candidate_text, job_text, make_vectoriser


class MatchJob(JobLike, JobText, Protocol):
    """Every job field the matrix reads."""


class MatchCandidate(CandidateLike, CandidateText, Protocol):
    """Every candidate field the matrix reads."""


Pair = tuple[MatchJob, MatchCandidate]


class PairFeaturiser(TransformerMixin, BaseEstimator):
    """Turns (job, candidate) pairs into the full feature matrix.

    ``fit`` learns the vocabulary from the candidates in the pairs, each counted
    once however many jobs it was paired with. ``transform`` never changes it:
    a term first seen at scoring time is ignored rather than added, so a model
    trained on one set of columns is always scored on the same set.
    """

    def fit(self, pairs: Sequence[Pair], y: object = None) -> PairFeaturiser:
        candidates = dict.fromkeys(candidate for _, candidate in pairs)
        self.vectoriser_ = make_vectoriser().fit([candidate_text(c) for c in candidates])
        return self

    def transform(self, pairs: Sequence[Pair]) -> sparse.csr_matrix:
        check_is_fitted(self, "vectoriser_")
        # Each job and candidate is vectorised once, however many pairs it is in.
        job_rows: dict[MatchJob, int] = {}
        candidate_rows: dict[MatchCandidate, int] = {}
        job_at = [job_rows.setdefault(job, len(job_rows)) for job, _ in pairs]
        candidate_at = [candidate_rows.setdefault(c, len(candidate_rows)) for _, c in pairs]
        jobs = self.vectoriser_.transform([job_text(job) for job in job_rows])[job_at]
        resumes = self.vectoriser_.transform([candidate_text(c) for c in candidate_rows])
        resumes = resumes[candidate_at]

        # Both sides are L2-normalised, so the row-wise dot product is the cosine.
        # Identical texts land a rounding error past 1.0; the range is [0, 1].
        similarity = np.minimum(jobs.multiply(resumes).sum(axis=1), 1.0)
        engineered = np.array([feature_vector(job, c) for job, c in pairs], dtype=float)

        return sparse.hstack(
            [sparse.csr_matrix(engineered), sparse.csr_matrix(similarity), resumes], format="csr"
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Column names, in matrix order. Changing them invalidates any persisted model."""
        check_is_fitted(self, "vectoriser_")
        terms = self.vectoriser_.get_feature_names_out()
        return np.array(
            [*FEATURE_NAMES, "text_similarity", *(f"resume:{term}" for term in terms)],
            dtype=object,
        )
