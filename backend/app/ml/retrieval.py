"""Retrieval scorers fitted to no label, and the fusion that combines them.

Three ways to rank a pool of candidates for a job without fitting anything to
a label:

* **Lexical, Okapi BM25.** The job's document is the query and each
  candidate's document is searched. Words are folded and tokenised exactly as
  the TF-IDF features do (:func:`app.ml.text.fold_text`,
  :data:`app.ml.text.TOKEN_PATTERN`), so ``C++`` and ``Node.js`` stay whole
  here too. ``k1 = 1.2`` and ``b = 0.75`` are the textbook defaults, not tuned.
  The inverse document frequency is the non-negative form Lucene uses,
  ``log(1 + (N - n + 0.5) / (n + 0.5))``, so a term in more than half the pool
  still counts for a little rather than against a candidate. Each distinct
  query term counts once: a job description naming Python in its text and in
  its required skills does not double Python's weight.
* **Dense.** Cosine between the job's and the candidate's sentence embeddings
  (:mod:`app.ml.encoder`, :func:`app.ml.embeddings.cosine_scores`).
* **Reciprocal rank fusion** (Cormack, Clarke and Buettcher, 2009). Each
  scorer ranks the pool on its own; a candidate's fused score is
  ``sum(1 / (k + rank))`` over the scorers, with ``k = 60`` as in the paper.
  Only ranks are fused, so BM25's unbounded scores and cosine's [-1, 1] need
  no common scale. Tied scores share their average rank, so a fused score never
  depends on the order the candidates happen to arrive in. Fusion makes ties
  of its own: two candidates whose ranks are swapped between the scorers, 3rd
  and 7th against 7th and 3rd, get exactly equal fused scores.

**Fusion depends on the pool it ranks.** A candidate's rank among 50 is not
its rank among 4,000, and fusing ranks is not monotone in either scorer, so
fusing over a job's slate and fusing over the whole pool can order the same
slate differently. :func:`fuse_per_group` ranks within each group (a job's
slate, in the evaluation), which is what ranking a slate means.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer

from app.ml.text import TOKEN_PATTERN, fold_text

#: Term-frequency saturation and length normalisation, the usual defaults.
BM25_K1 = 1.2
BM25_B = 0.75

#: The rank offset in reciprocal rank fusion, as in Cormack et al. (2009).
RRF_K = 60


class BM25Index:
    """Okapi BM25 over a fixed pool of documents, scored one query at a time.

    The per-document term weights are computed once, so scoring a query is a
    sparse row lookup and a sum. A query term the pool never uses adds nothing.
    """

    def __init__(self, documents: Sequence[str], *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        if isinstance(documents, str):
            raise TypeError("a BM25 index takes a sequence of documents, not one string")
        if not documents:
            raise ValueError("a BM25 index needs at least one document")
        if not (math.isfinite(k1) and k1 >= 0):
            raise ValueError(f"k1 must be a finite number of at least 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be between 0 and 1, got {b}")
        self._vectoriser = CountVectorizer(preprocessor=fold_text, token_pattern=TOKEN_PATTERN)
        try:
            counts = self._vectoriser.fit_transform(documents).tocsr().astype(np.float64)
        except ValueError as error:  # every document empty, or only punctuation
            raise ValueError(f"a BM25 index needs at least one word to search: {error}") from error
        n_documents = counts.shape[0]
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = lengths.mean()
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log1p((n_documents - document_frequency + 0.5) / (document_frequency + 0.5))
        # tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average)), times the idf.
        norms = k1 * (1.0 - b + b * lengths / average_length)
        rows = np.repeat(np.arange(n_documents), np.diff(counts.indptr))
        tf = counts.data
        counts.data = self.idf[counts.indices] * tf * (k1 + 1.0) / (tf + norms[rows])
        self._weights = counts.tocsc()
        self.size = n_documents

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document in the pool for ``query``, in pool order."""
        terms = self._vectoriser.transform([query]).indices
        return np.asarray(self._weights[:, terms].sum(axis=1)).ravel()


def reciprocal_rank_fusion(*scores: Sequence[float], k: float = RRF_K) -> np.ndarray:
    """Fused score of each item ranked by every one of ``scores``: ``sum(1 / (k + rank))``.

    Rank 1 is the highest score. Equal scores share their average rank.
    """
    if not scores:
        raise ValueError("fusion needs at least one ranking")
    if not (math.isfinite(k) and k >= 0):
        raise ValueError(f"k must be a finite number of at least 0, got {k}")
    arrays = [np.asarray(s, dtype=float) for s in scores]
    length = arrays[0].shape
    if any(a.ndim != 1 or a.shape != length for a in arrays):
        raise ValueError(
            f"every ranking must score the same items, got shapes {[a.shape for a in arrays]}"
        )
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("scores must be finite; NaN cannot be ranked")
    return sum(1.0 / (k + rankdata(-a, method="average")) for a in arrays)


def fuse_per_group(groups: Sequence[str], *scores: Sequence[float], k: float = RRF_K) -> np.ndarray:
    """:func:`reciprocal_rank_fusion` within each group, returned in the input order."""
    keys = np.asarray(groups)
    if keys.ndim != 1:
        raise ValueError(f"groups must hold one key per item, got shape {keys.shape}")
    arrays = [np.asarray(s, dtype=float) for s in scores]
    if any(a.shape != keys.shape for a in arrays):
        raise ValueError(f"{len(keys)} group keys for scores of shapes {[a.shape for a in arrays]}")
    fused = np.empty(len(keys), dtype=float)
    for key in np.unique(keys):
        mask = keys == key
        fused[mask] = reciprocal_rank_fusion(*(a[mask] for a in arrays), k=k)
    return fused
