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

**Two stages.** A cheap scorer keeps its top ``n`` and a precise one re-orders
only those (:func:`rerank_top_n`); the rest stay below, in the first stage's
order. A first stage is judged by what it keeps, :func:`recall_at_n`, since
the second can only promote a candidate the first let through. Candidates
tied with the ``n``-th score are all kept, so the cut never depends on the
order the candidates arrive in, and a stage that scores everyone equal keeps
everyone; :func:`first_stage_cut` is that rule, shared by both.
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


def _group_keys(groups: Sequence[object]) -> np.ndarray:
    """``groups`` as an array, refusing keys the conversion would merge or split.

    numpy stores ``[1, "1"]`` as ``["1", "1"]``, which would rank two groups
    as one; each distinct key has to stay exactly one distinct key.
    """
    keys = np.asarray(groups)
    if keys.ndim != 1:
        raise ValueError(f"groups must hold one key per item, got shape {keys.shape}")
    originals, converted = list(groups), keys.tolist()
    if (
        not len(set(originals))
        == len(set(converted))
        == len(set(zip(originals, converted, strict=True)))
    ):
        raise ValueError(
            "groups must use distinct group keys of one kind; "
            f"{sorted(map(repr, set(originals)))} do not stay distinct as an array"
        )
    return keys


def fuse_per_group(groups: Sequence[str], *scores: Sequence[float], k: float = RRF_K) -> np.ndarray:
    """:func:`reciprocal_rank_fusion` within each group, returned in the input order."""
    keys = _group_keys(groups)
    arrays = [np.asarray(s, dtype=float) for s in scores]
    if any(a.shape != keys.shape for a in arrays):
        raise ValueError(f"{len(keys)} group keys for scores of shapes {[a.shape for a in arrays]}")
    fused = np.empty(len(keys), dtype=float)
    for key in np.unique(keys):
        mask = keys == key
        fused[mask] = reciprocal_rank_fusion(*(a[mask] for a in arrays), k=k)
    return fused


def _finite_scores(scores: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty flat sequence, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite; NaN cannot be ranked")
    return array


def _check_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be a whole number of at least 1, got {n!r}")


def first_stage_cut(scores: Sequence[float], n: int) -> np.ndarray:
    """Which items a first stage keeping its top ``n`` lets through, as a mask.

    Every item scoring at least the ``n``-th highest score is kept, so ties at
    the cut make the kept set larger than ``n``; ``n`` past the end keeps all.
    """
    array = _finite_scores(scores, "scores")
    _check_n(n)
    if n >= len(array):
        return np.ones(len(array), dtype=bool)
    threshold = np.sort(array)[::-1][n - 1]
    return array >= threshold


def recall_at_n(scores: Sequence[float], relevant: Sequence[bool], n: int) -> float | None:
    """Share of the ``relevant`` items that :func:`first_stage_cut` keeps.

    ``None`` when nothing is relevant, as there is nothing to keep.
    """
    hits = np.asarray(relevant)
    if hits.dtype != bool:
        raise ValueError(f"relevant must be booleans, got dtype {hits.dtype}")
    kept = first_stage_cut(scores, n)
    if hits.shape != kept.shape:
        raise ValueError(f"{len(kept)} scores for relevance of shape {hits.shape}")
    total = int(hits.sum())
    return None if total == 0 else int(hits[kept].sum()) / total


def rerank_top_n(first: Sequence[float], second: Sequence[float], n: int) -> np.ndarray:
    """Scores that rank the kept items by ``second``, all above the rest by ``first``.

    The kept items are :func:`first_stage_cut` of ``first``. Only ranks are
    returned, as scores: the rest get their average rank by ``first`` (1 for
    the lowest), the kept ones ``len(first) + 1`` plus their average rank by
    ``second`` among themselves, so ties within either stage stay ties.
    ``second`` is only read for the kept items; the rest may hold anything.
    """
    first_array = _finite_scores(first, "first-stage scores")
    second_array = np.asarray(second, dtype=float)
    if second_array.shape != first_array.shape:
        raise ValueError(
            f"both stages must score the same items, got shapes {first_array.shape} "
            f"and {second_array.shape}"
        )
    kept = first_stage_cut(first_array, n)
    kept_scores = _finite_scores(second_array[kept], "second-stage scores of the kept items")
    out = np.empty(len(first_array), dtype=float)
    if (~kept).any():
        out[~kept] = rankdata(first_array[~kept], method="average")
    out[kept] = len(first_array) + 1 + rankdata(kept_scores, method="average")
    return out


def rerank_per_group(
    groups: Sequence[str], first: Sequence[float], second: Sequence[float], n: int
) -> np.ndarray:
    """:func:`rerank_top_n` within each group, returned in the input order."""
    _check_n(n)
    keys = _group_keys(groups)
    first_array, second_array = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if keys.ndim != 1 or first_array.shape != keys.shape or second_array.shape != keys.shape:
        raise ValueError(
            f"{keys.shape} group keys for scores of shapes {first_array.shape} "
            f"and {second_array.shape}"
        )
    out = np.empty(len(keys), dtype=float)
    for key in np.unique(keys):
        mask = keys == key
        out[mask] = rerank_top_n(first_array[mask], second_array[mask], n)
    return out
