"""Held-out evaluation of a matcher: per-slate ranking metrics and pooled ROC-AUC.

The matcher is trained on ``advanced`` (see :mod:`app.ml.train`), but what it is
for is ranking a job's slate so that the people who truly fit come first. So
the ranking metrics here grade a slate against ``relevance``, the generator's
ground truth, which the matcher never sees:

* **NDCG@10** with graded gains ``2**relevance - 1`` -- a strong fit (2) is
  worth three partial ones (1), and a non-fit nothing -- divided by the best
  DCG the job's own slate allows. A job with no relevant candidate on its slate
  has no ideal ordering to compare against, so it is left out of the mean and
  counted instead.
* **P@5**, the share of the top five who are at least a partial fit. The
  denominator is always five, so a slate shorter than five cannot score 1.

Both are computed per job over the job's whole held-out slate and then
averaged over jobs, so a job with a long slate does not count for more.
ROC-AUC is pooled over every held-out pair, the way the training script
already reports it, against ``advanced`` (the target), ``relevance >= 1`` and
``relevance == 2``.

**Ties are averaged, not broken.** Within a group of equal scores every order
is taken as equally likely, and the metric is its expected value over those
orders: each position the group spans gets the group's mean gain. That is
exact, and it is what scikit-learn's ``ndcg_score`` does by default. It matters
for baselines built from one coarse feature, where whole groups of candidates
tie, and it makes "every score equal" the exact expected value of a random
ranking rather than one lucky or unlucky draw. The calibrated matcher itself
is strictly monotone, so its scores do not tie.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from app.ml.train import TrainingData

#: The cut-off for NDCG.
NDCG_K = 10

#: The cut-off for precision.
PRECISION_K = 5

#: Lowest relevance grade that counts as relevant for P@k and ROC-AUC.
RELEVANT_GRADE = 1

#: The grade a strong fit has.
STRONG_GRADE = 2


def gain(relevance: np.ndarray) -> np.ndarray:
    """Graded gain, ``2**relevance - 1``: 0, 1 and 3 for grades 0, 1 and 2."""
    return np.exp2(np.asarray(relevance, dtype=float)) - 1.0


def _expected_by_position(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``values`` placed in descending score order, each tie group given its mean.

    Position ``i`` of the result is the expected value at rank ``i + 1`` when
    the order within every group of equal scores is uniformly random.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_values = values[order].astype(float)
    result = np.empty_like(sorted_values)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        result[start:end] = sorted_values[start:end].mean()
        start = end
    return result


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg_at_k(scores: Sequence[float], relevance: Sequence[int], k: int = NDCG_K) -> float | None:
    """NDCG@k of one slate ranked by ``scores``; ``None`` when no candidate is relevant."""
    scores_array, relevance_array = _validated(scores, relevance, k)
    gains = gain(relevance_array)
    ideal = np.sort(gains)[::-1][:k]
    ideal_dcg = float(ideal @ _discounts(len(ideal)))
    if ideal_dcg == 0.0:
        return None
    ranked = _expected_by_position(scores_array, gains)[:k]
    return float(ranked @ _discounts(len(ranked))) / ideal_dcg


def precision_at_k(
    scores: Sequence[float], relevance: Sequence[int], k: int = PRECISION_K
) -> float:
    """Share of the top ``k`` by ``scores`` who are at least :data:`RELEVANT_GRADE`."""
    scores_array, relevance_array = _validated(scores, relevance, k)
    hits = (relevance_array >= RELEVANT_GRADE).astype(float)
    return float(_expected_by_position(scores_array, hits)[:k].sum()) / k


def _validated(
    scores: Sequence[float], relevance: Sequence[int], k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores_array = np.asarray(scores, dtype=float)
    relevance_array = np.asarray(relevance, dtype=int)
    if scores_array.shape != relevance_array.shape or scores_array.ndim != 1:
        raise ValueError(
            f"scores and relevance must be two flat sequences of one length, got shapes "
            f"{scores_array.shape} and {relevance_array.shape}"
        )
    if len(scores_array) == 0:
        raise ValueError("a slate needs at least one candidate")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must be finite; NaN cannot be ranked")
    return scores_array, relevance_array


@dataclass(frozen=True)
class SlateMetrics:
    """One held-out job's ranking metrics."""

    job_ref: str
    size: int
    #: ``None`` when nobody on the slate is relevant.
    ndcg: float | None
    precision: float


@dataclass(frozen=True)
class Evaluation:
    """Everything measured for one set of scores over one held-out split."""

    slates: tuple[SlateMetrics, ...]
    #: Mean NDCG@10 over the jobs that have at least one relevant candidate.
    ndcg_at_10: float
    #: Mean P@5 over every job.
    precision_at_5: float
    #: Pooled ROC-AUC against each label; ``None`` when that label has one class.
    roc_auc_advanced: float | None
    roc_auc_relevant: float | None
    roc_auc_strong: float | None

    @property
    def jobs_without_relevant(self) -> int:
        return sum(slate.ndcg is None for slate in self.slates)


def per_slate(data: TrainingData, scores: Sequence[float]) -> tuple[SlateMetrics, ...]:
    """NDCG@10 and P@5 for every job in ``data``, ordered by job ref."""
    scores_array = np.asarray(scores, dtype=float)
    if len(scores_array) != len(data):
        raise ValueError(f"{len(scores_array)} scores for {len(data)} pairs")
    refs = np.array(data.job_refs)
    slates = []
    for ref in sorted(set(data.job_refs)):
        mask = refs == ref
        slate_scores, slate_relevance = scores_array[mask], data.relevance[mask]
        slates.append(
            SlateMetrics(
                job_ref=ref,
                size=int(mask.sum()),
                ndcg=ndcg_at_k(slate_scores, slate_relevance, NDCG_K),
                precision=precision_at_k(slate_scores, slate_relevance, PRECISION_K),
            )
        )
    return tuple(slates)


def evaluate_scores(data: TrainingData, scores: Sequence[float]) -> Evaluation:
    """Rank every held-out slate by ``scores`` and measure it against the ground truth."""
    slates = per_slate(data, scores)
    ndcgs = [slate.ndcg for slate in slates if slate.ndcg is not None]
    if not ndcgs:
        raise ValueError("no held-out job has a relevant candidate, so NDCG is undefined")
    scores_array = np.asarray(scores, dtype=float)
    return Evaluation(
        slates=slates,
        ndcg_at_10=float(np.mean(ndcgs)),
        precision_at_5=float(np.mean([slate.precision for slate in slates])),
        roc_auc_advanced=safe_auc(data.advanced, scores_array),
        roc_auc_relevant=safe_auc(data.relevance >= RELEVANT_GRADE, scores_array),
        roc_auc_strong=safe_auc(data.relevance == STRONG_GRADE, scores_array),
    )


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC, or ``None`` when ``labels`` hold only one class to score against."""
    if len(set(np.asarray(labels).tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_interval(
    values: Sequence[float], *, seed: int = 0, resamples: int = 2000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-job ``values``.

    Held-out jobs are the unit that was sampled, so they are what gets
    resampled. With 50 held-out jobs the interval is wide, and it says so.
    """
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(resamples, len(array)))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)
