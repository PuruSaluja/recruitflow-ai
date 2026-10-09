"""Group fairness of a ranking: who makes each slate's shortlist, by group.

A matcher does not accept or reject anyone; it orders a job's slate, and a
recruiter reads the top of it. So **selection here is a place in a slate's top
k** (:func:`top_k_selection`), and every metric compares how often each group
gets one:

* **Selection rate** -- the share of a group's pairs that are shortlisted.
* **Disparate impact ratio** -- a group's selection rate over the reference
  group's. The *four-fifths rule* of US employment guidance treats a ratio
  below 0.8 as evidence of adverse impact; it is a rule of thumb for a first
  look, not a legal test, and it says nothing about why the rates differ.
* **Equal-opportunity difference** -- a group's true-positive rate minus the
  reference group's, where the true-positive rate is the share of the group's
  *qualified* pairs (relevance at least 1) that are shortlisted. Selection
  rates can differ for a fair reason, when one group is more often qualified;
  among the qualified they should not. A negative value means qualified
  members of the group are shortlisted less often than qualified members of
  the reference group.

**Ties at the cut are shared, not broken**, as in :mod:`app.ml.evaluate`: when
a group of equal scores straddles position k, each of its members counts as
selected with probability (places left) / (group size), so a scorer that ties
everyone shortlists every pair with probability k / slate size, which is the
exact expectation of a random shortlist.

Every rate pools pairs across jobs. The 95% intervals resample **jobs**, the
unit that was sampled, and recompute each figure from the resampled pairs, so
they are wide when a group is small (nonbinary candidates are 5% of the pool).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: A disparate impact ratio below this fails the four-fifths rule.
FOUR_FIFTHS = 0.8

#: Lowest relevance grade that counts as qualified for equal opportunity.
QUALIFIED_GRADE = 1


def top_k_selection(job_refs: Sequence[str], scores: Sequence[float], k: int) -> np.ndarray:
    """How likely each pair is to be in its own slate's top ``k`` by ``scores``.

    1.0 above the cut, 0.0 below it, and a share for a tie group that straddles
    it. A slate of ``k`` or fewer pairs is shortlisted whole.
    """
    # A shortlist holds whole places: 2.5 would share out half a place, and
    # True would pass for 1.
    if isinstance(k, bool) or not isinstance(k, int | np.integer):
        raise TypeError(f"k must be a whole number, got {k!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    scores_array = np.asarray(scores, dtype=float)
    refs = np.asarray(job_refs)
    if scores_array.ndim != 1 or refs.shape != scores_array.shape:
        raise ValueError(
            f"job_refs and scores must be two flat sequences of one length, got shapes "
            f"{refs.shape} and {scores_array.shape}"
        )
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must be finite; NaN and infinity cannot be ranked")

    selected = np.zeros(len(scores_array), dtype=float)
    for ref in np.unique(refs):
        rows = np.flatnonzero(refs == ref)
        order = rows[np.argsort(-scores_array[rows], kind="stable")]
        ordered = scores_array[order]
        start = 0
        while start < len(order) and start < k:
            end = start + 1
            while end < len(order) and ordered[end] == ordered[start]:
                end += 1
            selected[order[start:end]] = (min(end, k) - start) / (end - start)
            start = end
    return selected


@dataclass(frozen=True)
class GroupRates:
    """One group's selection, pooled over every job."""

    pairs: int
    #: Expected number of the group's pairs on a shortlist.
    selected: float
    selection_rate: float
    #: Pairs graded at least :data:`QUALIFIED_GRADE`.
    qualified: int
    #: Share of the qualified pairs on a shortlist; ``None`` when none are qualified.
    true_positive_rate: float | None


@dataclass(frozen=True)
class Interval:
    value: float | None
    #: 95% bootstrap interval over jobs; ``None`` when ``value`` is.
    low: float | None
    high: float | None


@dataclass(frozen=True)
class GroupAudit:
    rates: GroupRates
    #: Selection rate over the reference group's.
    disparate_impact: Interval
    #: ``None`` for the reference group itself, whose ratio is 1 by definition.
    passes_four_fifths: bool | None
    #: True-positive rate minus the reference group's.
    equal_opportunity_difference: Interval


@dataclass(frozen=True)
class FairnessAudit:
    reference: str
    groups: dict[str, GroupAudit]
    #: Lowest disparate impact ratio of any group against the reference.
    min_disparate_impact: float | None


def group_rates(
    selected: Sequence[float], groups: Sequence[str], relevance: Sequence[int]
) -> dict[str, GroupRates]:
    """Selection and true-positive rates for each group, in sorted group order."""
    selected_array, group_array, qualified = _validated(selected, groups, relevance)
    rates = {}
    for group in sorted(set(group_array.tolist())):
        mask = group_array == group
        n_qualified = int(qualified[mask].sum())
        picked = float(selected_array[mask].sum())
        rates[group] = GroupRates(
            pairs=int(mask.sum()),
            selected=picked,
            selection_rate=picked / int(mask.sum()),
            qualified=n_qualified,
            true_positive_rate=(
                float(selected_array[mask & qualified].sum()) / n_qualified if n_qualified else None
            ),
        )
    return rates


def audit_selection(
    job_refs: Sequence[str],
    selected: Sequence[float],
    groups: Sequence[str],
    relevance: Sequence[int],
    *,
    reference: str,
    resamples: int = 2000,
    seed: int = 0,
) -> FairnessAudit:
    """Every metric for every group against ``reference``, with intervals over jobs."""
    selected_array, group_array, qualified = _validated(selected, groups, relevance)
    refs = np.asarray(job_refs)
    if refs.shape != selected_array.shape:
        raise ValueError(f"{len(refs)} job refs for {len(selected_array)} pairs")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    names = sorted(set(group_array.tolist()))
    if reference not in names:
        raise ValueError(f"reference group {reference!r} is not among the groups {names}")

    # Per job and group: pairs, selected, qualified pairs, qualified selected.
    # A bootstrap sample of jobs is then a sum of rows, not a pass over pairs.
    jobs, job_index = np.unique(refs, return_inverse=True)
    group_index = np.searchsorted(names, group_array)
    counts = np.zeros((4, len(jobs), len(names)))
    for layer, weights in enumerate(
        (np.ones_like(selected_array), selected_array, qualified, selected_array * qualified)
    ):
        np.add.at(counts[layer], (job_index, group_index), weights)

    point = _ratios(counts.sum(axis=1), names.index(reference))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(jobs), size=(resamples, len(jobs)))
    samples = [_ratios(counts[:, draw].sum(axis=1), names.index(reference)) for draw in draws]

    rates = group_rates(selected_array, group_array, qualified.astype(int))
    audits = {}
    for g, name in enumerate(names):
        impact = point[0][g]
        audits[name] = GroupAudit(
            rates=rates[name],
            disparate_impact=_interval(impact, [sample[0][g] for sample in samples]),
            passes_four_fifths=(
                None if name == reference or impact is None else bool(impact >= FOUR_FIFTHS)
            ),
            equal_opportunity_difference=_interval(
                point[1][g], [sample[1][g] for sample in samples]
            ),
        )
    others = [audits[name].disparate_impact.value for name in names if name != reference]
    known = [value for value in others if value is not None]
    return FairnessAudit(
        reference=reference, groups=audits, min_disparate_impact=min(known) if known else None
    )


def _ratios(totals: np.ndarray, reference: int) -> tuple[list[float | None], list[float | None]]:
    """Disparate impact and equal-opportunity difference of each group, from summed counts."""
    pairs, picked, qualified, qualified_picked = totals
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(pairs > 0, picked / pairs, np.nan)
        tpr = np.where(qualified > 0, qualified_picked / qualified, np.nan)
    impact = [
        None
        if not (np.isfinite(r) and np.isfinite(rate[reference]) and rate[reference] > 0)
        else float(r / rate[reference])
        for r in rate
    ]
    difference = [
        None if not (np.isfinite(t) and np.isfinite(tpr[reference])) else float(t - tpr[reference])
        for t in tpr
    ]
    return impact, difference


def _interval(value: float | None, samples: Sequence[float | None]) -> Interval:
    known = [sample for sample in samples if sample is not None]
    if value is None or not known:
        return Interval(value, None, None)
    low, high = np.quantile(known, [0.025, 0.975])
    return Interval(value, float(low), float(high))


def _validated(
    selected: Sequence[float], groups: Sequence[str], relevance: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_array = np.asarray(selected, dtype=float)
    group_array = np.asarray(groups)
    relevance_array = np.asarray(relevance)
    if not (
        selected_array.ndim == 1
        and selected_array.shape == group_array.shape
        and selected_array.shape == relevance_array.shape
    ):
        raise ValueError(
            f"selected, groups and relevance must be flat sequences of one length, got shapes "
            f"{selected_array.shape}, {group_array.shape} and {relevance_array.shape}"
        )
    if len(selected_array) == 0:
        raise ValueError("there are no pairs to audit")
    if not np.all((selected_array >= 0.0) & (selected_array <= 1.0)):
        raise ValueError("selected must hold shares between 0 and 1")
    return selected_array, group_array, relevance_array >= QUALIFIED_GRADE
