"""Bias mitigations for the matcher, measured by ``scripts/audit_fairness.py``.

The matcher the app serves uses two of these (see :func:`app.ml.train.make_matcher`
and ``docs/fairness_audit.md`` for why those two); the audit measures the
others beside them. Two act on the input and one on the training data:

* :class:`TermDroppingFeaturiser` leaves named terms out of the vocabulary --
  the "delete the proxy words" fix. It only works for the words someone thought
  to list, and on the synthetic corpus it barely helps.
* :class:`LineDroppingFeaturiser` removes every line that starts with a named
  heading before vectorising, so nothing written on that line can become a
  column. On the synthetic corpus every community line starts
  ``Community:``; real resumes rarely label such a section so reliably.
* :func:`reweighing_weights` (Kamiran and Calders, 2012) weights each
  training pair so that, in the weighted data, the group and the label are
  independent: each (group, label) cell is weighted by the count it would have
  if they were, over the count it has. It needs each training candidate's
  group, but the model never sees it as a feature and scoring does not need it.

:class:`WeightedPipeline` is the pipeline that lets training weights reach the
classifier through calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from app.ml.matrix import Pair, PairFeaturiser
from app.ml.text import candidate_text, fold_text, make_vectoriser


class TermDroppingFeaturiser(PairFeaturiser):
    """:class:`PairFeaturiser` with ``dropped_terms`` left out of the vocabulary.

    Terms are matched after folding, as the vectoriser sees them.
    """

    def __init__(self, dropped_terms: Sequence[str] = ()) -> None:
        self.dropped_terms = dropped_terms

    def fit(self, pairs: Sequence[Pair], y: object = None) -> TermDroppingFeaturiser:
        candidates = dict.fromkeys(candidate for _, candidate in pairs)
        stop_words = sorted({fold_text(term) for term in self.dropped_terms}) or None
        self.vectoriser_ = make_vectoriser().set_params(stop_words=stop_words)
        self.vectoriser_.fit([candidate_text(c) for c in candidates])
        return self


class _WithoutLines:
    """Folds a text, then drops every line that starts with one of ``prefixes``.

    A class rather than a closure so a fitted featuriser can be pickled.
    """

    def __init__(self, prefixes: Sequence[str]) -> None:
        self.prefixes = tuple(fold_text(prefix) for prefix in prefixes)

    def __call__(self, text: str) -> str:
        lines = fold_text(text).split("\n")
        return "\n".join(line for line in lines if not line.lstrip().startswith(self.prefixes))


class LineDroppingFeaturiser(PairFeaturiser):
    """:class:`PairFeaturiser` that never reads a line starting with ``dropped_headings``.

    The lines are dropped from every document, job and candidate alike, before
    tokenising, so they reach neither the resume columns nor
    ``text_similarity``. Headings are matched after folding, ignoring leading
    spaces.
    """

    def __init__(self, dropped_headings: Sequence[str] = ()) -> None:
        self.dropped_headings = dropped_headings

    def fit(self, pairs: Sequence[Pair], y: object = None) -> LineDroppingFeaturiser:
        candidates = dict.fromkeys(candidate for _, candidate in pairs)
        self.vectoriser_ = make_vectoriser().set_params(
            preprocessor=_WithoutLines(self.dropped_headings)
        )
        self.vectoriser_.fit([candidate_text(c) for c in candidates])
        return self


def reweighing_weights(groups: Sequence[str], labels: Sequence[bool]) -> np.ndarray:
    """Kamiran-Calders weights: ``P(group) * P(label) / P(group, label)`` for each pair.

    The weights sum to the number of pairs, and in the weighted data every
    group has the same positive rate as the whole.
    """
    group_array = np.asarray(groups)
    label_array = np.asarray(labels, dtype=bool)
    if group_array.shape != label_array.shape or group_array.ndim != 1:
        raise ValueError(
            f"groups and labels must be two flat sequences of one length, got shapes "
            f"{group_array.shape} and {label_array.shape}"
        )
    if len(group_array) == 0:
        raise ValueError("there are no pairs to weigh")
    weights = np.empty(len(label_array))
    for group in np.unique(group_array):
        in_group = group_array == group
        for label in (False, True):
            cell = in_group & (label_array == label)
            if cell.any():
                expected = in_group.mean() * (label_array == label).mean()
                weights[cell] = expected / cell.mean()
    return weights


class WeightedPipeline(Pipeline):
    """A pipeline whose ``fit`` takes ``sample_weight`` and gives it to the last step.

    ``CalibratedClassifierCV`` passes weights on only to an estimator whose
    ``fit`` names ``sample_weight``, which a plain ``Pipeline`` does not; it
    would warn and weight the calibration alone.
    """

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None, **params: Any) -> Any:
        if sample_weight is not None:
            params[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)
