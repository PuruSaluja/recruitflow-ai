"""Audit the matcher's shortlists for group fairness on the synthetic corpus.

Fairness is measured on shortlists -- each slate's top k, for k in
:data:`CUTOFFS` -- with :mod:`app.ml.fairness`: selection rate, disparate
impact ratio and the four-fifths rule, and equal-opportunity difference among
qualified pairs, by gender (against men) and by age band (against under 40).

**Every job is scored out of fold.** The jobs are dealt into ``--folds``
folds; for each fold a matcher is trained on the others and scores it. So all
200 jobs of the default corpus are audited, each by a matcher that never saw
its labels, rather than the 50 one held-out split leaves -- an audit of the
training procedure, which is what the persisted matcher is one run of. With
50 jobs the intervals on women's ratio spanned 0.79 to 1.25.

What is audited:

* ``historical_decisions`` -- the ``advanced`` label itself, the past
  recruiter's decisions the matcher learns from. Not a top k: a decision.
* ``matcher`` -- the matcher as trained, on ``advanced``.
* ``fitted_on_relevance`` -- the same pipeline fitted on ``relevance >= 1``.
  A reference: no real system has ground-truth labels.
* ``trained_without_historical_bias`` -- the matcher trained on the same
  corpus generated with ``historical_bias=0``: the same people, slates and
  ground truth, only the past decisions unbiased. Also a reference; it shows
  what the matcher does when the record it learns from is fair.

Each comes with its NDCG@10 against ground truth, so any fairness gain can be
read beside what it costs in ranking quality.

``proxy`` measures the planted text proxy directly. Regenerating the corpus
with ``proxy_share=0`` gives the same people, slates and labels, with each
women-in-tech line replaced by the neutral line the same person would
otherwise have written (the script checks the pairs and labels match). The
same out-of-fold matchers score both, so the difference in a pair's score is
what that line alone does to it.

Writes JSON (default ``docs/fairness_audit.json``):

    python scripts/audit_fairness.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.evaluate import NDCG_K, bootstrap_interval, evaluate_scores
from app.ml.fairness import (
    FOUR_FIFTHS,
    QUALIFIED_GRADE,
    FairnessAudit,
    audit_selection,
    top_k_selection,
)
from app.ml.train import TRAINING_LABEL, TrainingData, make_matcher, training_data

DEFAULT_OUT = REPO_ROOT / "docs" / "fairness_audit.json"

#: Shortlist lengths audited.
CUTOFFS = (5, 10)

#: Each audited attribute, and the group the others are compared with.
REFERENCES = {"gender": "man", "age_band": "under_40"}

#: Fits a model to (pairs, labels) and returns it; ``predict_proba`` must work on it.
Fitter = Callable[[TrainingData], Any]


class AuditRefused(RuntimeError):
    """The audit cannot be run as asked."""


def build_parser() -> argparse.ArgumentParser:
    corpus_defaults = CorpusConfig()
    parser = argparse.ArgumentParser(
        prog="audit_fairness",
        description="Audit the matcher's shortlists for group fairness on the synthetic corpus.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="JSON to write (default: %(default)s)"
    )
    parser.add_argument("--seed", type=int, default=corpus_defaults.seed, help="corpus seed")
    parser.add_argument("--jobs", type=int, default=corpus_defaults.n_jobs, dest="n_jobs")
    parser.add_argument(
        "--candidates", type=int, default=corpus_defaults.n_candidates, dest="n_candidates"
    )
    parser.add_argument(
        "--folds", type=int, default=4, help="job folds to score out of (default: %(default)s)"
    )
    parser.add_argument("--fold-seed", type=int, default=0, help="seed for dealing jobs into folds")
    parser.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap resamples (default: %(default)s)"
    )
    return parser


def job_folds(job_refs: Sequence[str], folds: int, *, seed: int) -> dict[str, int]:
    """Deal the distinct jobs into ``folds`` folds of near-equal size.

    Jobs are dealt from a sorted, shuffled list, so the same seed gives the
    same folds whatever order the pairs arrive in.
    """
    refs = sorted(set(job_refs))
    if not 2 <= folds <= len(refs):
        raise AuditRefused(f"--folds must be between 2 and the number of jobs ({len(refs)})")
    random.Random(seed).shuffle(refs)
    return {ref: index % folds for index, ref in enumerate(refs)}


def out_of_fold_scores(
    data: TrainingData,
    fold_of: dict[str, int],
    fit: Fitter,
    *,
    also_score: Sequence[TrainingData] = (),
) -> list[np.ndarray]:
    """Score every pair of ``data`` with a model trained on the other folds.

    ``also_score`` are other versions of the same pairs, in the same order,
    scored by the very same models; the result holds ``data``'s scores first,
    then each of theirs.
    """
    folds = np.array([fold_of[ref] for ref in data.job_refs])
    results = [np.full(len(data), np.nan) for _ in range(1 + len(also_score))]
    for fold in sorted(set(fold_of.values())):
        held = folds == fold
        model = fit(_subset(data, ~held))
        for result, version in zip(results, [data, *also_score], strict=True):
            rows = [pair for pair, keep in zip(version.pairs, held, strict=True) if keep]
            result[held] = model.predict_proba(rows)[:, 1]
    return results


def _subset(data: TrainingData, mask: np.ndarray) -> TrainingData:
    return TrainingData(
        pairs=tuple(pair for pair, keep in zip(data.pairs, mask, strict=True) if keep),
        advanced=data.advanced[mask],
        relevance=data.relevance[mask],
        job_refs=tuple(ref for ref, keep in zip(data.job_refs, mask, strict=True) if keep),
    )


def fit_on_advanced(data: TrainingData) -> Any:
    return make_matcher().fit(data.pairs, getattr(data, TRAINING_LABEL))


def fit_on_relevance(data: TrainingData) -> Any:
    return make_matcher().fit(data.pairs, data.relevance >= QUALIFIED_GRADE)


def groups_of(data: TrainingData, attribute: str) -> list[str]:
    return [str(getattr(candidate, attribute)) for _, candidate in data.pairs]


def audit_dict(audit: FairnessAudit) -> dict[str, Any]:
    return {
        "reference": audit.reference,
        "min_disparate_impact": audit.min_disparate_impact,
        "groups": {name: dataclasses.asdict(group) for name, group in audit.groups.items()},
    }


def audit_all(
    data: TrainingData, selected: np.ndarray, *, resamples: int
) -> dict[str, dict[str, Any]]:
    return {
        attribute: audit_dict(
            audit_selection(
                data.job_refs,
                selected,
                groups_of(data, attribute),
                data.relevance,
                reference=reference,
                resamples=resamples,
            )
        )
        for attribute, reference in REFERENCES.items()
    }


def scorer_report(data: TrainingData, scores: np.ndarray, *, resamples: int) -> dict[str, Any]:
    """Ranking quality and the audit at every cut-off, for one set of out-of-fold scores."""
    evaluation = evaluate_scores(data, scores)
    ndcgs = [slate.ndcg for slate in evaluation.slates if slate.ndcg is not None]
    return {
        f"ndcg_at_{NDCG_K}": evaluation.ndcg_at_10,
        f"ndcg_at_{NDCG_K}_95ci": bootstrap_interval(ndcgs, resamples=resamples),
        "at_k": {
            str(k): audit_all(data, top_k_selection(data.job_refs, scores, k), resamples=resamples)
            for k in CUTOFFS
        },
    }


def _same_pairs(first: TrainingData, second: TrainingData, *, compare_advanced: bool) -> bool:
    """Whether both hold the same jobs and candidates in one order, with one ground truth."""
    return (
        [(job.ref, candidate.ref) for job, candidate in first.pairs]
        == [(job.ref, candidate.ref) for job, candidate in second.pairs]
        and np.array_equal(first.relevance, second.relevance)
        and (not compare_advanced or np.array_equal(first.advanced, second.advanced))
    )


def proxy_report(
    data: TrainingData,
    with_line: np.ndarray,
    without_line: np.ndarray,
    swapped: np.ndarray,
    *,
    resamples: int,
) -> dict[str, Any]:
    """What the planted line does to the pairs that carry it, scored both ways."""
    change = with_line - without_line
    report: dict[str, Any] = {
        "pairs_with_line": int(swapped.sum()),
        "candidates_with_line": len(
            {c.ref for (_, c), s in zip(data.pairs, swapped, strict=True) if s}
        ),
        "mean_score_change": float(change[swapped].mean()),
        "median_score_change": float(np.median(change[swapped])),
        "share_of_pairs_scored_lower": float((change[swapped] < 0).mean()),
        "max_abs_change_elsewhere": float(np.abs(change[~swapped]).max()),
        "at_k": {},
    }
    for k in CUTOFFS:
        selected_with = top_k_selection(data.job_refs, with_line, k)
        selected_without = top_k_selection(data.job_refs, without_line, k)
        women = np.array(groups_of(data, "gender")) == "woman"
        report["at_k"][str(k)] = {
            "selection_rate_with_line": float(selected_with[swapped].mean()),
            "selection_rate_without_line": float(selected_without[swapped].mean()),
            "women_selection_rate_with_line": float(selected_with[women].mean()),
            "women_selection_rate_without_line": float(selected_without[women].mean()),
            "gender_without_line": audit_all(data, selected_without, resamples=resamples)["gender"],
        }
    return report


def audit(config: CorpusConfig, *, folds: int, fold_seed: int, resamples: int) -> dict[str, Any]:
    """Everything the JSON holds."""
    corpus = generate_corpus(config)
    data = training_data(corpus)
    fold_of = job_folds(data.job_refs, folds, seed=fold_seed)

    without_proxy = training_data(generate_corpus(dataclasses.replace(config, proxy_share=0.0)))
    unbiased = training_data(generate_corpus(dataclasses.replace(config, historical_bias=0.0)))
    # Both comparisons rest on the generator keeping everything else fixed.
    if not _same_pairs(data, without_proxy, compare_advanced=True):
        raise AuditRefused("the corpus with proxy_share=0 does not hold the same labelled pairs")
    if not _same_pairs(data, unbiased, compare_advanced=False):
        raise AuditRefused("the corpus with historical_bias=0 does not hold the same pairs")

    swapped = np.array(
        [
            a.resume_text != b.resume_text
            for (_, a), (_, b) in zip(data.pairs, without_proxy.pairs, strict=True)
        ]
    )
    matcher, matcher_no_line = out_of_fold_scores(
        data, fold_of, fit_on_advanced, also_score=[without_proxy]
    )
    (relevance_scores,) = out_of_fold_scores(data, fold_of, fit_on_relevance)
    # Trained on the unbiased record, scored on the same pairs as everything else.
    (unbiased_scores,) = out_of_fold_scores(unbiased, fold_of, fit_on_advanced)

    return {
        "synthetic": True,
        "generator": "scripts/audit_fairness.py",
        "corpus_config": dataclasses.asdict(config),
        "training_label": TRAINING_LABEL,
        "folds": folds,
        "fold_seed": fold_seed,
        "resamples": resamples,
        "cutoffs": list(CUTOFFS),
        "four_fifths": FOUR_FIFTHS,
        "qualified_grade": QUALIFIED_GRADE,
        "references": REFERENCES,
        "jobs": len(fold_of),
        "pairs": len(data),
        "group_pairs": {
            attribute: dict(sorted(_count(groups_of(data, attribute)).items()))
            for attribute in REFERENCES
        },
        "historical_decisions": audit_all(data, data.advanced.astype(float), resamples=resamples),
        "scorers": {
            "matcher": scorer_report(data, matcher, resamples=resamples),
            "fitted_on_relevance": scorer_report(data, relevance_scores, resamples=resamples),
            "trained_without_historical_bias": scorer_report(
                data, unbiased_scores, resamples=resamples
            ),
        },
        "proxy": proxy_report(data, matcher, matcher_no_line, swapped, resamples=resamples),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }


def _count(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resamples < 1:
        print(f"error: --resamples must be at least 1, got {args.resamples}", file=sys.stderr)
        return 2
    # The audit takes minutes; a folder that cannot be made should not wait for it.
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
        return 2
    try:
        config = CorpusConfig(seed=args.seed, n_jobs=args.n_jobs, n_candidates=args.n_candidates)
        results = audit(
            config, folds=args.folds, fold_seed=args.fold_seed, resamples=args.resamples
        )
    except (AuditRefused, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        args.out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as error:
        print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {args.out}")
    print(f"  {results['pairs']} pairs across {results['jobs']} jobs, {args.folds} folds")
    record = results["historical_decisions"]["gender"]["groups"]["woman"]
    print(f"  historical decisions, women/men ratio {_fmt(record['disparate_impact']['value'])}")
    for name, report in results["scorers"].items():
        cells = []
        for k in CUTOFFS:
            woman = report["at_k"][str(k)]["gender"]["groups"]["woman"]
            cells.append(
                f"@{k} DI {_fmt(woman['disparate_impact']['value'])} "
                f"EOD {_fmt(woman['equal_opportunity_difference']['value'], 4)}"
            )
        print(f"  {name:<32} NDCG@10 {report['ndcg_at_10']:.4f}  women: {'; '.join(cells)}")
    proxy = results["proxy"]
    print(
        f"  proxy: {proxy['pairs_with_line']} pairs, mean score change "
        f"{proxy['mean_score_change']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
