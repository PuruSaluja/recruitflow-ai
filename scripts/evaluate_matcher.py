"""Evaluate the persisted matcher on its own held-out jobs, beside baselines.

The matcher's manifest records the corpus config and the split it was trained
with, so this script regenerates that corpus in memory, holds out the same
jobs, and scores them with the matcher on disk. Before measuring anything it
recomputes the held-out ROC-AUC the training run wrote into the manifest; if
the two differ, the matcher is not the one the manifest describes (or the
corpus generator has changed since), and the script refuses rather than
report numbers for the wrong split.

Every scorer is measured on the same held-out pairs with :mod:`app.ml.evaluate`:

* ``matcher`` -- the persisted, calibrated matcher;
* ``random`` -- every score equal, which the metrics turn into the exact
  expected value of a random ranking;
* ``required_skill_coverage`` -- one engineered feature on its own, untrained,
  to show what the rest of the model adds;
* ``fitted_on_relevance`` -- the same pipeline fitted on ``relevance >= 1``
  instead of ``advanced``. **A reference, not a candidate**: no real system has
  ground-truth labels. It shows how much ranking quality learning from the past
  recruiter's biased decisions costs on this corpus.

NDCG@10 and P@5 carry a 95% bootstrap interval over held-out jobs. Then the
whole training is repeated on ``--split-seeds`` other splits of the same
corpus, so the spread from which jobs happen to be held out is visible too.

Writes JSON (default ``docs/model_evaluation.json``), which
``docs/model_evaluation.md`` quotes:

    python scripts/evaluate_matcher.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from app.config import REPO_ROOT
from app.ml.artifact import (
    MatcherUnavailable,
    default_matcher_path,
    load_manifest,
    load_versioned_matcher,
)
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.evaluate import (
    NDCG_K,
    PRECISION_K,
    RELEVANT_GRADE,
    Evaluation,
    bootstrap_interval,
    evaluate_scores,
)
from app.ml.features import feature_dict
from app.ml.train import TRAINING_LABEL, TrainingData, make_matcher, split_by_job, training_data

DEFAULT_OUT = REPO_ROOT / "docs" / "model_evaluation.json"

#: Other splits of the same corpus to retrain and re-measure on.
DEFAULT_SPLIT_SEEDS = tuple(range(1, 10))

#: How far a recomputed ROC-AUC may sit from the manifest's before the matcher
#: is taken not to match it. Both come from the same float arithmetic, so the
#: real difference is zero; this only absorbs summation order.
MANIFEST_TOLERANCE = 1e-9


class EvaluationRefused(RuntimeError):
    """The persisted matcher cannot be evaluated as its manifest describes."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_matcher",
        description="Measure the persisted matcher on its held-out jobs, beside baselines.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_matcher_path(),
        help="the matcher to evaluate; its manifest must sit beside it (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="JSON to write (default: %(default)s)"
    )
    parser.add_argument(
        "--split-seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SPLIT_SEEDS),
        help="other split seeds to retrain and re-measure on (default: 1 to 9; none to skip)",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap resamples (default: %(default)s)"
    )
    return parser


def _baseline_coverage(data: TrainingData) -> np.ndarray:
    return np.array(
        [feature_dict(job, candidate)["required_skill_coverage"] for job, candidate in data.pairs]
    )


def _summary(evaluation: Evaluation, *, resamples: int) -> dict[str, Any]:
    ndcgs = [slate.ndcg for slate in evaluation.slates if slate.ndcg is not None]
    precisions = [slate.precision for slate in evaluation.slates]
    return {
        f"ndcg_at_{NDCG_K}": evaluation.ndcg_at_10,
        f"ndcg_at_{NDCG_K}_95ci": bootstrap_interval(ndcgs, resamples=resamples),
        f"precision_at_{PRECISION_K}": evaluation.precision_at_5,
        f"precision_at_{PRECISION_K}_95ci": bootstrap_interval(precisions, resamples=resamples),
        "roc_auc_advanced": evaluation.roc_auc_advanced,
        "roc_auc_relevant": evaluation.roc_auc_relevant,
        "roc_auc_strong": evaluation.roc_auc_strong,
    }


def _spread(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _manifest_fields(manifest: dict[str, Any]) -> tuple[CorpusConfig, float, int, float]:
    """The corpus config, test share, split seed and recorded ROC-AUC, each checked.

    A manifest edited by hand or written by other code can lack any of them or
    hold the wrong type; each case is refused by name rather than surfacing as
    a ``KeyError`` or ``TypeError`` from deep inside the run.
    """

    def refuse(problem: str) -> EvaluationRefused:
        return EvaluationRefused(
            f"the manifest {problem}. Retrain the matcher with: python scripts/train_matcher.py"
        )

    if manifest.get("training_label") != TRAINING_LABEL:
        raise EvaluationRefused(
            f"the manifest says the matcher was fitted on {manifest.get('training_label')!r}, "
            f"but this code trains on {TRAINING_LABEL!r}; retrain it"
        )
    corpus_config = manifest.get("corpus_config")
    if not isinstance(corpus_config, dict):
        raise refuse("has no corpus_config object")
    try:
        config = CorpusConfig(**corpus_config)
    except (TypeError, ValueError) as error:
        raise refuse(f"holds a corpus_config this code cannot use ({error})") from error
    test_share, split_seed = manifest.get("test_share"), manifest.get("split_seed")
    if not _is_number(test_share):
        raise refuse(f"has no numeric test_share (found {test_share!r})")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise refuse(f"has no integer split_seed (found {split_seed!r})")
    metrics = manifest.get("metrics")
    recorded = metrics.get("roc_auc_advanced") if isinstance(metrics, dict) else None
    if not _is_number(recorded):
        raise refuse(
            f"records no held-out ROC-AUC vs {TRAINING_LABEL} to check the matcher against "
            f"(found {recorded!r})"
        )
    return config, float(test_share), split_seed, float(recorded)


def evaluate(model: Path, *, split_seeds: Sequence[int], resamples: int) -> dict[str, Any]:
    """Everything the JSON holds. Raises :class:`EvaluationRefused` or ``MatcherUnavailable``."""
    matcher, version = load_versioned_matcher(model)
    manifest = load_manifest(model)
    config, test_share, split_seed, recorded = _manifest_fields(manifest)
    # A repeated seed, or the matcher's own, would count one split twice in the spread.
    if len(set(split_seeds)) != len(split_seeds) or split_seed in split_seeds:
        raise EvaluationRefused(
            f"--split-seeds must be distinct and must not repeat the matcher's own split "
            f"({split_seed}); got {list(split_seeds)}"
        )

    try:
        data = training_data(generate_corpus(config))
        train_set, test_set = split_by_job(data, test_share=test_share, seed=split_seed)
    except ValueError as error:
        raise EvaluationRefused(
            f"the manifest's corpus or split cannot be built: {error}"
        ) from error

    scores = matcher.predict_proba(test_set.pairs)[:, 1]
    measured = evaluate_scores(test_set, scores)
    if not math.isclose(
        measured.roc_auc_advanced or math.nan, recorded, rel_tol=0.0, abs_tol=MANIFEST_TOLERANCE
    ):
        raise EvaluationRefused(
            f"held-out ROC-AUC vs {TRAINING_LABEL} recomputes to {measured.roc_auc_advanced} "
            f"but the manifest recorded {recorded}: the matcher at {model} was not trained on "
            "the split its manifest describes, or the corpus generator has changed since. "
            "Retrain it with: python scripts/train_matcher.py"
        )

    reference = make_matcher().fit(train_set.pairs, train_set.relevance >= RELEVANT_GRADE)
    scorers = {
        "matcher": measured,
        "random": evaluate_scores(test_set, np.zeros(len(test_set))),
        "required_skill_coverage": evaluate_scores(test_set, _baseline_coverage(test_set)),
        "fitted_on_relevance": evaluate_scores(
            test_set, reference.predict_proba(test_set.pairs)[:, 1]
        ),
    }

    per_split = []
    for seed in split_seeds:
        other_train, other_test = split_by_job(data, test_share=test_share, seed=seed)
        fitted = make_matcher().fit(other_train.pairs, getattr(other_train, TRAINING_LABEL))
        result = evaluate_scores(other_test, fitted.predict_proba(other_test.pairs)[:, 1])
        per_split.append(
            {
                "split_seed": seed,
                f"ndcg_at_{NDCG_K}": result.ndcg_at_10,
                f"precision_at_{PRECISION_K}": result.precision_at_5,
                "roc_auc_advanced": result.roc_auc_advanced,
                "roc_auc_relevant": result.roc_auc_relevant,
            }
        )

    every_split = [
        {
            "split_seed": split_seed,
            f"ndcg_at_{NDCG_K}": measured.ndcg_at_10,
            f"precision_at_{PRECISION_K}": measured.precision_at_5,
            "roc_auc_advanced": measured.roc_auc_advanced,
            "roc_auc_relevant": measured.roc_auc_relevant,
        },
        *per_split,
    ]
    metric_names = [
        f"ndcg_at_{NDCG_K}",
        f"precision_at_{PRECISION_K}",
        "roc_auc_advanced",
        "roc_auc_relevant",
    ]
    return {
        "synthetic": True,
        "generator": "scripts/evaluate_matcher.py",
        "matcher_version": version,
        "corpus_config": manifest["corpus_config"],
        "test_share": test_share,
        "split_seed": split_seed,
        "training_label": TRAINING_LABEL,
        "held_out": {
            "pairs": len(test_set),
            "jobs": len(set(test_set.job_refs)),
            "jobs_without_relevant": measured.jobs_without_relevant,
            "relevant_share": float((test_set.relevance >= RELEVANT_GRADE).mean()),
        },
        "bootstrap_resamples": resamples,
        "scorers": {
            name: _summary(result, resamples=resamples) for name, result in scorers.items()
        },
        "splits": every_split,
        "split_spread": {
            name: _spread([split[name] for split in every_split]) for name in metric_names
        },
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resamples < 1:
        print(f"error: --resamples must be at least 1, got {args.resamples}", file=sys.stderr)
        return 2
    try:
        results = evaluate(args.model, split_seeds=args.split_seeds, resamples=args.resamples)
    except (MatcherUnavailable, EvaluationRefused) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as error:
        print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
        return 2

    held_out = results["held_out"]
    print(f"Wrote {args.out}")
    print(
        f"  matcher {results['matcher_version']}, held out: {held_out['pairs']} pairs across "
        f"{held_out['jobs']} jobs"
    )
    print(f"  {'scorer':<26} {'NDCG@10':>8} {'P@5':>8} {'AUC adv':>8} {'AUC rel':>8}")
    for name, summary in results["scorers"].items():
        print(
            f"  {name:<26} {_fmt(summary['ndcg_at_10']):>8} {_fmt(summary['precision_at_5']):>8} "
            f"{_fmt(summary['roc_auc_advanced']):>8} {_fmt(summary['roc_auc_relevant']):>8}"
        )
    spread = results["split_spread"]
    print(f"  over {len(results['splits'])} splits (mean, std):")
    for name, values in spread.items():
        print(f"    {name:<24} {values['mean']:.4f} +/- {values['std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
