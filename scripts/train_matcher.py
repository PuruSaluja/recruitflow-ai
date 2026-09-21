"""Train the matcher on a synthetic corpus and persist it.

The corpus is generated in memory by ``app.ml.corpus`` -- this script does not
read the JSON Lines files ``data/generate_corpus.py`` writes, since training
only needs the same config, not the files on disk. Output is two files next
to ``--out`` (default: the app's configured ``model_path``):

* ``<name>.joblib`` -- the fitted, calibrated matcher (see ``app.ml.train``).
* ``<name>.manifest.json`` -- the corpus config, the split, the training
  label, measured held-out metrics, and the interpreter that produced it.
  ``random``'s draws and the calibration folds both depend on it, the way the
  corpus manifest already documents for the corpus itself.

Neither file is committed -- ``.gitignore`` excludes the whole
``backend/app/ml/artifacts`` directory -- so this script is how the matcher
is regenerated:

    python scripts/train_matcher.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

from app.ml.artifact import default_matcher_path, save_matcher
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.train import (
    CALIBRATION_METHOD,
    DEFAULT_TEST_SHARE,
    TRAINING_LABEL,
    TrainingData,
    split_by_job,
    train_matcher,
    training_data,
)

#: CalibratedClassifierCV's default is 5-fold (pinned by test_train.py); a
#: minority class smaller than this only draws a warning and calibrates on
#: folds too thin to trust, which is not something to ship silently.
MIN_CLASS_FOR_CALIBRATION = 5


def build_parser() -> argparse.ArgumentParser:
    corpus_defaults = CorpusConfig()
    parser = argparse.ArgumentParser(
        prog="train_matcher",
        description="Train the calibrated matcher on a synthetic corpus and persist it.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_matcher_path(),
        help="path to write the matcher to, alongside its manifest (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=corpus_defaults.seed, help="corpus seed")
    parser.add_argument("--jobs", type=int, default=corpus_defaults.n_jobs, dest="n_jobs")
    parser.add_argument(
        "--candidates", type=int, default=corpus_defaults.n_candidates, dest="n_candidates"
    )
    parser.add_argument("--slate-size", type=int, default=corpus_defaults.slate_size)
    parser.add_argument(
        "--same-family-share", type=float, default=corpus_defaults.same_family_share
    )
    parser.add_argument("--historical-bias", type=float, default=corpus_defaults.historical_bias)
    parser.add_argument("--proxy-share", type=float, default=corpus_defaults.proxy_share)
    parser.add_argument(
        "--test-share",
        type=float,
        default=DEFAULT_TEST_SHARE,
        help="share of jobs held out for evaluation",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0, help="seed for which jobs are held out"
    )
    return parser


def corpus_config_from(args: argparse.Namespace) -> CorpusConfig:
    fields = {field.name for field in dataclasses.fields(CorpusConfig)}
    return CorpusConfig(**{key: value for key, value in vars(args).items() if key in fields})


def _untrainable_reason(train_set: TrainingData) -> str | None:
    """Why ``train_set`` cannot be trained on, or ``None`` if it can.

    scikit-learn's own failure here is either an opaque solver error ("needs
    samples of at least 2 classes") when one label never occurs, or a bare
    warning -- not a refusal -- when it does occur but too rarely for the
    calibration folds, which would silently ship a matcher calibrated on
    almost nothing.
    """
    n_pos = int(train_set.advanced.sum())
    n_neg = len(train_set) - n_pos
    if n_pos == 0 or n_neg == 0:
        missing = "advanced" if n_pos == 0 else "rejected"
        return (
            f"nobody in the training split was ever {missing} ({n_pos} advanced of "
            f"{len(train_set)} training pairs) -- a classifier needs both. Try a larger "
            "corpus or a different --seed/--split-seed."
        )
    minority = min(n_pos, n_neg)
    if minority < MIN_CLASS_FOR_CALIBRATION:
        return (
            f"only {minority} of {len(train_set)} training pairs are in the minority class, "
            f"fewer than the {MIN_CLASS_FOR_CALIBRATION} calibration folds need -- scikit-learn "
            "would only warn about this and calibrate on folds that thin anyway. Use a larger "
            "corpus."
        )
    return None


def _safe_auc(labels, scores) -> float | None:
    """ROC-AUC, or ``None`` when the held-out set has only one class to score against."""
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if math.isnan(args.test_share) or not 0.0 < args.test_share < 1.0:
        print(
            f"error: --test-share must be a number between 0 and 1, got {args.test_share}",
            file=sys.stderr,
        )
        return 2

    try:
        corpus_config = corpus_config_from(args)
        corpus = generate_corpus(corpus_config)
        data = training_data(corpus)
        train_set, test_set = split_by_job(data, test_share=args.test_share, seed=args.split_seed)
    except ValueError as error:
        # The corpus config, the slate pool and the split all raise this for a
        # request that cannot be met; that is the caller's mistake, not a crash.
        print(f"error: {error}", file=sys.stderr)
        return 2

    reason = _untrainable_reason(train_set)
    if reason is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 2

    matcher = train_matcher(train_set)
    scores = matcher.predict_proba(test_set.pairs)[:, 1]

    metrics: dict[str, Any] = {
        "roc_auc_advanced": _safe_auc(test_set.advanced, scores),
        "roc_auc_relevance": _safe_auc(test_set.relevance >= 1, scores),
        "mean_predicted_probability": float(scores.mean()),
        "base_rate": float(train_set.advanced.mean()),
    }
    manifest = {
        "synthetic": True,
        "generator": "app.ml.train.train_matcher",
        "corpus_config": dataclasses.asdict(corpus_config),
        "test_share": args.test_share,
        "split_seed": args.split_seed,
        "training_label": TRAINING_LABEL,
        "calibration_method": CALIBRATION_METHOD,
        "rows": {
            "train_pairs": len(train_set),
            "test_pairs": len(test_set),
            "train_jobs": len(set(train_set.job_refs)),
            "test_jobs": len(set(test_set.job_refs)),
        },
        "metrics": metrics,
    }
    written = save_matcher(matcher, manifest, path=args.out)

    rows = manifest["rows"]
    print(f"Wrote {written}")
    print(f"  trained on: {rows['train_pairs']} pairs, {rows['train_jobs']} jobs")
    print(f"  held out: {rows['test_pairs']} pairs, {rows['test_jobs']} jobs")
    auc = metrics["roc_auc_advanced"]
    print(f"  held-out ROC-AUC vs {TRAINING_LABEL}: {'n/a' if auc is None else f'{auc:.4f}'}")
    print(f"  mean predicted probability: {metrics['mean_predicted_probability']:.4f}")
    print(f"  base rate ({TRAINING_LABEL}, training split): {metrics['base_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
