"""Time each stage of the two-stage ranking, one job's slate at a time.

Takes the configuration the evaluation measured (``docs/model_evaluation.json``:
its corpus, the first stage and N it chose, and the cross-encoder it ran) and
ranks every job's slate the way a request would, timing each step on its own:

* ``bm25`` -- the job's document searched against the whole candidate pool
  (the index is built once, beforehand, as a service would keep it);
* ``dense`` -- the job's document encoded, and its cosine with each slate
  candidate's vector (candidate vectors are encoded beforehand, as they are
  stored by :mod:`app.services.embeddings`);
* ``fuse_and_cut`` -- reciprocal rank fusion of the two and the top-N cut;
* ``cross_encoder_kept`` -- the cross-encoder on the kept candidates, the
  second stage;
* ``cross_encoder_slate`` -- the cross-encoder on the whole slate, for
  comparison: what re-ranking without a first stage would cost.

Only the steps the chosen first stage needs are timed. Each repeat goes over
every slate once, after one untimed warm-up slate; the JSON holds each
repeat's median, 90th percentile and mean in milliseconds, so the spread from
one run to the next is visible. Timings depend on the machine and on what
else it is doing; the JSON records the platform and torch's thread count.

    python scripts/time_rerank.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from app.config import REPO_ROOT
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.cross_encoder import CrossEncoderScorer, SentenceTransformerCrossEncoder
from app.ml.embeddings import cosine_scores
from app.ml.encoder import (
    Encoder,
    EncoderUnavailable,
    SentenceTransformerEncoder,
    candidate_document,
    job_document,
)
from app.ml.retrieval import BM25Index, first_stage_cut, reciprocal_rank_fusion
from app.ml.train import training_data

DEFAULT_EVALUATION = REPO_ROOT / "docs" / "model_evaluation.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "rerank_latency.json"

STAGES = ("bm25", "dense", "fuse_and_cut", "cross_encoder_kept", "cross_encoder_slate")


class TimingRefused(RuntimeError):
    """The evaluation JSON does not describe a re-rank to time."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="time_rerank",
        description="Time each stage of the two-stage ranking, per slate.",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=DEFAULT_EVALUATION,
        help="the evaluation JSON whose re-rank to time (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="JSON to write (default: %(default)s)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="passes over every slate (default: %(default)s)"
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="time only the first N jobs' slates (default: all)"
    )
    return parser


def load_encoder() -> Encoder:
    """The pinned sentence-transformer. Raises ``EncoderUnavailable``."""
    return SentenceTransformerEncoder()


def load_cross_encoder() -> CrossEncoderScorer:
    """The pinned cross-encoder. Raises ``EncoderUnavailable``."""
    return SentenceTransformerCrossEncoder()


def _rerank_config(evaluation: dict[str, Any], encoder: Encoder | None) -> dict[str, Any]:
    """The re-rank the evaluation measured, checked against the models loaded here."""
    rerank = evaluation.get("rerank")
    if not isinstance(rerank, dict):
        raise TimingRefused(
            "the evaluation has no re-rank to time; run: python scripts/evaluate_matcher.py"
        )
    if rerank.get("first_stage") not in ("bm25", "dense", "hybrid_rrf"):
        raise TimingRefused(f"unknown first stage {rerank.get('first_stage')!r}")
    if not isinstance(evaluation.get("corpus_config"), dict):
        raise TimingRefused("the evaluation records no corpus_config")
    measured_encoder = evaluation.get("retrieval", {}).get("encoder")
    if encoder is not None and measured_encoder != encoder.name:
        raise TimingRefused(
            f"the evaluation ran encoder {measured_encoder!r}, but {encoder.name!r} is loaded"
        )
    return rerank


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values) * 1000
    return {
        "median_ms": round(float(np.median(array)), 3),
        "p90_ms": round(float(np.percentile(array, 90)), 3),
        "mean_ms": round(float(array.mean()), 3),
    }


def _timed(clock: Callable[[], float], seconds: list[float], call: Callable[[], Any]) -> Any:
    started = clock()
    result = call()
    seconds.append(clock() - started)
    return result


def run(
    evaluation: dict[str, Any],
    *,
    repeats: int,
    jobs: int | None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Everything the JSON holds. Raises ``TimingRefused`` or ``EncoderUnavailable``."""
    first_stage = (evaluation.get("rerank") or {}).get("first_stage")
    encoder = load_encoder() if first_stage in ("dense", "hybrid_rrf") else None
    rerank = _rerank_config(evaluation, encoder)
    cross_encoder = load_cross_encoder()
    if cross_encoder.name != rerank.get("cross_encoder"):
        raise TimingRefused(
            f"the evaluation ran cross-encoder {rerank.get('cross_encoder')!r}, "
            f"but {cross_encoder.name!r} is loaded"
        )
    n = rerank["n"]
    corpus = generate_corpus(CorpusConfig(**evaluation["corpus_config"]))
    data = training_data(corpus)
    refs = np.array(data.job_refs)
    job_refs = sorted(set(data.job_refs))[:jobs]
    by_ref = {job.ref: job for job in corpus.jobs}
    pool_row = {candidate.ref: i for i, candidate in enumerate(corpus.candidates)}

    # Built once, as a service keeps them: the pool's index and stored vectors.
    index = BM25Index([candidate_document(c) for c in corpus.candidates])
    slates = []
    for ref in job_refs:
        candidates = [data.pairs[i][1] for i in np.flatnonzero(refs == ref)]
        documents = [candidate_document(c) for c in candidates]
        slates.append(
            {
                "job": job_document(by_ref[ref]),
                "rows": [pool_row[c.ref] for c in candidates],
                "documents": documents,
                "vectors": None if encoder is None else encoder.encode(documents),
            }
        )

    def rank(slate: dict[str, Any], times: dict[str, list[float]]) -> int:
        lexical = dense = None
        if first_stage in ("bm25", "hybrid_rrf"):
            lexical = _timed(
                clock, times["bm25"], lambda: index.scores(slate["job"])[slate["rows"]]
            )
        if first_stage in ("dense", "hybrid_rrf"):
            dense = _timed(
                clock,
                times["dense"],
                lambda: cosine_scores(slate["vectors"], encoder.encode([slate["job"]])[0]),
            )

        def cut() -> np.ndarray:
            scores = {"bm25": lexical, "dense": dense}.get(first_stage)
            if scores is None:
                scores = reciprocal_rank_fusion(lexical, dense)
            return first_stage_cut(scores, n)

        kept = _timed(clock, times["fuse_and_cut"], cut)
        documents = [d for d, keep in zip(slate["documents"], kept, strict=True) if keep]
        _timed(
            clock, times["cross_encoder_kept"], lambda: cross_encoder.score(slate["job"], documents)
        )
        _timed(
            clock,
            times["cross_encoder_slate"],
            lambda: cross_encoder.score(slate["job"], slate["documents"]),
        )
        return len(documents)

    stages = [
        stage
        for stage in STAGES
        if not (stage == "bm25" and first_stage == "dense")
        and not (stage == "dense" and first_stage == "bm25")
    ]
    rank(slates[0], {stage: [] for stage in STAGES})  # warm-up, untimed
    passes, kept_sizes = [], []
    for _ in range(repeats):
        times: dict[str, list[float]] = {stage: [] for stage in STAGES}
        kept_sizes = [rank(slate, times) for slate in slates]
        passes.append({stage: _stats(times[stage]) for stage in stages})

    return {
        "evaluation_corpus_config": evaluation["corpus_config"],
        "first_stage": first_stage,
        "n": n,
        "encoder": None if encoder is None else encoder.name,
        "cross_encoder": cross_encoder.name,
        "slates": len(slates),
        "slate_size_mean": float(np.mean([len(s["documents"]) for s in slates])),
        "kept_mean": float(np.mean(kept_sizes)),
        "pool": len(corpus.candidates),
        "repeats": passes,
        "median_ms_across_repeats": {
            stage: {
                "min": min(p[stage]["median_ms"] for p in passes),
                "max": max(p[stage]["median_ms"] for p in passes),
            }
            for stage in stages
        },
        "torch_threads": _torch_threads(),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "processor": platform.processor() or None,
    }


def _torch_threads() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    return int(torch.get_num_threads())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        print(f"error: --repeats must be at least 1, got {args.repeats}", file=sys.stderr)
        return 2
    if args.jobs is not None and args.jobs < 1:
        print(f"error: --jobs must be at least 1, got {args.jobs}", file=sys.stderr)
        return 2
    try:
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: cannot read {args.evaluation}: {error}", file=sys.stderr)
        return 2
    try:
        results = run(evaluation, repeats=args.repeats, jobs=args.jobs)
    except (TimingRefused, EncoderUnavailable) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    except OSError as error:
        print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {args.out}")
    print(
        f"  {results['slates']} slates of {results['slate_size_mean']:.0f}, first stage "
        f"{results['first_stage']} keeping {results['kept_mean']:.1f} on average "
        f"(N = {results['n']})"
    )
    for stage, spread in results["median_ms_across_repeats"].items():
        print(f"    {stage:<20} median {spread['min']:.2f} to {spread['max']:.2f} ms per slate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
