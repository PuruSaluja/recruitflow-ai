"""Bring the stored embeddings up to date for every candidate and job.

Encodes each candidate and job whose vector is missing, or was made from text
that has changed since, with the pinned sentence-transformer
(:mod:`app.ml.encoder`), and stores the vectors through
:mod:`app.services.embeddings`. Anything already current is left alone, so a
second run with nothing edited encodes nothing.

    python scripts/embed.py
    python scripts/embed.py --kind candidates --batch-size 128 --out report.json

Needs the ``embeddings`` extra (``sentence-transformers``, and with it torch)
and the model files, downloaded on first use into the Hugging Face cache.
Prints what it did per kind, with the time spent in the model separately from
the total, and with ``--out`` writes the same as JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.encoder import Encoder, EncoderUnavailable, SentenceTransformerEncoder
from app.services.embeddings import CANDIDATES, DEFAULT_BATCH_SIZE, JOBS, refresh_embeddings

KINDS = {"candidates": CANDIDATES, "jobs": JOBS}


class TimedEncoder:
    """Passes texts to ``encoder`` and adds up the seconds it takes."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.name = encoder.name
        self.dimensions = encoder.dimensions
        self.seconds = 0.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        start = time.perf_counter()
        try:
            return self.encoder.encode(texts)
        finally:
            self.seconds += time.perf_counter() - start


def make_encoder(device: str) -> Encoder:
    """The encoder a run uses; a separate function so tests can replace it."""
    return SentenceTransformerEncoder(device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embed",
        description="Encode every candidate and job whose stored embedding is missing or stale.",
    )
    parser.add_argument(
        "--url", default=None, help="database URL (default: the app's configured database_url)"
    )
    parser.add_argument(
        "--kind",
        choices=[*KINDS, "all"],
        default="all",
        help="what to embed (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="owners encoded and committed per batch (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    return parser


def _postgres_url(url: str) -> str:
    """The URL, refused unless it names a PostgreSQL database. Never echoes it."""
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise ValueError("cannot parse --url as a database URL") from None
    if parsed.get_backend_name() != "postgresql":
        raise ValueError(
            f"--url must name a PostgreSQL database, not {parsed.get_backend_name()!r}"
        )
    return url


def run(url: str, kinds: Sequence[str], batch_size: int, device: str) -> dict[str, Any]:
    engine = create_engine(url)
    try:
        # Reach the database before spending seconds loading the model.
        with engine.connect():
            pass
        started = time.perf_counter()
        encoder = TimedEncoder(make_encoder(device))
        load_seconds = time.perf_counter() - started
    except BaseException:
        engine.dispose()
        raise

    report: dict[str, Any] = {
        "encoder": encoder.name,
        "dimensions": encoder.dimensions,
        "batch_size": batch_size,
        "device": device,
        "model_load_seconds": round(load_seconds, 2),
        "kinds": {},
    }
    try:
        for name in kinds:
            before = encoder.seconds
            start = time.perf_counter()
            with Session(engine) as db:
                result = refresh_embeddings(db, KINDS[name], encoder, batch_size=batch_size)
            report["kinds"][name] = {
                "encoded": result.encoded,
                "current": result.current,
                "vanished": result.vanished,
                "seconds": round(time.perf_counter() - start, 2),
                "encode_seconds": round(encoder.seconds - before, 2),
            }
    finally:
        engine.dispose()
    report["python"] = platform.python_version()
    report["platform"] = platform.platform(terse=True)
    for package in ("torch", "sentence_transformers"):
        module = sys.modules.get(package)
        report[package] = getattr(module, "__version__", None)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print(f"error: --batch-size must be at least 1, got {args.batch_size}", file=sys.stderr)
        return 2
    try:
        url = _postgres_url(args.url or get_settings().database_url)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    kinds = list(KINDS) if args.kind == "all" else [args.kind]

    try:
        report = run(url, kinds, args.batch_size, args.device)
    except EncoderUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OperationalError as error:
        print(f"error: cannot reach the database: {error.orig}", file=sys.stderr)
        return 2
    except ProgrammingError as error:
        print(
            f"error: the database refused the refresh ({error.orig}); "
            "is it migrated? run: alembic upgrade head",
            file=sys.stderr,
        )
        return 2

    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        except OSError as error:
            print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
            return 2
        print(f"Wrote {args.out}")

    loaded = report["model_load_seconds"]
    print(f"{report['encoder']} ({report['dimensions']} dims), loaded in {loaded} s")
    for name, kind in report["kinds"].items():
        print(
            f"  {name}: {kind['encoded']} encoded, {kind['current']} already current, "
            f"{kind['vanished']} deleted meanwhile; {kind['seconds']} s "
            f"({kind['encode_seconds']} s in the model)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
