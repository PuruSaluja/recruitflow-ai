"""Feasibility spike: where should resume and job embeddings live?

The planned dense-retrieval upgrade needs one embedding per candidate and per
job, and a cosine search over a job's candidate pool. pgvector would keep them
in a ``vector`` column and search in SQL, but the local PostgreSQL is a
binaries-only Windows build without it. pgvector's own Windows instructions
are a build from source with Visual C++, which this machine lacks, and the
conda-forge package is built against conda-forge's PostgreSQL, so it is
untested here (see ``docs/embedding_storage.md``). The fallback is to store
the embeddings as plain PostgreSQL values and search in numpy.

This script answers both halves against a real database:

1. Whether ``vector`` is among the server's available extensions.
2. What the fallback costs. It loads random unit vectors into a temporary
   table in two forms, ``real[]`` (readable, its length checkable in SQL) and
   ``bytea`` (raw float32 bytes), reads them back three ways, checks every
   read returns the stored float32 values bit for bit, and times the read and
   an exact cosine top-k in numpy. Read forms are timed in a rotating order,
   and medians reported.

Nothing persists: the table is ``TEMPORARY`` and goes with the connection.
The vectors are random, not real embeddings; storage and search cost do not
depend on what the vectors mean.

    python scripts/spike_vector_storage.py --out docs/embedding_storage.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from app.cli import check_writable
from app.config import get_settings

#: all-MiniLM-L6-v2, the usual small sentence-transformer, embeds in 384 dims.
DEFAULT_DIM = 384
#: The default corpus has 4,000 candidates; the larger size shows how exact
#: search grows past it.
DEFAULT_SIZES = (4_000, 20_000)
DEFAULT_TOP_K = 10

TABLE = "spike_embeddings"
READ_FORMS = ("real_array_text", "real_array_binary", "bytea")


class SpikeFailed(Exception):
    """A read did not return what was stored; no timing is reported."""


class SpikeRefused(Exception):
    """The arguments name nothing the spike can run against."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spike_vector_storage",
        description="Check for pgvector and measure the numpy fallback for embedding search.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="database URL (default: the app's configured database_url)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="numbers of stored vectors to measure (default: %(default)s)",
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="embedding dimension")
    parser.add_argument("--rounds", type=int, default=5, help="timed rounds per size")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="also write the results as JSON")
    return parser


def _libpq_url(url: str) -> str:
    """SQLAlchemy's ``postgresql+psycopg://`` URL, as one psycopg itself accepts.

    The messages never repeat the URL, which may carry a password.
    """
    try:
        parsed = make_url(url)
    except ArgumentError:
        raise SpikeRefused("cannot parse --url as a database URL") from None
    backend = parsed.get_backend_name()
    if backend != "postgresql":
        raise SpikeRefused(f"--url must name a PostgreSQL database, not {backend!r}")
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


def vector_extension(conn: psycopg.Connection) -> dict[str, Any]:
    row = conn.execute(
        "select default_version, installed_version from pg_available_extensions"
        " where name = 'vector'"
    ).fetchone()
    return {
        "available": row is not None,
        "default_version": row[0] if row else None,
        "installed_version": row[1] if row else None,
    }


def unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def load(conn: psycopg.Connection, vectors: np.ndarray) -> None:
    dim = vectors.shape[1]
    conn.execute(sql.SQL("drop table if exists {}").format(sql.Identifier(TABLE)))
    conn.execute(
        sql.SQL(
            "create temporary table {table} ("
            " id integer primary key,"
            " as_array real[] not null"
            "  check (array_ndims(as_array) = 1 and cardinality(as_array) = {dim}),"
            " as_bytes bytea not null check (octet_length(as_bytes) = {nbytes}))"
        ).format(
            table=sql.Identifier(TABLE),
            dim=sql.Literal(dim),
            nbytes=sql.Literal(dim * vectors.itemsize),
        )
    )
    with (
        conn.cursor() as cur,
        cur.copy(
            sql.SQL("copy {} (id, as_array, as_bytes) from stdin").format(sql.Identifier(TABLE))
        ) as copy,
    ):
        for i, vector in enumerate(vectors):
            copy.write_row((i, vector.tolist(), vector.tobytes()))
    conn.execute(sql.SQL("analyze {}").format(sql.Identifier(TABLE)))


def _select(column: str) -> sql.Composed:
    return sql.SQL("select {} from {} order by id").format(
        sql.Identifier(column), sql.Identifier(TABLE)
    )


def read_array_text(conn: psycopg.Connection, dim: int) -> np.ndarray:
    with conn.cursor() as cur:
        rows = cur.execute(_select("as_array")).fetchall()
    return np.array([row[0] for row in rows], dtype=np.float32).reshape(-1, dim)


def read_array_binary(conn: psycopg.Connection, dim: int) -> np.ndarray:
    with conn.cursor(binary=True) as cur:
        rows = cur.execute(_select("as_array")).fetchall()
    return np.array([row[0] for row in rows], dtype=np.float32).reshape(-1, dim)


def read_bytea(conn: psycopg.Connection, dim: int) -> np.ndarray:
    with conn.cursor() as cur:
        rows = cur.execute(_select("as_bytes")).fetchall()
    return np.frombuffer(b"".join(row[0] for row in rows), dtype=np.float32).reshape(-1, dim)


READERS: dict[str, Callable[[psycopg.Connection, int], np.ndarray]] = {
    "real_array_text": read_array_text,
    "real_array_binary": read_array_binary,
    "bytea": read_bytea,
}


def top_k_cosine(stored: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` stored vectors most cosine-similar to ``query``, best first.

    Normalises both sides rather than trusting the stored vectors to be unit
    length, so the timing covers what a caller that cannot assume it pays.
    """
    norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(query)
    scores = (stored @ query) / norms
    k = min(k, len(scores))
    best = np.argpartition(-scores, k - 1)[:k]
    return best[np.argsort(-scores[best], kind="stable")]


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def measure_size(
    conn: psycopg.Connection, n: int, dim: int, *, rounds: int, top_k: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vectors = unit_vectors(rng, n, dim)
    query = unit_vectors(rng, 1, dim)[0]

    started = time.perf_counter()
    load(conn, vectors)
    load_seconds = time.perf_counter() - started

    stored_bytes = conn.execute(
        sql.SQL(
            "select sum(pg_column_size(as_array)), sum(pg_column_size(as_bytes)) from {}"
        ).format(sql.Identifier(TABLE))
    ).fetchone()

    reads: dict[str, list[float]] = {form: [] for form in READ_FORMS}
    searches: list[float] = []
    for round_index in range(rounds):
        shift = round_index % len(READ_FORMS)
        for form in READ_FORMS[shift:] + READ_FORMS[:shift]:
            started = time.perf_counter()
            got = READERS[form](conn, dim)
            reads[form].append(time.perf_counter() - started)
            if got.shape != vectors.shape or not np.array_equal(got, vectors):
                raise SpikeFailed(f"{form} read of {n} vectors differs from what was stored")
        # Every read matched ``vectors`` exactly, so searching ``got`` is
        # searching what was stored.
        started = time.perf_counter()
        top_k_cosine(got, query, top_k)
        searches.append(time.perf_counter() - started)

    return {
        "vectors": n,
        "load_ms": _ms(load_seconds),
        "stored_bytes": {"real_array": int(stored_bytes[0]), "bytea": int(stored_bytes[1])},
        "read_ms_median": {form: _ms(statistics.median(reads[form])) for form in READ_FORMS},
        "search_ms_median": _ms(statistics.median(searches)),
    }


def run(
    url: str, sizes: list[int], dim: int, *, rounds: int, top_k: int, seed: int
) -> dict[str, Any]:
    with psycopg.connect(_libpq_url(url), autocommit=True, connect_timeout=10) as conn:
        server = conn.execute("show server_version").fetchone()[0]
        extension = vector_extension(conn)
        measured = [
            measure_size(conn, n, dim, rounds=rounds, top_k=top_k, seed=seed) for n in sizes
        ]
    return {
        "server_version": server,
        "vector_extension": extension,
        "dim": dim,
        "rounds": rounds,
        "top_k": top_k,
        "seed": seed,
        "sizes": measured,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "psycopg": psycopg.__version__,
        "platform": platform.platform(terse=True),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name, value in (("--dim", args.dim), ("--rounds", args.rounds), ("--top-k", args.top_k)):
        if value < 1:
            print(f"error: {name} must be at least 1, got {value}", file=sys.stderr)
            return 2
    if min(args.sizes) < 1:
        print(f"error: --sizes must all be at least 1, got {args.sizes}", file=sys.stderr)
        return 2

    if args.out is not None:
        # The timings take a while; an output it cannot write should not wait for them.
        try:
            check_writable(args.out)
        except OSError as error:
            print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
            return 2

    url = args.url or get_settings().database_url
    try:
        results = run(
            url, args.sizes, args.dim, rounds=args.rounds, top_k=args.top_k, seed=args.seed
        )
    except SpikeRefused as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except psycopg.OperationalError as error:
        print(f"error: cannot reach the database: {error}", file=sys.stderr)
        return 2
    except psycopg.Error as error:
        print(f"error: the database refused the spike: {error}", file=sys.stderr)
        return 2
    except SpikeFailed as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        except OSError as error:
            print(f"error: cannot write {args.out}: {error}", file=sys.stderr)
            return 2
        print(f"Wrote {args.out}")

    extension = results["vector_extension"]
    if extension["available"]:
        status = f"available ({extension['default_version']})"
    else:
        status = "not available"
    print(f"PostgreSQL {results['server_version']}: pgvector {status}")
    print("  median ms per read of every vector, and per top-k search:")
    print(f"  {'vectors':>8} {'text[]':>9} {'binary[]':>9} {'bytea':>9} {'search':>8}")
    for size in results["sizes"]:
        reads = size["read_ms_median"]
        print(
            f"  {size['vectors']:>8} {reads['real_array_text']:>9} "
            f"{reads['real_array_binary']:>9} {reads['bytea']:>9} {size['search_ms_median']:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
