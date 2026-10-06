"""Embedding storage format and exact cosine search.

The 2026-10-01 spike (``docs/embedding_storage.md``) found pgvector is not
available on the local server (adding it needs a Visual Studio build or an
untested conda-forge package), and chose to keep embeddings in PostgreSQL as
``bytea`` holding float32 values, searched exactly in numpy over a job's pool.
This module is that format and that search, with no database and no encoder
in it, so both can be tested on their own.

**Byte order is part of the format.** Vectors are written as little-endian
float32 (``<f4``) whatever the machine, so the stored bytes mean the same
everywhere; the database checks the length (``4 * dimensions``) but cannot
check the order, so the code has to keep it.

**What can be stored.** A one-dimensional, non-empty vector of finite values
that is not all zeros. NaN or infinity would make every cosine against the
vector NaN, and a zero vector has no direction, so its cosine is undefined;
both are refused when written rather than turning up later as a bad ranking.
A float64 value too large for float32 becomes infinity on conversion, so the
check runs on the converted values.

**Decoded vectors are copies.** ``np.frombuffer`` returns a read-only view of
the fetched bytes (the spike's own test tripped on that), so decoding copies
into a writable native-order array that a caller may normalise in place.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Little-endian float32, the stored element type.
STORED_DTYPE = np.dtype("<f4")
BYTES_PER_VALUE = STORED_DTYPE.itemsize


def _checked(vector: np.ndarray) -> np.ndarray:
    """``vector`` as ``<f4``, refused if it could not be searched."""
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"an embedding is a non-empty 1-D vector, got shape {vector.shape}")
    with np.errstate(over="ignore"):  # an overflow is refused just below
        stored = vector.astype(STORED_DTYPE)
    if not np.isfinite(stored).all():
        raise ValueError("an embedding must hold finite float32 values")
    if not stored.any():
        raise ValueError("an all-zero embedding has no direction to compare")
    return stored


def to_bytes(vector: np.ndarray | Iterable[float]) -> bytes:
    """The stored form of ``vector``: its values as little-endian float32."""
    return _checked(np.asarray(vector)).tobytes()


def from_bytes(data: bytes, dimensions: int) -> np.ndarray:
    """A writable float32 copy of a vector stored by :func:`to_bytes`."""
    if dimensions <= 0:
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    if len(data) != BYTES_PER_VALUE * dimensions:
        raise ValueError(
            f"{len(data)} bytes is not {dimensions} float32 values "
            f"({BYTES_PER_VALUE * dimensions} bytes)"
        )
    return np.frombuffer(data, dtype=STORED_DTYPE).astype(np.float32)


def stack(blobs: Iterable[bytes], dimensions: int) -> np.ndarray:
    """Stored vectors as one writable ``(n, dimensions)`` float32 matrix, in order.

    Every blob is length-checked, so one of the wrong size is refused rather
    than shifting every vector after it.
    """
    rows = [from_bytes(blob, dimensions) for blob in blobs]
    if not rows:
        return np.empty((0, dimensions), dtype=np.float32)
    return np.vstack(rows)


def cosine_scores(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of every row of ``stored`` with ``query``, in row order.

    Both sides are normalised here rather than trusted to be unit length. A
    zero-length row or query is refused, as :func:`to_bytes` refuses to store
    one, and so is NaN or infinity on either side: either makes a similarity
    NaN, which no ranking can place. That is checked on the similarities, one
    per row, rather than on every value of the matrix.
    """
    if stored.ndim != 2 or query.ndim != 1 or stored.shape[1] != query.shape[0]:
        raise ValueError(
            f"cannot compare a query of shape {query.shape} with vectors of shape {stored.shape}"
        )
    row_norms = np.linalg.norm(stored, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or (row_norms == 0).any():
        raise ValueError("cosine similarity is undefined for a zero vector")
    with np.errstate(invalid="ignore"):  # inf / inf is refused just below
        similarities = (stored @ query) / (row_norms * query_norm)
    if not np.isfinite(similarities).all():
        raise ValueError("cosine similarity needs finite vectors; NaN cannot be ranked")
    return similarities


def top_k_cosine(stored: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``stored`` most cosine-similar to ``query``: (indices, similarities).

    Best first; equal similarities keep the order of ``stored``, so the
    result is the same on every run. ``k`` larger than the pool returns the
    whole pool. The similarities are :func:`cosine_scores`, with its refusals.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    similarities = cosine_scores(stored, query)
    # A full stable sort rather than argpartition, so ties keep their order.
    # Measured for a top-10, best of 5, in two sessions: 0.18-0.21 ms over
    # 4,000 scores and 1.18-1.34 ms over 20,000, against 0.02 and 0.04-0.05 ms
    # for argpartition.
    order = np.argsort(-similarities, kind="stable")[:k]
    return order, similarities[order]
