"""A deterministic stand-in for the cross-encoder, shared by the evaluation tests.

Each (query, document) pair's score is seeded by the pair's hash, so a test can
say exactly what the re-rank should have done; every call is recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def fake_relevance(query: str, document: str) -> float:
    digest = hashlib.sha256(f"{query}\0{document}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


class FakeCrossEncoder:
    """Each pair's score is seeded by its hash; every call is recorded."""

    def __init__(self, name: str = "fake-cross-encoder@1") -> None:
        self.name = name
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        documents = list(documents)
        self.calls.append((query, documents))
        return np.array([fake_relevance(query, d) for d in documents], dtype=np.float64)
