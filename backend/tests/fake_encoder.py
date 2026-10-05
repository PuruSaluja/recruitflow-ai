"""A deterministic stand-in for the sentence-transformer, shared by the embedding tests.

Each text's vector is seeded by the text's hash, so a test can say exactly
which vector should be stored for it; every call is recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np


def fake_vector(document: str, dimensions: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(document.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(dimensions).astype(np.float32)


class FakeEncoder:
    """Each text's vector is seeded by its hash; every call is recorded."""

    def __init__(
        self,
        name: str = "fake-encoder@1",
        dimensions: int = 8,
        on_encode: Callable[[int], None] | None = None,
    ) -> None:
        self.name = name
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.on_encode = on_encode

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        self.calls.append(texts)
        if self.on_encode is not None:
            self.on_encode(len(self.calls))
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([fake_vector(t, self.dimensions) for t in texts])
