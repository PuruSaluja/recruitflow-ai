"""Sentence-transformer encoder for resumes and job descriptions.

The dense half of retrieval: one embedding per candidate and per job, from a
pretrained sentence-transformer, stored as :mod:`app.ml.embeddings` describes
and cached by :mod:`app.services.embeddings`. The default model is
all-MiniLM-L6-v2 (384 dimensions, the size the 2026-10-01 storage spike
measured), **pinned to one revision**: the encoder's name is the model id and
that revision, and it is stored beside every vector, so a change of weights
makes the old vectors visibly another encoder's.

What gets encoded is the same document the TF-IDF features read
(:func:`app.ml.text.candidate_text`, :func:`app.ml.text.job_text`), folded
with :func:`app.ml.text.fold_text`. The model's tokenizer lowercases, but it
does not apply NFKC: measured on this model, full-width ``Python`` has cosine
0.21 with ``Python`` as typed, and ``certified`` with an ``fi`` ligature 0.90;
folded, both are 1.0.

The model reads at most 256 word-pieces and ignores the rest. Every document
in the default synthetic corpus fits (the longest is 130), but a real resume
can run past it, and then only its opening reaches the embedding.

``sentence-transformers`` (and with it torch) is an optional dependency, the
``embeddings`` extra: nothing else in the app needs it, and the CPU build of
torch alone is over 500 MB. :class:`EncoderUnavailable` names the install
when it is missing, and when the model's files can neither be found in the
local cache nor downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.ml.text import CandidateText, JobText, candidate_text, fold_text, job_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: The model repository's commit on the Hugging Face Hub, as of 2026-10-04.
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
#: Texts per forward pass. Batch composition moves a vector by at most about
#: 1e-7 per value (measured), so it does not matter which batch a text was in.
DEFAULT_BATCH_SIZE = 64

INSTALL_HINT = (
    'install the embeddings extra: pip install -e ".[embeddings]" '
    "(for the CPU-only torch build, first: pip install torch "
    "--index-url https://download.pytorch.org/whl/cpu)"
)


class EncoderUnavailable(RuntimeError):
    """The encoder cannot be loaded: its package is missing, or its model files."""


class Encoder(Protocol):
    """What the embedding cache needs from an encoder."""

    #: Identifies the weights, revision included; stored beside each vector.
    name: str
    dimensions: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One float32 row of ``dimensions`` values per text, in order."""
        ...


def candidate_document(candidate: CandidateText) -> str:
    """The text encoded for a candidate."""
    return fold_text(candidate_text(candidate))


def job_document(job: JobText) -> str:
    """The text encoded for a job."""
    return fold_text(job_text(job))


class SentenceTransformerEncoder:
    """A pinned sentence-transformer model, run on the CPU unless told otherwise."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not revision:
            raise ValueError("a revision is required: the encoder's name has to pin its weights")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EncoderUnavailable(
                f"sentence-transformers is not installed; {INSTALL_HINT}"
            ) from exc
        try:
            self._model = SentenceTransformer(model, revision=revision, device=device)
        # Files neither cached nor downloadable raise OSError; a revision the
        # repository does not have was measured to raise ValueError instead.
        except (OSError, ValueError) as exc:
            raise EncoderUnavailable(f"cannot load {model}@{revision}: {exc}") from exc
        self.name = f"{model}@{revision}"
        self.dimensions = int(self._model.get_embedding_dimension())
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
