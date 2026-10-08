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

**Long documents are encoded in chunks.** The model reads at most 256
word-pieces, two of them its own markers, and would silently ignore the rest,
so a real resume longer than that would be embedded from its opening only. A
document that fits is encoded as it is. A longer one is cut at word
boundaries into pieces of at most 254 word-pieces, each piece is encoded, and
the document's vector is the mean of theirs weighted by their word-piece
counts, rescaled to unit length like the model's own outputs. The method is
part of the encoder's name (``+chunks254``), so vectors made before it, which
truncated, show up as another encoder's. Every document in the default
synthetic corpus fits in one piece (the longest is 130 word-pieces), so its
vectors are exactly what the model gives for the whole text.

``sentence-transformers`` (and with it torch) is an optional dependency, the
``embeddings`` extra: nothing else in the app needs it, and the CPU build of
torch alone is over 500 MB. :class:`EncoderUnavailable` names the install
when it is missing, and when the model's files can neither be found in the
local cache nor downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from app.ml.text import CandidateText, JobText, candidate_text, fold_text, job_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: The model repository's commit on the Hugging Face Hub, as of 2026-10-04.
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
#: Texts per forward pass. Which batch a text is encoded in moves its vector by
#: up to 1.6e-7 per value (measured over 64 corpus documents, alone against in
#: batches of 64 and 512), far below anything a ranking could notice.
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


def word_piece_chunks(tokenizer: Any, text: str, limit: int) -> list[tuple[str, int]]:
    """``text`` cut into pieces of at most ``limit`` word-pieces, each with its count.

    One piece, the text itself, when it fits. Otherwise each piece is cut from
    the original text at the start of a word, so that it tokenizes back to
    exactly the word-pieces counted for it; only a single word longer than a
    whole piece is cut inside the word. ``tokenizer`` is a Hugging Face fast
    tokenizer, or anything answering the same call.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    offsets = encoding["offset_mapping"]
    if len(offsets) <= limit:
        return [(text, len(offsets))]
    words = encoding.word_ids()
    pieces = []
    start = 0
    while start < len(offsets):
        end = min(start + limit, len(offsets))
        if end < len(offsets) and words[end] == words[end - 1]:
            first = end
            while first > start and words[first - 1] == words[end]:
                first -= 1
            if first > start:
                end = first
        pieces.append((text[offsets[start][0] : offsets[end - 1][1]], end - start))
        start = end
    return pieces


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
        dimensions = self._model.get_embedding_dimension()
        if dimensions is None:  # the method is typed int | None
            raise EncoderUnavailable(f"{model}@{revision} does not report its embedding size")
        self._tokenizer = self._model.tokenizer
        #: Word-pieces of a document per piece the model reads in full.
        self.chunk_tokens = int(self._model.max_seq_length) - int(
            self._tokenizer.num_special_tokens_to_add()
        )
        self.name = f"{model}@{revision}+chunks{self.chunk_tokens}"
        self.dimensions = int(dimensions)
        self.batch_size = batch_size

    def chunks(self, text: str) -> list[tuple[str, int]]:
        """``text`` as the pieces it is encoded in, each with its word-piece count."""
        return word_piece_chunks(self._tokenizer, text, self.chunk_tokens)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        split = [self.chunks(text) for text in texts]
        pieces = [piece for parts in split for piece, _ in parts]
        vectors = np.asarray(
            self._model.encode(
                pieces, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )
        out = np.empty((len(texts), self.dimensions), dtype=np.float32)
        row = 0
        for index, parts in enumerate(split):
            if len(parts) == 1:
                out[index] = vectors[row]
            else:
                weights = np.array([count for _, count in parts], dtype=np.float64)
                mean = weights @ vectors[row : row + len(parts)].astype(np.float64)
                out[index] = mean / np.linalg.norm(mean)
            row += len(parts)
        return out
