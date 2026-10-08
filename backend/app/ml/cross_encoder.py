"""Cross-encoder that scores a candidate against a job, for re-ranking a shortlist.

The second stage of two-stage retrieval (:func:`app.ml.retrieval.rerank_top_n`).
Where the dense encoder embeds the job and the candidate apart and compares the
vectors, a cross-encoder reads the two documents together in one pass and
gives a single relevance score, so it is far slower per pair and is only run
on the candidates a first stage keeps. The default model is
ms-marco-MiniLM-L6-v2, trained on MS MARCO passage ranking (web search queries
against passages), **pinned to one revision**; the scorer's name is the model
id, that revision and the window it reads.

The documents are the ones the encoder reads (:func:`app.ml.encoder.job_document`,
:func:`app.ml.encoder.candidate_document`), the job's as the query. Scores are
the model's raw logits, not passed through a sigmoid: only their order is used,
and a sigmoid would squash the strongest matches towards equal scores.

**Long resumes are read as passages.** The model was trained to score short
passages, and unrelated text around a match drowns it even inside the 512
word-pieces the model can read: measured, a three-line Python developer query
scores a one-line Python resume 4.01, the same line after 60 lines of
warehouse work -10.86 when read as one window, and the warehouse lines alone
-11.16. So a candidate's document is cut at word boundaries
(:func:`app.ml.encoder.word_piece_chunks`) into passages of at most
:data:`DEFAULT_PASSAGE_TOKENS` word-pieces, each is scored beside the job,
and the candidate's score is the best passage's ("MaxP", as in Dai and
Callan, 2019): the best passage of a resume is what shows a match, and an
average would mark down a strong resume for a long unrelated section. Read
that way, the long resume above scores 0.14 and the warehouse lines alone
-11.36. The
job's document is kept whole when it fits in half the window (254 of the 512
word-pieces, three of which are the model's markers) and otherwise cut to its
opening at the start of a word. Every pair in the default synthetic corpus is
read whole (the longest job document is 118 word-pieces, the longest
candidate's 128), so its scores are exactly the model's for the whole texts.

Needs the same optional ``embeddings`` extra as the encoder;
:class:`app.ml.encoder.EncoderUnavailable` names the install when it is
missing, and when the model's files can neither be found nor downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.ml.encoder import INSTALL_HINT, EncoderUnavailable, word_piece_chunks

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
#: The model repository's commit on the Hugging Face Hub, as of 2026-10-06.
DEFAULT_CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
#: Pairs per forward pass.
DEFAULT_BATCH_SIZE = 32
#: The longest passage of a candidate's document scored at once, in
#: word-pieces, about a hundred words of resume.
DEFAULT_PASSAGE_TOKENS = 128


class CrossEncoderScorer(Protocol):
    """What a re-rank needs from a cross-encoder."""

    #: Identifies the weights, revision included.
    name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """One relevance score per document for ``query``, in order; higher is better."""
        ...


class SentenceTransformerCrossEncoder:
    """A pinned cross-encoder model, run on the CPU unless told otherwise."""

    def __init__(
        self,
        model: str = DEFAULT_CROSS_ENCODER,
        revision: str = DEFAULT_CROSS_ENCODER_REVISION,
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        passage_tokens: int = DEFAULT_PASSAGE_TOKENS,
    ) -> None:
        if not revision:
            raise ValueError("a revision is required: the scorer's name has to pin its weights")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if passage_tokens < 1:
            raise ValueError(f"passage_tokens must be at least 1, got {passage_tokens}")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise EncoderUnavailable(
                f"sentence-transformers is not installed; {INSTALL_HINT}"
            ) from exc
        try:
            self._model = CrossEncoder(model, revision=revision, device=device)
        # As for the encoder: files neither cached nor downloadable raise
        # OSError, a revision the repository does not have ValueError.
        except (OSError, ValueError) as exc:
            raise EncoderUnavailable(f"cannot load {model}@{revision}: {exc}") from exc
        if self._model.num_labels != 1:
            raise EncoderUnavailable(
                f"{model}@{revision} gives {self._model.num_labels} scores per pair, not one"
            )
        self._tokenizer = self._model.tokenizer
        self._identity = torch.nn.Identity()
        #: Word-pieces of the pair the model reads, its own markers included.
        self.window = int(self._model.max_seq_length)
        self.markers = int(self._tokenizer.num_special_tokens_to_add(pair=True))
        #: The most of the job's document kept with every passage of a candidate's.
        self.query_tokens = (self.window - self.markers) // 2
        #: The longest passage of a candidate's document, as given or as fits.
        self.passage_tokens = min(passage_tokens, self.window - self.markers - self.query_tokens)
        self.name = f"{model}@{revision}+maxp{self.passage_tokens}"
        self.batch_size = batch_size

    def read_query(self, query: str) -> tuple[str, int]:
        """The part of ``query`` the model reads, and its word-piece count."""
        return word_piece_chunks(self._tokenizer, query, self.query_tokens)[0]

    def passages(self, document: str) -> list[tuple[str, int]]:
        """``document`` as the passages scored, each with its word-piece count."""
        return word_piece_chunks(self._tokenizer, document, self.passage_tokens)

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        documents = list(documents)
        if not documents:
            return np.empty(0, dtype=np.float64)
        read = self.read_query(query)[0]
        split = [self.passages(document) for document in documents]
        pairs = [(read, piece) for parts in split for piece, _ in parts]
        logits = np.asarray(
            self._model.predict(
                pairs,
                batch_size=self.batch_size,
                activation_fn=self._identity,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )
        out = np.empty(len(documents), dtype=np.float64)
        row = 0
        for index, parts in enumerate(split):
            out[index] = logits[row : row + len(parts)].max()
            row += len(parts)
        return out
