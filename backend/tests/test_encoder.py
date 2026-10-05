"""Tests for app.ml.encoder.

The tests that run the real model are skipped where ``sentence-transformers``
is not installed or its pinned model cannot be loaded, as in CI's main job,
whose dev install leaves the ``embeddings`` extra out. CI's ``embeddings`` job
installs it and sets ``RECRUITFLOW_REQUIRE_ENCODER``, which turns that skip
into a failure, so a model that failed to download cannot pass as a skip.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.encoder import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    EncoderUnavailable,
    SentenceTransformerEncoder,
    candidate_document,
    job_document,
)

# --- documents -------------------------------------------------------------


def test_documents_are_the_text_features_documents_folded() -> None:
    candidate = SimpleNamespace(
        current_title="\uff24ata Analyst",
        resume_text="Certi\ufb01ed in Py\u200bthon.",
        skills=["SQL", "Python"],
    )
    job = SimpleNamespace(
        title="Data Analyst",
        description="\uff37eekly metrics.",
        required_skills=["SQL"],
        preferred_skills=["Tableau", "dbt"],
    )
    assert candidate_document(candidate) == "data analyst\ncertified in python.\nsql python"
    assert job_document(job) == "data analyst\nweekly metrics.\nsql\ntableau dbt"


def test_a_candidate_without_a_title_still_gets_a_document() -> None:
    candidate = SimpleNamespace(current_title=None, resume_text="Resume.", skills=[])
    assert candidate_document(candidate) == "\nresume.\n"


# --- loading ---------------------------------------------------------------


def test_a_missing_package_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises
    with pytest.raises(EncoderUnavailable, match=r"pip install -e \"\.\[embeddings\]\""):
        SentenceTransformerEncoder()


@pytest.mark.parametrize("error", [OSError("not cached"), ValueError("unrecognized model")])
def test_a_model_that_cannot_load_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=refuse)
    )
    with pytest.raises(EncoderUnavailable, match=f"cannot load {DEFAULT_MODEL}@{DEFAULT_REVISION}"):
        SentenceTransformerEncoder()


def test_another_failure_while_loading_is_not_disguised(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=broken)
    )
    with pytest.raises(RuntimeError, match="bug") as caught:
        SentenceTransformerEncoder()
    # EncoderUnavailable is a RuntimeError too, so check it was not wrapped.
    assert type(caught.value) is RuntimeError


def test_the_model_is_asked_for_by_its_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Recording:
        max_seq_length = 128
        tokenizer = SimpleNamespace(num_special_tokens_to_add=lambda: 2)

        def __init__(self, model: str, **kwargs: object) -> None:
            calls.append((model, kwargs))

        def get_embedding_dimension(self) -> int:
            return 7

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Recording)
    )
    encoder = SentenceTransformerEncoder("org/model", "abc123", device="cuda", batch_size=5)
    assert calls == [("org/model", {"revision": "abc123", "device": "cuda"})]
    # The name pins the weights and the chunking: 128 - 2 markers per piece.
    assert encoder.name == "org/model@abc123+chunks126"
    assert encoder.chunk_tokens == 126
    assert encoder.dimensions == 7
    assert encoder.batch_size == 5
    assert SentenceTransformerEncoder().name == f"{DEFAULT_MODEL}@{DEFAULT_REVISION}+chunks126"
    assert calls[-1] == (DEFAULT_MODEL, {"revision": DEFAULT_REVISION, "device": "cpu"})


def test_a_model_that_reports_no_embedding_size_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sizeless:
        def __init__(self, model: str, **kwargs: object) -> None:
            pass

        def get_embedding_dimension(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Sizeless)
    )
    with pytest.raises(EncoderUnavailable, match="does not report its embedding size"):
        SentenceTransformerEncoder()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"revision": ""}, "revision is required"), ({"batch_size": 0}, "at least 1")],
)
def test_bad_arguments_are_refused_before_loading(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SentenceTransformerEncoder(**kwargs)


def test_the_default_revision_is_a_full_commit_hash() -> None:
    assert len(DEFAULT_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_REVISION)


# --- the real model --------------------------------------------------------


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformerEncoder:
    try:
        return SentenceTransformerEncoder()
    except EncoderUnavailable as exc:  # pragma: no cover - depends on the machine
        if os.environ.get("RECRUITFLOW_REQUIRE_ENCODER"):
            raise
        pytest.skip(str(exc))


def test_real_model_shape_type_and_name(encoder: SentenceTransformerEncoder) -> None:
    assert encoder.name == f"{DEFAULT_MODEL}@{DEFAULT_REVISION}+chunks254"
    assert encoder.chunk_tokens == 254
    assert encoder.dimensions == 384
    vectors = encoder.encode(["python developer", "registered nurse", "data analyst"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    # The model ends in a Normalize layer.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_real_model_nothing_in_nothing_out(encoder: SentenceTransformerEncoder) -> None:
    out = encoder.encode([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32


def test_real_model_keeps_order_and_repeats_exactly(encoder: SentenceTransformerEncoder) -> None:
    texts = ["python developer", "registered nurse", "python developer"]
    vectors = encoder.encode(texts)
    assert np.array_equal(vectors, encoder.encode(texts))
    assert np.allclose(vectors[0], vectors[2], atol=1e-6)
    assert not np.allclose(vectors[0], vectors[1], atol=1e-2)


def test_real_model_batch_composition_barely_moves_a_vector(
    encoder: SentenceTransformerEncoder,
) -> None:
    alone = encoder.encode(["built reporting pipelines in sql and python"])[0]
    batch = encoder.encode(
        ["built reporting pipelines in sql and python", "x " * 120, "registered nurse, icu"]
    )[0]
    assert np.abs(alone - batch).max() < 1e-6


def test_real_model_ranks_related_text_above_unrelated(
    encoder: SentenceTransformerEncoder,
) -> None:
    job, close, far = encoder.encode(
        [
            "senior data engineer: python, sql, airflow pipelines",
            "built etl pipelines in python and sql, scheduled with airflow",
            "registered nurse with icu and emergency room experience",
        ]
    )
    assert float(job @ close) > float(job @ far) + 0.2


# --- chunking (real model) -------------------------------------------------

DATA = "Senior data engineer. Built ETL pipelines in Python and SQL, scheduled with Airflow. "
NURSE = "Registered nurse with ICU and emergency room experience, caring for critical patients. "


def _pieces(encoder: SentenceTransformerEncoder, text: str) -> int:
    return len(encoder._tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def test_a_text_that_fits_is_one_piece_and_the_model_s_own_vector(
    encoder: SentenceTransformerEncoder,
) -> None:
    text = DATA * 3
    assert encoder.chunks(text) == [(text, _pieces(encoder, text))]
    # Bit for bit, over many documents: rescaling a unit vector again moves
    # about half of them by up to 1.5e-8, which one text can miss.
    corpus = generate_corpus(CorpusConfig(n_jobs=10, n_candidates=200))
    docs = [candidate_document(candidate) for candidate in corpus.candidates[:40]]
    plain = encoder._model.encode(
        docs, batch_size=encoder.batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    assert np.array_equal(encoder.encode(docs), plain)


def test_a_long_text_is_cut_into_pieces_that_cover_it_exactly(
    encoder: SentenceTransformerEncoder,
) -> None:
    text = DATA * 20 + NURSE * 20
    pieces = encoder.chunks(text)
    assert len(pieces) > 1
    assert sum(count for _, count in pieces) == _pieces(encoder, text)
    for piece, count in pieces:
        assert count <= encoder.chunk_tokens
        assert _pieces(encoder, piece) == count  # cut where it tokenizes back the same


def test_pieces_are_cut_at_the_start_of_a_word(encoder: SentenceTransformerEncoder) -> None:
    # "terraforming" is three word-pieces, which do not divide a piece, so a
    # cut at the piece size would land inside the word, starting the next
    # piece with a continuation such as "##form".
    word_pieces = _pieces(encoder, "terraforming")
    assert word_pieces == 3 and encoder.chunk_tokens % word_pieces != 0
    text = "terraforming " * 400
    pieces = encoder.chunks(text)
    assert len(pieces) > 1
    assert all(piece.startswith("terraforming") for piece, _ in pieces)
    assert all(piece.endswith("terraforming") for piece, _ in pieces)


def test_the_end_of_a_long_text_now_reaches_its_vector(encoder: SentenceTransformerEncoder) -> None:
    long_text = DATA * 20 + NURSE * 20
    nurse_query = encoder.encode(["registered nurse, icu experience"])[0]
    truncated = encoder._model.encode([long_text], convert_to_numpy=True, show_progress_bar=False)[
        0
    ]
    chunked = encoder.encode([long_text])[0]
    assert np.linalg.norm(chunked) == pytest.approx(1.0, abs=1e-6)
    # Truncated, the model reads only the data-engineering half.
    assert float(nurse_query @ chunked) > float(nurse_query @ truncated) + 0.1


def test_the_vector_is_the_word_piece_weighted_mean_of_its_pieces(
    encoder: SentenceTransformerEncoder,
) -> None:
    text = DATA * 20 + NURSE * 10
    pieces = encoder.chunks(text)
    parts = encoder._model.encode(
        [piece for piece, _ in pieces], convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float64)
    weights = np.array([count for _, count in pieces], dtype=np.float64)
    expected = weights @ parts
    expected /= np.linalg.norm(expected)
    assert np.allclose(encoder.encode([text])[0], expected, atol=1e-6)


def test_long_and_short_texts_mixed_keep_their_order(encoder: SentenceTransformerEncoder) -> None:
    short, long_text = DATA, DATA * 20 + NURSE * 20
    mixed = encoder.encode([short, long_text, short])
    assert np.allclose(mixed[0], encoder.encode([short])[0], atol=1e-6)
    assert np.allclose(mixed[1], encoder.encode([long_text])[0], atol=1e-6)
    assert np.allclose(mixed[2], mixed[0], atol=1e-6)


def test_chunking_does_not_log_the_tokenizer_s_length_warning(
    encoder: SentenceTransformerEncoder,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # transformers' loggers do not propagate, and the tokenizer warns once per
    # instance; both are undone here so the warning would be seen if logged.
    monkeypatch.setattr(logging.getLogger("transformers"), "propagate", True)
    monkeypatch.setattr(encoder._tokenizer, "deprecation_warnings", {})
    with caplog.at_level("WARNING"):
        encoder.chunks(DATA * 40)
    assert not [r for r in caplog.records if "sequence length" in r.getMessage()]


# --- chunking logic, with a stand-in tokenizer (runs without the model) -----


class _Encoding(dict):
    def __init__(self, offsets: list[tuple[int, int]], words: list[int]) -> None:
        super().__init__(offset_mapping=offsets)
        self._words = words

    def word_ids(self) -> list[int]:
        return self._words


class _CharTokenizer:
    """One token per non-space character; a word is a run of them."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool, verbose: bool
    ) -> _Encoding:
        assert (add_special_tokens, return_offsets_mapping, verbose) == (False, True, False)
        offsets, words, word, gap = [], [], -1, True
        for index, char in enumerate(text):
            if char.isspace():
                gap = True
                continue
            if gap:
                word, gap = word + 1, False
            offsets.append((index, index + 1))
            words.append(word)
        return _Encoding(offsets, words)


def _chunker(chunk_tokens: int) -> SentenceTransformerEncoder:
    encoder = object.__new__(SentenceTransformerEncoder)
    encoder._tokenizer = _CharTokenizer()
    encoder.chunk_tokens = chunk_tokens
    return encoder


def test_a_text_of_exactly_one_piece_is_returned_as_it_is() -> None:
    assert _chunker(5).chunks(" ab cde ") == [(" ab cde ", 5)]


def test_a_cut_inside_a_word_backs_off_to_the_start_of_that_word() -> None:
    assert _chunker(5).chunks("abc defg hi") == [("abc", 3), ("defg", 4), ("hi", 2)]


def test_a_cut_that_falls_between_words_stays_where_it_is() -> None:
    assert _chunker(5).chunks("abcde fgh") == [("abcde", 5), ("fgh", 3)]


def test_one_word_longer_than_a_piece_is_cut_inside_it_and_covered() -> None:
    assert _chunker(5).chunks("abcdefghijkl") == [("abcde", 5), ("fghij", 5), ("kl", 2)]


def test_a_long_word_after_a_short_one_starts_its_own_piece() -> None:
    assert _chunker(5).chunks("ab cdefghij") == [("ab", 2), ("cdefg", 5), ("hij", 3)]


def test_an_empty_text_is_one_empty_piece() -> None:
    assert _chunker(5).chunks("") == [("", 0)]
