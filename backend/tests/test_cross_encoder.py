"""Tests for app.ml.cross_encoder.

As in test_encoder.py, the tests that run the real model are skipped where
``sentence-transformers`` is not installed or the pinned model cannot be
loaded, unless ``RECRUITFLOW_REQUIRE_ENCODER`` is set, as in CI's
``embeddings`` job. The rest stand in for the model, the tokenizer and torch,
so they run on the main job's install, which has none of them.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.ml.corpus import Corpus, CorpusConfig, generate_corpus
from app.ml.cross_encoder import (
    DEFAULT_CROSS_ENCODER,
    DEFAULT_CROSS_ENCODER_REVISION,
    SentenceTransformerCrossEncoder,
)
from app.ml.encoder import EncoderUnavailable, candidate_document, job_document

# --- a stand-in model -----------------------------------------------------


class _Encoding(dict):
    def __init__(self, offsets: list[tuple[int, int]], words: list[int]) -> None:
        super().__init__(offset_mapping=offsets)
        self._words = words

    def word_ids(self) -> list[int]:
        return self._words


class _CharTokenizer:
    """One token per non-space character; a word is a run of them; three pair markers."""

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

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 2


_IDENTITY = object()


def _stand_in(
    monkeypatch: pytest.MonkeyPatch, *, window: int = 23, num_labels: int = 1
) -> tuple[list, list]:
    """Installs a stand-in CrossEncoder scoring a pair by the piece's count of ``x``."""
    loads, predictions = [], []

    class StandIn:
        max_seq_length = window
        tokenizer = _CharTokenizer()

        def __init__(self, model: str, **kwargs: object) -> None:
            loads.append((model, kwargs))
            self.num_labels = num_labels

        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
            predictions.append((list(pairs), kwargs))
            return np.array([float(piece.count("x")) for _, piece in pairs], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=StandIn))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Identity=lambda: _IDENTITY))
    )
    return loads, predictions


# --- loading ---------------------------------------------------------------


def test_a_missing_package_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises
    with pytest.raises(EncoderUnavailable, match=r"pip install -e \"\.\[embeddings\]\""):
        SentenceTransformerCrossEncoder()


def test_missing_torch_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _stand_in(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(EncoderUnavailable, match=r"pip install -e \"\.\[embeddings\]\""):
        SentenceTransformerCrossEncoder()


@pytest.mark.parametrize("error", [OSError("not cached"), ValueError("unrecognized model")])
def test_a_model_that_cannot_load_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise error

    _stand_in(monkeypatch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=refuse))
    with pytest.raises(
        EncoderUnavailable,
        match=f"cannot load {DEFAULT_CROSS_ENCODER}@{DEFAULT_CROSS_ENCODER_REVISION}",
    ):
        SentenceTransformerCrossEncoder()


def test_another_failure_while_loading_is_not_disguised(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bug")

    _stand_in(monkeypatch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=broken))
    with pytest.raises(RuntimeError, match="bug") as caught:
        SentenceTransformerCrossEncoder()
    assert type(caught.value) is RuntimeError


def test_the_model_is_asked_for_by_its_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    loads, _ = _stand_in(monkeypatch, window=128)
    scorer = SentenceTransformerCrossEncoder("org/model", "abc123", device="cuda", batch_size=5)
    assert loads == [("org/model", {"revision": "abc123", "device": "cuda"})]
    # The name pins the weights and the window each pair is read in.
    assert scorer.name == "org/model@abc123+maxp63"
    # 128 - 3 markers - a query of up to 62 leaves 63 for a passage.
    assert (scorer.window, scorer.markers, scorer.query_tokens) == (128, 3, 62)
    assert scorer.passage_tokens == 63
    assert scorer.batch_size == 5
    assert SentenceTransformerCrossEncoder().name == (
        f"{DEFAULT_CROSS_ENCODER}@{DEFAULT_CROSS_ENCODER_REVISION}+maxp63"
    )
    assert loads[-1] == (
        DEFAULT_CROSS_ENCODER,
        {"revision": DEFAULT_CROSS_ENCODER_REVISION, "device": "cpu"},
    )


def test_a_model_giving_more_than_one_score_per_pair_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stand_in(monkeypatch, num_labels=3)
    with pytest.raises(EncoderUnavailable, match="gives 3 scores per pair, not one"):
        SentenceTransformerCrossEncoder()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revision": ""}, "revision is required"),
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"passage_tokens": 0}, "passage_tokens must be at least 1"),
    ],
)
def test_bad_arguments_are_refused_before_loading(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    loads, _ = _stand_in(monkeypatch)
    with pytest.raises(ValueError, match=message):
        SentenceTransformerCrossEncoder(**kwargs)
    assert loads == []


def test_the_default_revision_is_a_full_commit_hash() -> None:
    assert len(DEFAULT_CROSS_ENCODER_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_CROSS_ENCODER_REVISION)


# --- scoring, with the stand-in --------------------------------------------


def test_nothing_in_nothing_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _, predictions = _stand_in(monkeypatch)
    scores = SentenceTransformerCrossEncoder().score("query", [])
    assert scores.shape == (0,) and scores.dtype == np.float64
    assert predictions == []


def test_one_string_is_refused_rather_than_scored_letter_by_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions = _stand_in(monkeypatch)
    with pytest.raises(TypeError, match="sequence of documents, not one string"):
        SentenceTransformerCrossEncoder().score("query", "xx")
    assert predictions == []


def test_documents_that_fit_are_read_whole_in_order_as_raw_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions = _stand_in(monkeypatch)
    scorer = SentenceTransformerCrossEncoder(batch_size=7)
    scores = scorer.score("job", ["x", "xx yy", "", "xxx"])
    assert scores.tolist() == [1.0, 2.0, 0.0, 3.0]
    assert scores.dtype == np.float64
    [(pairs, kwargs)] = predictions
    assert pairs == [("job", "x"), ("job", "xx yy"), ("job", ""), ("job", "xxx")]
    # The logits as they are: no sigmoid, which would squash the top towards a tie.
    assert kwargs["activation_fn"] is _IDENTITY
    assert kwargs["batch_size"] == 7


def test_a_long_query_is_cut_to_its_opening_at_a_word(monkeypatch: pytest.MonkeyPatch) -> None:
    # Window 23, 3 markers: the query keeps at most 10 word-pieces.
    _, predictions = _stand_in(monkeypatch)
    scorer = SentenceTransformerCrossEncoder()
    assert scorer.query_tokens == 10
    assert scorer.read_query("abcd efgh ijkl") == ("abcd efgh", 8)
    assert scorer.read_query("abcd efgh ij") == ("abcd efgh ij", 10)
    scorer.score("abcd efgh ijkl", ["x"])
    assert predictions[0][0] == [("abcd efgh", "x")]


def test_a_long_document_is_cut_into_passages_at_words(monkeypatch: pytest.MonkeyPatch) -> None:
    _, predictions = _stand_in(monkeypatch)
    scorer = SentenceTransformerCrossEncoder(passage_tokens=8)
    assert scorer.name.endswith("+maxp8")
    assert scorer.passages("aaaa bbbb xxxxxxx cc") == [("aaaa bbbb", 8), ("xxxxxxx", 7), ("cc", 2)]
    assert scorer.passages("aaaa bbbb") == [("aaaa bbbb", 8)]
    scorer.score("abcd", ["aaaa bbbb xxxxxxx cc"])
    assert predictions[0][0] == [("abcd", "aaaa bbbb"), ("abcd", "xxxxxxx"), ("abcd", "cc")]


def test_a_passage_never_outgrows_what_fits_beside_the_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Window 23, 3 markers and a query of up to 10 leave 10 for the passage.
    _stand_in(monkeypatch)
    scorer = SentenceTransformerCrossEncoder(passage_tokens=100)
    assert scorer.passage_tokens == 10
    assert scorer.name.endswith("+maxp10")
    assert scorer.passages("abcdefghijkl") == [("abcdefghij", 10), ("kl", 2)]


def test_a_documents_score_is_its_best_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    _, predictions = _stand_in(monkeypatch)
    scorer = SentenceTransformerCrossEncoder(passage_tokens=8)
    # Passages with 8, 0 and 1 x's: the best is the first. Then a short one,
    # and one whose best passage is its last.
    long = "xxxxxxxx bbbbbbbb ax aaaa"
    assert len(scorer.passages(long)) == 3
    scores = scorer.score("abcd", [long, "xx", "aaaa bbbb xxxxxxx"])
    assert scores.tolist() == [8.0, 2.0, 7.0]
    # Every passage of every document went in one call, in order.
    [(pairs, _)] = predictions
    assert [passage for _, passage in pairs] == [
        "xxxxxxxx",
        "bbbbbbbb",
        "ax aaaa",
        "xx",
        "aaaa bbbb",
        "xxxxxxx",
    ]


# --- the real model --------------------------------------------------------


@pytest.fixture(scope="module")
def scorer() -> SentenceTransformerCrossEncoder:
    try:
        return SentenceTransformerCrossEncoder()
    except EncoderUnavailable as exc:  # pragma: no cover - depends on the machine
        if os.environ.get("RECRUITFLOW_REQUIRE_ENCODER"):
            raise
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


def _length(scorer: SentenceTransformerCrossEncoder, *texts: str) -> int:
    """Word-pieces the model's tokenizer gives ``texts``, its markers included."""
    return len(scorer._tokenizer(*texts)["input_ids"])


QUERY = "senior python developer\nbuild django services\npython django sql"
FILLER = "warehouse operative, forklift licence, stock rotation and picking. " * 60
MATCH = "python developer, five years building django apis on postgresql."


def test_real_model_name_and_window(scorer: SentenceTransformerCrossEncoder) -> None:
    assert scorer.name == f"{DEFAULT_CROSS_ENCODER}@{DEFAULT_CROSS_ENCODER_REVISION}+maxp128"
    assert (scorer.window, scorer.markers, scorer.query_tokens) == (512, 3, 254)
    assert scorer.passage_tokens == 128


def test_real_model_ranks_a_matching_resume_above_others(
    scorer: SentenceTransformerCrossEncoder,
) -> None:
    scores = scorer.score(
        QUERY,
        [
            "registered nurse\nicu nurse with ten years on wards\npatient care",
            "python developer\nfive years building django apis\npython django postgresql",
            "line cook\nprep and grill\nfood safety",
        ],
    )
    assert scores.shape == (3,) and np.isfinite(scores).all()
    assert scores.argmax() == 1
    # Raw logits, not probabilities: unrelated pairs score far below zero.
    assert scores.min() < -5


def test_real_model_corpus_pairs_are_read_whole_as_the_model_scores_them(
    scorer: SentenceTransformerCrossEncoder, corpus: Corpus
) -> None:
    query = job_document(corpus.jobs[0])
    documents = [candidate_document(c) for c in corpus.candidates[:8]]
    assert scorer.read_query(query) == (query, _length(scorer, query) - 2)
    scores = scorer.score(query, documents)
    for document, score in zip(documents, scores, strict=True):
        assert scorer.passages(document) == [(document, _length(scorer, document) - 2)]
        alone = scorer._model.predict([(query, document)], activation_fn=scorer._identity)
        # Padding in a batch moves a logit by a hair.
        assert score == pytest.approx(float(alone[0]), abs=1e-4)


def test_real_model_every_passage_fits_beside_the_longest_query(
    scorer: SentenceTransformerCrossEncoder, corpus: Corpus
) -> None:
    query = "\n".join(job_document(job) for job in corpus.jobs[:4])
    assert _length(scorer, query) - 2 > 254
    read, query_count = scorer.read_query(query)
    assert query.startswith(read) and 250 <= query_count <= 254
    document = " ".join(candidate_document(c) for c in corpus.candidates[:20])
    passages = scorer.passages(document)
    assert len(passages) > 2 and max(count for _, count in passages) <= 128
    for passage, count in passages:
        # Counted apart, a passage and the query tokenize to the same length as a pair.
        assert _length(scorer, read, passage) == query_count + count + 3 <= 512


def test_real_model_the_end_of_a_long_resume_reaches_its_score(
    scorer: SentenceTransformerCrossEncoder,
) -> None:
    without, with_match = scorer.score(QUERY, [FILLER, FILLER + MATCH])
    passages = scorer.passages(FILLER + MATCH)
    assert len(passages) > 4
    # Read as one window, the match barely moves the score (see app.ml.cross_encoder).
    assert with_match > without + 5
    best = scorer.score(QUERY, [passage for passage, _ in passages])
    assert best.argmax() == len(passages) - 1
    assert with_match == pytest.approx(float(best.max()), abs=1e-4)
