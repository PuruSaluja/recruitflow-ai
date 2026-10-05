"""Tests for app.ml.encoder.

The tests that run the real model are skipped where ``sentence-transformers``
is not installed or its pinned model is not cached, which includes CI: the
``embeddings`` extra is not part of the dev install.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

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
        current_title="Ｄata Analyst",
        resume_text="Certiﬁed in Py​thon.",
        skills=["SQL", "Python"],
    )
    job = SimpleNamespace(
        title="Data Analyst",
        description="Ｗeekly metrics.",
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
        def __init__(self, model: str, **kwargs: object) -> None:
            calls.append((model, kwargs))

        def get_embedding_dimension(self) -> int:
            return 7

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Recording)
    )
    encoder = SentenceTransformerEncoder("org/model", "abc123", device="cuda", batch_size=5)
    assert calls == [("org/model", {"revision": "abc123", "device": "cuda"})]
    assert encoder.name == "org/model@abc123"
    assert encoder.dimensions == 7
    assert encoder.batch_size == 5
    assert SentenceTransformerEncoder().name == f"{DEFAULT_MODEL}@{DEFAULT_REVISION}"
    assert calls[-1] == (DEFAULT_MODEL, {"revision": DEFAULT_REVISION, "device": "cpu"})


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
    pytest.importorskip("sentence_transformers")
    try:
        return SentenceTransformerEncoder()
    except EncoderUnavailable as exc:  # pragma: no cover - depends on the machine
        pytest.skip(str(exc))


def test_real_model_shape_type_and_name(encoder: SentenceTransformerEncoder) -> None:
    assert encoder.name == f"{DEFAULT_MODEL}@{DEFAULT_REVISION}"
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
