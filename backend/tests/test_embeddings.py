"""Tests for app.ml.embeddings: the stored float32 form and the exact search."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.embeddings import (
    BYTES_PER_VALUE,
    cosine_scores,
    from_bytes,
    stack,
    to_bytes,
    top_k_cosine,
)

# --- to_bytes / from_bytes -------------------------------------------------


def test_round_trip_is_exact_for_float32_values() -> None:
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(384).astype(np.float32)
    data = to_bytes(vector)
    assert len(data) == 384 * BYTES_PER_VALUE == 1536
    assert np.array_equal(from_bytes(data, 384), vector)


def test_stored_bytes_are_little_endian_whatever_the_input_order() -> None:
    expected = b"\x00\x00\x80\x3f" + b"\x00\x00\x00\xc0"  # 1.0, -2.0 as <f4
    assert to_bytes(np.array([1.0, -2.0], dtype="<f4")) == expected
    assert to_bytes(np.array([1.0, -2.0], dtype=">f4")) == expected
    assert to_bytes(np.array([1.0, -2.0], dtype=np.float64)) == expected
    assert to_bytes([1.0, -2.0]) == expected


def test_decoding_reads_little_endian_on_any_machine() -> None:
    decoded = from_bytes(b"\x00\x00\x80\x3f" + b"\x00\x00\x00\xc0", 2)
    assert decoded.tolist() == [1.0, -2.0]
    assert decoded.dtype == np.float32
    assert decoded.dtype.isnative


def test_decoded_vector_is_a_writable_copy() -> None:
    data = to_bytes([3.0, 4.0])
    decoded = from_bytes(data, 2)
    decoded /= np.linalg.norm(decoded)  # would raise on frombuffer's read-only view
    assert decoded.tolist() == pytest.approx([0.6, 0.8])
    assert from_bytes(data, 2).tolist() == [3.0, 4.0]


@pytest.mark.parametrize(
    "vector",
    [
        np.array([1.0, np.nan]),
        np.array([np.inf, 1.0]),
        np.array([-np.inf, 1.0]),
        np.array([1e39, 1.0]),  # finite as float64, infinity as float32
    ],
    ids=["nan", "inf", "-inf", "float32-overflow"],
)
def test_non_finite_values_are_refused(vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        to_bytes(vector)


def test_a_value_float32_can_still_hold_is_kept() -> None:
    assert from_bytes(to_bytes([3.0e38, 1.0]), 2)[0] == np.float32(3.0e38)


@pytest.mark.parametrize(
    "vector",
    [np.zeros(4), np.array([0.0, -0.0]), np.array([1e-50, 0.0])],
    ids=["zeros", "signed-zeros", "float32-underflow"],
)
def test_a_vector_with_no_direction_is_refused(vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="no direction"):
        to_bytes(vector)


def test_a_tiny_but_non_zero_vector_is_kept() -> None:
    assert to_bytes([1e-30, 0.0]) == np.array([1e-30, 0.0], dtype="<f4").tobytes()


@pytest.mark.parametrize(
    "vector",
    [np.array([]), np.array(1.0), np.ones((2, 2))],
    ids=["empty", "scalar", "matrix"],
)
def test_only_a_non_empty_1d_vector_can_be_stored(vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1-D"):
        to_bytes(vector)


@pytest.mark.parametrize("length", [0, 7, 9, 12])
def test_a_blob_of_the_wrong_length_is_refused(length: int) -> None:
    with pytest.raises(ValueError, match="is not 2 float32 values"):
        from_bytes(b"\x00" * length, 2)


@pytest.mark.parametrize("dimensions", [0, -1])
def test_dimensions_must_be_positive(dimensions: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        from_bytes(b"", dimensions)


# --- stack -----------------------------------------------------------------


def test_stack_keeps_the_order_and_values() -> None:
    vectors = [[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5], [0.0, 0.0, 7.0]]
    matrix = stack((to_bytes(v) for v in vectors), 3)
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    assert matrix.tolist() == vectors
    matrix[0, 0] = 9.0  # writable


def test_stack_of_nothing_is_an_empty_matrix_of_the_right_width() -> None:
    matrix = stack([], 5)
    assert matrix.shape == (0, 5)
    assert matrix.dtype == np.float32


def test_stack_refuses_one_short_blob_instead_of_shifting_the_rest() -> None:
    # Joined, these 16 bytes would read as two 2-D vectors, the second made
    # of the first one's tail and the third's head.
    blobs = [to_bytes([1.0, 2.0]), to_bytes([3.0])[:4], to_bytes([4.0, 5.0])[:4]]
    with pytest.raises(ValueError, match="is not 2 float32 values"):
        stack(blobs, 2)


# --- top_k_cosine ----------------------------------------------------------


def _reference_cosines(stored: np.ndarray, query: np.ndarray) -> list[float]:
    """Cosine per row, one at a time in float64, written from the definition."""
    out = []
    for row in stored.astype(np.float64):
        q = query.astype(np.float64)
        out.append(float(row @ q / (np.sqrt(row @ row) * np.sqrt(q @ q))))
    return out


def test_top_k_matches_a_from_the_definition_reference() -> None:
    rng = np.random.default_rng(7)
    stored = rng.standard_normal((500, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)
    indices, similarities = top_k_cosine(stored, query, 10)

    reference = _reference_cosines(stored, query)
    expected = sorted(range(500), key=lambda i: -reference[i])[:10]
    assert indices.tolist() == expected
    assert similarities.tolist() == pytest.approx([reference[i] for i in expected], abs=1e-6)
    assert all(a >= b for a, b in zip(similarities, similarities[1:], strict=False))


def test_similarity_ignores_length_and_reads_direction() -> None:
    stored = np.array([[10.0, 0.0], [0.0, 0.1], [-3.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    indices, similarities = top_k_cosine(stored, np.array([2.0, 0.0], dtype=np.float32), 4)
    assert indices.tolist() == [0, 3, 1, 2]
    assert similarities.tolist() == pytest.approx([1.0, 2**-0.5, 0.0, -1.0])


def test_equal_similarities_keep_the_stored_order() -> None:
    # Rows 1, 3 and 4 tie, and so do 0 and 2; each group stays in index order.
    stored = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.5, 0.0]], dtype=np.float32
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert top_k_cosine(stored, query, 5)[0].tolist() == [1, 3, 4, 0, 2]
    assert top_k_cosine(stored, query, 2)[0].tolist() == [1, 3]
    assert top_k_cosine(stored[::-1], query, 5)[0].tolist() == [0, 1, 3, 2, 4]


def test_ties_keep_the_stored_order_in_a_pool_of_real_size() -> None:
    # Five rows in a toy pool are sorted stably by any algorithm numpy picks;
    # 4,000 rows over three directions are not, unless the sort is stable.
    rng = np.random.default_rng(5)
    directions = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    group = rng.integers(0, 3, size=4000)
    # Lengths vary by powers of two, which scale a float exactly, so rows
    # sharing a direction tie exactly rather than to within rounding.
    scale = 2.0 ** rng.integers(0, 4, size=(4000, 1))
    stored = (directions[group] * scale).astype(np.float32)
    indices, similarities = top_k_cosine(stored, np.array([1.0, 0.0], np.float32), 4000)

    expected = [i for g in (0, 1, 2) for i in np.flatnonzero(group == g)]
    assert indices.tolist() == expected
    cut = int((group == 0).sum()) + 5  # five rows into the second tie group
    assert (
        top_k_cosine(stored, np.array([1.0, 0.0], np.float32), cut)[0].tolist() == (expected[:cut])
    )
    assert np.all(np.diff(similarities) <= 0)


def test_a_large_ranking_is_in_descending_order() -> None:
    rng = np.random.default_rng(9)
    stored = rng.standard_normal((3000, 24)).astype(np.float32)
    query = rng.standard_normal(24).astype(np.float32)
    for k in (25, 3000):
        indices, similarities = top_k_cosine(stored, query, k)
        assert np.all(np.diff(similarities) <= 0)
        reference = (stored @ query) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(query))
        assert indices.tolist() == np.argsort(-reference, kind="stable")[:k].tolist()


def test_k_is_a_cut_of_the_full_ranking() -> None:
    rng = np.random.default_rng(3)
    stored = rng.standard_normal((200, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    full, _ = top_k_cosine(stored, query, 200)
    for k in (1, 2, 5, 37, 199):
        assert top_k_cosine(stored, query, k)[0].tolist() == full[:k].tolist()


def test_k_beyond_the_pool_returns_the_whole_pool() -> None:
    stored = np.eye(3, dtype=np.float32)
    indices, similarities = top_k_cosine(stored, np.array([0.0, 1.0, 0.0], np.float32), 10)
    assert indices.tolist() == [1, 0, 2]
    assert len(similarities) == 3


def test_an_empty_pool_returns_nothing() -> None:
    indices, similarities = top_k_cosine(np.empty((0, 4), np.float32), np.ones(4, np.float32), 5)
    assert indices.tolist() == [] and similarities.tolist() == []


def test_search_does_not_modify_its_inputs() -> None:
    stored = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    query = np.array([0.0, 2.0], dtype=np.float32)
    top_k_cosine(stored, query, 2)
    assert stored.tolist() == [[3.0, 4.0], [1.0, 0.0]]
    assert query.tolist() == [0.0, 2.0]


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        top_k_cosine(np.eye(2, dtype=np.float32), np.ones(2, np.float32), k)


@pytest.mark.parametrize(
    ("stored", "query"),
    [
        (np.eye(3), np.ones(2)),
        (np.ones(3), np.ones(3)),
        (np.eye(3), np.eye(3)),
    ],
    ids=["width-mismatch", "1d-pool", "2d-query"],
)
def test_shapes_must_agree(stored: np.ndarray, query: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cannot compare"):
        top_k_cosine(stored, query, 1)


@pytest.mark.parametrize(
    ("stored", "query"),
    [
        (np.array([[1.0, 0.0], [np.nan, 1.0]], np.float32), np.ones(2, np.float32)),
        (np.array([[1.0, 0.0], [np.inf, 1.0]], np.float32), np.ones(2, np.float32)),
        (np.eye(2, dtype=np.float32), np.array([np.nan, 1.0], np.float32)),
        (np.eye(2, dtype=np.float32), np.array([-np.inf, 1.0], np.float32)),
    ],
    ids=["nan-row", "inf-row", "nan-query", "inf-query"],
)
@pytest.mark.filterwarnings("error::RuntimeWarning")  # refused, not warned about first
def test_a_vector_that_is_not_finite_is_refused(stored: np.ndarray, query: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        top_k_cosine(stored, query, 1)
    with pytest.raises(ValueError, match="finite"):
        cosine_scores(stored, query)


def test_cosine_scores_are_in_row_order_and_ignore_length() -> None:
    stored = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(cosine_scores(stored, np.array([5.0, 0.0])), [1.0, 0.0, 0.5**0.5])


@pytest.mark.parametrize(
    ("stored", "query", "message"),
    [
        (np.ones((2, 3)), np.ones(2), "cannot compare"),
        (np.ones(3), np.ones(3), "cannot compare"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2), "zero vector"),
        (np.ones((2, 2)), np.zeros(2), "zero vector"),
    ],
)
def test_cosine_scores_refuse_mismatched_or_zero_vectors(
    stored: np.ndarray, query: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_scores(stored, query)


def test_a_zero_query_or_row_is_refused() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        top_k_cosine(np.eye(2, dtype=np.float32), np.zeros(2, np.float32), 1)
    with pytest.raises(ValueError, match="zero vector"):
        top_k_cosine(np.array([[1.0, 0.0], [0.0, 0.0]], np.float32), np.ones(2, np.float32), 1)


def test_searching_stored_vectors_end_to_end() -> None:
    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((50, 32))
    matrix = stack((to_bytes(v) for v in vectors), 32)
    query = from_bytes(to_bytes(vectors[17]), 32)
    indices, similarities = top_k_cosine(matrix, query, 3)
    assert indices[0] == 17
    assert similarities[0] == pytest.approx(1.0, abs=1e-6)
