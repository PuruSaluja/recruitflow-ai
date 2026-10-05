"""Unit tests for app.ml.retrieval: BM25, cosine scores and reciprocal rank fusion."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from app.ml.retrieval import (
    BM25_B,
    BM25_K1,
    RRF_K,
    BM25Index,
    cosine_scores,
    fuse_per_group,
    reciprocal_rank_fusion,
)
from app.ml.text import TOKEN_PATTERN, fold_text

POOL = [
    "Python developer, Python and SQL",
    "Java developer with Spring",
    "Data analyst: SQL, Excel and Python",
    "Nurse with ICU experience",
    "Senior Python engineer who writes Python, Python and more Python every day",
]


def _reference_bm25(documents: list[str], query: str, k1: float, b: float) -> list[float]:
    """BM25 written out term by term, to check the vectorised index against."""
    docs = [re.findall(TOKEN_PATTERN, fold_text(d)) for d in documents]
    average = sum(len(d) for d in docs) / len(docs)
    terms = set(re.findall(TOKEN_PATTERN, fold_text(query)))
    out = []
    for doc in docs:
        total = 0.0
        for term in terms:
            n = sum(term in d for d in docs)
            if n == 0:
                continue
            idf = math.log(1 + (len(docs) - n + 0.5) / (n + 0.5))
            tf = doc.count(term)
            if tf == 0:  # contributes nothing, and k1 = 0 would divide 0 by 0
                continue
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / average))
        out.append(total)
    return out


@pytest.mark.parametrize(("k1", "b"), [(BM25_K1, BM25_B), (2.0, 0.0), (0.5, 1.0), (0.0, 0.5)])
@pytest.mark.parametrize(
    "query", ["Python SQL", "python python developer", "nurse ICU", "spring data excel"]
)
def test_scores_match_the_formula_written_out(query: str, k1: float, b: float) -> None:
    index = BM25Index(POOL, k1=k1, b=b)
    assert index.scores(query) == pytest.approx(_reference_bm25(POOL, query, k1, b), rel=1e-12)


def test_defaults_are_the_textbook_values() -> None:
    assert (BM25_K1, BM25_B, RRF_K) == (1.2, 0.75, 60)


def test_a_query_term_counts_once_however_often_it_is_repeated() -> None:
    index = BM25Index(POOL)
    np.testing.assert_array_equal(index.scores("python"), index.scores("python python python"))


def test_a_term_the_pool_never_uses_adds_nothing() -> None:
    index = BM25Index(POOL)
    np.testing.assert_array_equal(index.scores("kubernetes"), np.zeros(len(POOL)))
    np.testing.assert_array_equal(index.scores("python kubernetes"), index.scores("python"))


def test_term_frequency_saturates() -> None:
    # Same length, one mention against four: more mentions score higher, but
    # far less than four times as high.
    index = BM25Index(["python a b c", "python python python python", "x y z w"], b=0.0)
    one, four, _ = index.scores("python")
    assert one < four < 2 * one


def test_a_longer_document_scores_lower_for_the_same_mentions_unless_b_is_zero() -> None:
    pool = ["python", "python and many other words besides", "java"]
    short, long, _ = BM25Index(pool).scores("python")
    assert short > long
    short, long, _ = BM25Index(pool, b=0.0).scores("python")
    assert short == long


def test_a_term_in_every_document_still_counts_a_little() -> None:
    index = BM25Index(["python a", "python b", "python c"])
    assert (index.idf > 0).all()
    assert (index.scores("python") > 0).all()


def test_skill_names_stay_whole_and_text_is_folded() -> None:
    index = BM25Index(["C++ developer", "C developer", "Node.js developer", "node developer"])
    cpp, c, _, _ = index.scores("c++")
    assert cpp > 0 and c == 0
    _, _, nodejs, node = index.scores("node.js")
    assert nodejs > 0 and node == 0
    full_width = BM25Index(["Ｐｙｔｈｏｎ developer", "Java developer"])
    assert full_width.scores("python")[0] > 0


@pytest.mark.parametrize(
    ("documents", "kwargs", "message"),
    [
        ([], {}, "at least one document"),
        (["", "  "], {}, "at least one word"),
        (["a b"], {"k1": -0.1}, "k1"),
        (["a b"], {"b": 1.5}, "b must be"),
        (["a b"], {"b": -0.1}, "b must be"),
    ],
)
def test_an_index_refuses_what_it_cannot_score(
    documents: list[str], kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BM25Index(documents, **kwargs)


# --- cosine -----------------------------------------------------------------


def test_cosine_scores_ignore_length() -> None:
    stored = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(
        cosine_scores(stored, np.array([5.0, 0.0])), [1.0, 0.0, math.sqrt(0.5)]
    )


@pytest.mark.parametrize(
    ("stored", "query"),
    [
        (np.ones((2, 3)), np.ones(2)),
        (np.ones(3), np.ones(3)),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2)),
        (np.ones((2, 2)), np.zeros(2)),
    ],
)
def test_cosine_refuses_mismatched_or_zero_vectors(stored: np.ndarray, query: np.ndarray) -> None:
    with pytest.raises(ValueError):
        cosine_scores(stored, query)


# --- fusion -----------------------------------------------------------------


def test_fusion_sums_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([3.0, 2.0, 1.0], [0.1, 0.3, 0.2], k=60)
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    np.testing.assert_allclose(fused, expected, rtol=0, atol=1e-15)


def test_fusion_uses_ranks_only() -> None:
    a, b = np.array([10.0, -4.0, 2.5, 7.0]), np.array([0.2, 0.9, 0.1, 0.3])
    np.testing.assert_array_equal(
        reciprocal_rank_fusion(a, b), reciprocal_rank_fusion(a**3 * 100, np.exp(b))
    )


def test_tied_scores_share_their_average_rank() -> None:
    np.testing.assert_allclose(
        reciprocal_rank_fusion([5.0, 5.0, 1.0], k=0), [1 / 1.5, 1 / 1.5, 1 / 3]
    )


def test_fusion_does_not_depend_on_the_order_items_arrive_in() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 40).astype(float)  # many ties
    b = rng.standard_normal(40)
    order = rng.permutation(40)
    np.testing.assert_allclose(
        reciprocal_rank_fusion(a, b)[order], reciprocal_rank_fusion(a[order], b[order])
    )


def test_an_item_both_rank_first_is_first_when_fused() -> None:
    fused = reciprocal_rank_fusion([0.9, 0.5, 0.1], [12.0, 3.0, 7.0])
    assert int(np.argmax(fused)) == 0


@pytest.mark.parametrize(
    ("scores", "kwargs"),
    [
        ((), {}),
        (([1.0, 2.0], [1.0]), {}),
        (([[1.0, 2.0]],), {}),
        (([1.0, math.nan],), {}),
        (([1.0, 2.0],), {"k": -1}),
    ],
)
def test_fusion_refuses_what_it_cannot_rank(scores: tuple, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(*scores, **kwargs)


def test_fusing_per_group_ranks_each_group_on_its_own() -> None:
    groups = ["J2", "J1", "J2", "J1", "J2"]
    a = np.array([3.0, 1.0, 2.0, 5.0, 1.0])
    b = np.array([0.1, 0.4, 0.3, 0.2, 0.2])
    fused = fuse_per_group(groups, a, b)
    np.testing.assert_allclose(fused[[1, 3]], reciprocal_rank_fusion(a[[1, 3]], b[[1, 3]]))
    np.testing.assert_allclose(fused[[0, 2, 4]], reciprocal_rank_fusion(a[[0, 2, 4]], b[[0, 2, 4]]))


def test_fusing_per_group_refuses_misaligned_input() -> None:
    with pytest.raises(ValueError, match="group keys"):
        fuse_per_group(["J1", "J1"], [1.0, 2.0], [1.0])
