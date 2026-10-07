"""Unit tests for app.ml.retrieval: BM25, fusion, and the two-stage cut and re-rank."""

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
    first_stage_cut,
    fuse_per_group,
    recall_at_n,
    reciprocal_rank_fusion,
    rerank_per_group,
    rerank_top_n,
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
        (["a b"], {"k1": -0.1}, "k1 must be"),
        (["a b"], {"k1": math.nan}, "k1 must be"),
        (["a b"], {"k1": math.inf}, "k1 must be"),
        (["a b"], {"b": 1.5}, "b must be"),
        (["a b"], {"b": -0.1}, "b must be"),
        (["a b"], {"b": math.nan}, "b must be"),
    ],
)
def test_an_index_refuses_what_it_cannot_score(
    documents: list[str], kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BM25Index(documents, **kwargs)


def test_a_bare_string_is_refused_as_a_pool_rather_than_read_letter_by_letter() -> None:
    with pytest.raises(TypeError, match="a sequence of documents, not one string"):
        BM25Index("python developer")


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


def test_ranks_swapped_between_two_scorers_fuse_to_exactly_equal_scores() -> None:
    # 3rd and 7th against 7th and 3rd: a tie the fused ranking has to break.
    a = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    b = np.array([9.0, 8.0, 3.0, 6.0, 5.0, 4.0, 7.0])
    fused = reciprocal_rank_fusion(a, b)
    assert fused[2] == fused[6]


def test_fusing_a_slate_alone_can_order_it_differently_from_fusing_the_pool() -> None:
    a = np.array([5.0, 0.0, 1.0, 4.0, 2.0, 6.0, 3.0, 7.0])
    b = np.array([1.0, 6.0, 7.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    slate = [0, 1, 2]
    over_pool = reciprocal_rank_fusion(a, b)[slate]
    within = reciprocal_rank_fusion(a[slate], b[slate])
    assert np.argsort(-over_pool).tolist() == [2, 1, 0]
    assert np.argsort(-within).tolist() == [2, 0, 1]


@pytest.mark.parametrize(
    ("scores", "kwargs", "message"),
    [
        ((), {}, "at least one ranking"),
        (([1.0, 2.0], [1.0]), {}, "same items"),
        (([[1.0, 2.0]],), {}, "same items"),
        (([1.0, math.nan],), {}, "finite"),
        (([1.0, math.inf],), {}, "finite"),
        (([1.0, 2.0],), {"k": -1}, "k must be"),
        (([1.0, 2.0],), {"k": math.nan}, "k must be"),
        (([1.0, 2.0],), {"k": math.inf}, "k must be"),
    ],
)
def test_fusion_refuses_what_it_cannot_rank(scores: tuple, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
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


def test_fusing_per_group_refuses_keys_that_are_not_one_flat_sequence() -> None:
    with pytest.raises(ValueError, match="one key per item"):
        fuse_per_group([["J1"], ["J1"]], [[1.0], [2.0]])


# --- two stages ------------------------------------------------------------


def test_the_cut_keeps_the_top_n() -> None:
    kept = first_stage_cut([0.2, 0.9, 0.5, 0.1, 0.7], 3)
    assert kept.tolist() == [False, True, True, False, True]


def test_the_cut_keeps_every_item_tied_with_the_nth() -> None:
    kept = first_stage_cut([3.0, 2.0, 2.0, 2.0, 1.0], 2)
    assert kept.tolist() == [True, True, True, True, False]


def test_the_cut_keeps_everyone_when_n_reaches_the_end_or_all_tie() -> None:
    assert first_stage_cut([1.0, 2.0, 3.0], 3).all()
    assert first_stage_cut([1.0, 2.0, 3.0], 50).all()
    assert first_stage_cut([0.0] * 6, 1).all()


def test_the_cut_does_not_depend_on_the_order_items_arrive_in() -> None:
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 6, 60).astype(float)  # ties at most cuts
    order = rng.permutation(60)
    for n in (1, 7, 20, 59):
        assert (first_stage_cut(scores, n)[order] == first_stage_cut(scores[order], n)).all()


def test_recall_is_the_share_of_relevant_items_kept() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    relevant = [True, False, False, True, True, False]
    assert recall_at_n(scores, relevant, 1) == pytest.approx(1 / 3)
    assert recall_at_n(scores, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_n(scores, relevant, 5) == 1.0


def test_recall_counts_a_relevant_item_tied_at_the_cut_as_kept() -> None:
    assert recall_at_n([2.0, 1.0, 1.0], [False, False, True], 2) == 1.0


def test_recall_is_none_with_nothing_relevant() -> None:
    assert recall_at_n([1.0, 2.0], [False, False], 1) is None


@pytest.mark.parametrize(
    ("scores", "relevant", "n", "message"),
    [
        ([1.0, 2.0], [1, 0], 1, "booleans"),
        ([1.0, 2.0], [True], 1, "shape"),
        ([1.0, math.nan], [True, False], 1, "finite"),
        ([], [], 1, "non-empty"),
        ([1.0, 2.0], [True, False], 0, "at least 1"),
        ([1.0, 2.0], [True, False], 1.5, "whole number"),
        ([1.0, 2.0], [True, False], True, "whole number"),
    ],
)
def test_recall_refuses_what_it_cannot_measure(
    scores: list, relevant: list, n: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        recall_at_n(scores, np.array(relevant, dtype=None if relevant else bool), n)


def _two_stage_order(first: np.ndarray, second: np.ndarray, n: int) -> list[int]:
    """The two-stage order written out: the kept by second, then the rest by first."""
    threshold = sorted(first, reverse=True)[min(n, len(first)) - 1]
    kept = [i for i in range(len(first)) if first[i] >= threshold]
    rest = [i for i in range(len(first)) if first[i] < threshold]
    return sorted(kept, key=lambda i: -second[i]) + sorted(rest, key=lambda i: -first[i])


@pytest.mark.parametrize("n", [1, 3, 10, 29, 30, 45])
def test_the_re_rank_orders_the_kept_by_the_second_stage_above_the_rest(n: int) -> None:
    rng = np.random.default_rng(n)
    first, second = rng.standard_normal(30), rng.standard_normal(30)
    scores = rerank_top_n(first, second, n)
    assert np.argsort(-scores, kind="stable").tolist() == _two_stage_order(first, second, n)


def test_the_re_rank_returns_ranks_as_scores() -> None:
    first = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    second = np.array([0.0, 9.0, 1.0, 9.0, 2.0])
    # kept: items 0, 4, 2; second ranks them 1, 3, 2 among themselves.
    assert rerank_top_n(first, second, 3).tolist() == [7.0, 1.0, 8.0, 2.0, 9.0]


def test_the_re_rank_keeps_ties_in_either_stage_as_ties() -> None:
    first = np.array([5.0, 4.0, 3.0, 1.0, 1.0])
    second = np.array([0.5, 0.5, 0.2, 0.0, 0.0])
    scores = rerank_top_n(first, second, 3)
    assert scores[0] == scores[1] > scores[2] > scores[3] == scores[4]


def test_the_re_rank_never_reads_the_second_stage_of_an_item_it_cut() -> None:
    first = np.array([3.0, 2.0, 1.0])
    second = np.array([0.1, 0.2, math.nan])
    assert rerank_top_n(first, second, 2).tolist() == [5.0, 6.0, 1.0]


def test_the_re_rank_with_n_past_the_end_is_the_second_stage_alone() -> None:
    first, second = np.array([3.0, 2.0, 1.0]), np.array([0.1, 0.3, 0.2])
    assert np.argsort(-rerank_top_n(first, second, 3)).tolist() == [1, 2, 0]


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([1.0, 2.0], [1.0], "same items"),
        ([1.0, math.inf], [1.0, 2.0], "first-stage scores must be finite"),
        ([1.0, 2.0], [math.nan, 1.0], "second-stage scores of the kept items must be finite"),
        ([], [], "non-empty"),
    ],
)
def test_the_re_rank_refuses_what_it_cannot_rank(first: list, second: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rerank_top_n(first, second, 2)


def test_re_ranking_per_group_cuts_and_ranks_each_group_on_its_own() -> None:
    groups = ["J2", "J1", "J2", "J1", "J2", "J1"]
    first = np.array([3.0, 1.0, 2.0, 5.0, 1.0, 4.0])
    second = np.array([0.1, 0.9, 0.3, 0.2, 0.2, 0.4])
    out = rerank_per_group(groups, first, second, 2)
    for rows in ([1, 3, 5], [0, 2, 4]):
        np.testing.assert_array_equal(out[rows], rerank_top_n(first[rows], second[rows], 2))


def test_re_ranking_per_group_refuses_misaligned_input() -> None:
    with pytest.raises(ValueError, match="group keys"):
        rerank_per_group(["J1", "J1"], [1.0, 2.0], [1.0], 1)
