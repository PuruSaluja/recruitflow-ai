"""Unit tests for the matching schemas. No database needed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.features import FEATURE_NAMES, feature_dict
from app.ml.train import training_data
from app.schemas.match import MAX_CANDIDATE_IDS, MatchFeatures, RankedCandidate, ScoreJobRequest

_BOUNDS = {name: (0.0, 1.0) for name in FEATURE_NAMES} | {"seniority_direction": (-1.0, 1.0)}


# --- MatchFeatures --------------------------------------------------------


def test_the_breakdown_is_exactly_the_stored_features_in_order() -> None:
    assert tuple(MatchFeatures.model_fields) == FEATURE_NAMES


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_each_feature_is_held_to_its_documented_range(name: str) -> None:
    low, high = _BOUNDS[name]
    at_lower_bounds = {feature: _BOUNDS[feature][0] for feature in FEATURE_NAMES}
    MatchFeatures(**at_lower_bounds)
    MatchFeatures(**{**at_lower_bounds, name: high})

    for outside in (low - 0.001, high + 0.001):
        with pytest.raises(ValidationError):
            MatchFeatures(**{**at_lower_bounds, name: outside})


def test_every_breakdown_feature_dict_produces_fits_the_schema() -> None:
    """The ranges above hold for what the features really produce, not just on paper."""
    corpus = generate_corpus(CorpusConfig(seed=3, n_jobs=20, n_candidates=600))

    for job, candidate in training_data(corpus).pairs:
        MatchFeatures.model_validate(feature_dict(job, candidate))


# --- RankedCandidate ------------------------------------------------------


def _ranked(**overrides: object) -> dict[str, object]:
    return {
        "rank": 1,
        "score": 0.5,
        "scored_at": "2026-09-22T12:00:00Z",
        "candidate": {
            "id": 1,
            "full_name": "Ada Example",
            "email": "ada@example.com",
            "current_title": None,
            "years_experience": 3,
            "seniority": "mid",
            "skills": [],
        },
        "features": {feature: _BOUNDS[feature][0] for feature in FEATURE_NAMES},
        **overrides,
    }


def test_a_ranked_candidate_validates() -> None:
    RankedCandidate.model_validate(_ranked())


@pytest.mark.parametrize(
    "overrides", [{"rank": 0}, {"score": -0.01}, {"score": 1.01}], ids=["rank-0", "below", "above"]
)
def test_a_ranked_candidate_refuses_an_impossible_rank_or_score(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        RankedCandidate.model_validate(_ranked(**overrides))


# --- ScoreJobRequest ------------------------------------------------------


def test_an_empty_request_scores_everyone_and_returns_fifty() -> None:
    request = ScoreJobRequest()

    assert request.candidate_ids is None
    assert request.limit == 50


def test_an_explicit_null_list_also_means_everyone() -> None:
    assert ScoreJobRequest.model_validate({"candidate_ids": None}).candidate_ids is None


def test_an_empty_list_is_accepted_and_means_nobody() -> None:
    assert ScoreJobRequest(candidate_ids=[]).candidate_ids == []


def test_a_misspelt_field_is_refused_rather_than_scoring_everyone() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScoreJobRequest.model_validate({"candidate_id": [1, 2]})


@pytest.mark.parametrize("bad_id", [0, -1])
def test_a_non_positive_candidate_id_is_refused(bad_id: int) -> None:
    with pytest.raises(ValidationError):
        ScoreJobRequest(candidate_ids=[1, bad_id])


def test_candidate_ids_are_capped() -> None:
    ScoreJobRequest(candidate_ids=list(range(1, MAX_CANDIDATE_IDS + 1)))

    with pytest.raises(ValidationError):
        ScoreJobRequest(candidate_ids=list(range(1, MAX_CANDIDATE_IDS + 2)))


@pytest.mark.parametrize(("limit", "valid"), [(0, False), (1, True), (500, True), (501, False)])
def test_limit_runs_from_1_to_500(limit: int, valid: bool) -> None:
    if valid:
        assert ScoreJobRequest(limit=limit).limit == limit
    else:
        with pytest.raises(ValidationError):
            ScoreJobRequest(limit=limit)
