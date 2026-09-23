"""API tests for the matching endpoints: scoring a job's pool, and reading back the ranking.

The matcher is really trained on a small synthetic corpus and written to disk,
and the app finds it through ``RECRUITFLOW_MODEL_PATH`` -- the setting a
deployment uses -- so the real load path runs rather than an injected model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete, insert, select
from sqlalchemy.orm import Session

from app.api.v1 import matches
from app.config import get_settings
from app.ml.artifact import load_matcher, matcher_version, save_matcher
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.features import FEATURE_NAMES, feature_dict
from app.ml.train import train_matcher, training_data
from app.models import Candidate, Job, MatchScore
from app.schemas.match import MAX_CANDIDATE_IDS
from app.services.matching import score_pool

Factory = Callable[..., dict[str, Any]]
Pool = tuple[dict[str, Any], list[dict[str, Any]]]

SUMMARY_FIELDS = {
    "id",
    "full_name",
    "email",
    "current_title",
    "years_experience",
    "seniority",
    "skills",
}


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


@pytest.fixture(scope="module")
def artifact(corpus, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A matcher really trained on the corpus, saved the way the training script saves it."""
    path = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    return save_matcher(train_matcher(training_data(corpus)), manifest={}, path=path)


@pytest.fixture
def use_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path], None]]:
    """Point the app's configured model path at a file for the rest of one test."""

    def _use(path: Path) -> None:
        monkeypatch.setenv("RECRUITFLOW_MODEL_PATH", str(path))
        get_settings.cache_clear()

    yield _use
    get_settings.cache_clear()


@pytest.fixture
def trained(artifact: Path, use_model: Callable[[Path], None]) -> Path:
    use_model(artifact)
    return artifact


@pytest.fixture
def pool(make_job: Factory, make_candidate: Factory, corpus) -> Pool:
    """One job and six candidates from the corpus, created through the API."""
    job = make_job(**corpus.jobs[0].api_payload())
    candidates = [make_candidate(**c.api_payload()) for c in corpus.candidates[:6]]
    return job, candidates


def _cached(engine: Engine, job_id: int) -> list[MatchScore]:
    with Session(engine) as db:
        return list(
            db.scalars(
                select(MatchScore).where(MatchScore.job_id == job_id).order_by(MatchScore.id)
            )
        )


def _ids(rows: list[dict[str, Any]]) -> list[int]:
    return [row["candidate"]["id"] for row in rows]


# --- scoring a pool ---------------------------------------------------------


def test_scoring_without_a_body_scores_every_candidate_on_file(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, candidates = pool

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job_id"] == job["id"]
    assert body["model_version"] == matcher_version(trained)
    assert body["scored_count"] == len(candidates)
    assert sorted(_ids(body["results"])) == sorted(c["id"] for c in candidates)


def test_results_are_ranked_best_first_from_1(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, candidates = pool

    results = client.post(f"/jobs/{job['id']}/score").json()["results"]

    assert [row["rank"] for row in results] == list(range(1, len(candidates) + 1))
    scores = [row["score"] for row in results]
    assert scores == sorted(scores, reverse=True)


def test_every_candidate_scored_is_cached_under_the_matchers_version(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool

    client.post(f"/jobs/{job['id']}/score")

    cached = _cached(clean_db, job["id"])
    assert sorted(row.candidate_id for row in cached) == sorted(c["id"] for c in candidates)
    assert {row.model_version for row in cached} == {matcher_version(trained)}


def test_each_score_is_the_matchers_probability_for_that_pair(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    """Pins that each score reaches the candidate it belongs to, not a neighbour's."""
    job, _ = pool

    results = client.post(f"/jobs/{job['id']}/score").json()["results"]

    with Session(clean_db) as db:
        job_row = db.get(Job, job["id"])
        everyone = list(db.scalars(select(Candidate).order_by(Candidate.id)))
        expected = load_matcher(trained).predict_proba([(job_row, c) for c in everyone])[:, 1]
    by_id = {
        candidate.id: float(score) for candidate, score in zip(everyone, expected, strict=True)
    }
    for row in results:
        assert row["score"] == by_id[row["candidate"]["id"]]


def test_each_breakdown_is_the_feature_dict_of_its_pair(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, _ = pool

    results = client.post(f"/jobs/{job['id']}/score").json()["results"]

    with Session(clean_db) as db:
        job_row = db.get(Job, job["id"])
        for row in results:
            candidate = db.get(Candidate, row["candidate"]["id"])
            assert row["features"] == feature_dict(job_row, candidate)


def test_each_result_carries_a_summary_of_its_candidate_and_no_more(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool

    results = client.post(f"/jobs/{job['id']}/score").json()["results"]

    for row in results:
        assert set(row["candidate"]) == SUMMARY_FIELDS
        full = client.get(f"/candidates/{row['candidate']['id']}").json()
        assert row["candidate"] == {field: full[field] for field in SUMMARY_FIELDS}


def test_candidate_ids_restrict_scoring_to_those_candidates(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool
    chosen = [candidates[4]["id"], candidates[1]["id"]]

    body = client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": chosen}).json()

    assert body["scored_count"] == 2
    assert sorted(_ids(body["results"])) == sorted(chosen)
    assert sorted(row.candidate_id for row in _cached(clean_db, job["id"])) == sorted(chosen)


def test_limit_trims_the_results_but_the_whole_pool_is_scored(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool

    body = client.post(f"/jobs/{job['id']}/score", json={"limit": 2}).json()

    assert [row["rank"] for row in body["results"]] == [1, 2]
    assert body["scored_count"] == len(candidates)
    assert len(_cached(clean_db, job["id"])) == len(candidates)


def test_a_repeated_candidate_id_is_scored_once(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, candidates = pool
    first, second = candidates[0]["id"], candidates[1]["id"]

    body = client.post(
        f"/jobs/{job['id']}/score", json={"candidate_ids": [first, second, first]}
    ).json()

    assert body["scored_count"] == 2
    assert sorted(_ids(body["results"])) == sorted([first, second])


def test_an_empty_id_list_scores_nobody(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, _ = pool

    response = client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": []})

    assert response.status_code == 200, response.text
    assert response.json()["scored_count"] == 0
    assert response.json()["results"] == []
    assert _cached(clean_db, job["id"]) == []


def test_scoring_with_no_candidates_on_file_scores_nobody(
    client: TestClient, trained: Path, make_job: Factory
) -> None:
    job = make_job()

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 200, response.text
    assert response.json()["scored_count"] == 0


def test_scoring_again_updates_the_cached_rows_instead_of_adding_more(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool

    first = client.post(f"/jobs/{job['id']}/score").json()
    rows_before = {row.candidate_id: row.id for row in _cached(clean_db, job["id"])}
    second = client.post(f"/jobs/{job['id']}/score").json()
    rows_after = {row.candidate_id: row.id for row in _cached(clean_db, job["id"])}

    assert rows_after == rows_before
    assert len(rows_after) == len(candidates)
    assert datetime.fromisoformat(second["results"][0]["scored_at"]) > datetime.fromisoformat(
        first["results"][0]["scored_at"]
    )


# --- what a scoring request may not do ---------------------------------------


def test_scoring_a_missing_job_is_a_404_even_without_a_matcher(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path
) -> None:
    """The request is checked before the matcher is looked for."""
    use_model(tmp_path / "missing.joblib")

    response = client.post("/jobs/999999/score")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job 999999 not found"


def test_an_unknown_candidate_id_is_a_404_even_without_a_matcher(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path, pool: Pool
) -> None:
    use_model(tmp_path / "missing.joblib")
    job, _ = pool

    response = client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": [999999]})

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidates not found: 999999"


def test_an_unknown_candidate_id_is_a_404_naming_every_missing_one(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool

    response = client.post(
        f"/jobs/{job['id']}/score", json={"candidate_ids": [999999, candidates[0]["id"], 888888]}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidates not found: 888888, 999999"
    assert _cached(clean_db, job["id"]) == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"candidate_id": [1]}, id="misspelt-field"),
        pytest.param({"candidate_ids": [0]}, id="zero-id"),
        pytest.param({"candidate_ids": [-3]}, id="negative-id"),
        pytest.param({"candidate_ids": "1,2"}, id="ids-as-a-string"),
        pytest.param({"candidate_ids": list(range(1, MAX_CANDIDATE_IDS + 2))}, id="too-many-ids"),
        pytest.param({"limit": 0}, id="limit-0"),
        pytest.param({"limit": 501}, id="limit-501"),
    ],
)
def test_a_malformed_scoring_request_is_a_422(
    client: TestClient, pool: Pool, body: dict[str, Any]
) -> None:
    job, _ = pool

    assert client.post(f"/jobs/{job['id']}/score", json=body).status_code == 422


# --- no usable matcher ---------------------------------------------------------


def test_scoring_without_a_trained_matcher_is_a_503_naming_the_fix(
    client: TestClient,
    use_model: Callable[[Path], None],
    tmp_path: Path,
    pool: Pool,
    clean_db: Engine,
) -> None:
    missing = tmp_path / "missing.joblib"
    use_model(missing)
    job, _ = pool

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 503
    assert response.json()["code"] == "matcher_unavailable"
    assert "python scripts/train_matcher.py" in response.json()["detail"]
    assert _cached(clean_db, job["id"]) == []


def test_the_503_does_not_reveal_where_the_server_keeps_its_files(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path, pool: Pool
) -> None:
    missing = tmp_path / "missing.joblib"
    use_model(missing)
    job, _ = pool

    detail = client.post(f"/jobs/{job['id']}/score").json()["detail"]

    assert tmp_path.name not in detail
    assert missing.name not in detail


def test_an_unreadable_matcher_is_a_503_too(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path, pool: Pool
) -> None:
    damaged = tmp_path / "matcher.joblib"
    damaged.write_bytes(b"not a pickle at all\n")
    use_model(damaged)
    job, _ = pool

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 503
    assert response.json()["code"] == "matcher_unavailable"


def test_the_503_log_names_the_file_and_the_underlying_error(
    client: TestClient,
    use_model: Callable[[Path], None],
    tmp_path: Path,
    pool: Pool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    damaged = tmp_path / "matcher.joblib"
    damaged.write_bytes(b"not a pickle at all\n")
    use_model(damaged)
    job, _ = pool

    with caplog.at_level(logging.ERROR, logger="app.main"):
        client.post(f"/jobs/{job['id']}/score")

    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors, "the handler logged nothing"
    assert str(damaged) in errors[0].getMessage()
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[1].__cause__ is not None


# --- reading the cached ranking ----------------------------------------------


def test_the_listing_repeats_the_ranking_scoring_returned(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool

    scored = client.post(f"/jobs/{job['id']}/score").json()
    listed = client.get(f"/jobs/{job['id']}/matches").json()

    assert listed["job_id"] == job["id"]
    assert listed["model_version"] == scored["model_version"]
    assert listed["total"] == scored["scored_count"]
    for got, want in zip(listed["items"], scored["results"], strict=True):
        assert datetime.fromisoformat(got.pop("scored_at")) == datetime.fromisoformat(
            want.pop("scored_at")
        )
        assert got == want


def test_the_listing_pages_with_ranks_continuing_across_pages(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, candidates = pool
    client.post(f"/jobs/{job['id']}/score")
    everything = client.get(f"/jobs/{job['id']}/matches").json()["items"]

    pages = [
        client.get(f"/jobs/{job['id']}/matches", params={"limit": 4, "offset": offset}).json()
        for offset in (0, 4)
    ]

    assert [page["total"] for page in pages] == [len(candidates), len(candidates)]
    assert [len(page["items"]) for page in pages] == [4, 2]
    assert pages[0]["items"] + pages[1]["items"] == everything
    assert [item["rank"] for item in pages[1]["items"]] == [5, 6]


def test_the_listing_ranks_stay_1_to_total_after_a_smaller_rescore(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool
    first = client.post(f"/jobs/{job['id']}/score").json()["results"]
    bottom_two = _ids(first[-2:])
    client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": bottom_two})
    stored_ranks = [row.rank for row in _cached(clean_db, job["id"])]
    assert sorted(stored_ranks) == [1, 1, 2, 2, 3, 4], "premise: the smaller run reused 1 and 2"

    items = client.get(f"/jobs/{job['id']}/matches").json()["items"]

    assert [item["rank"] for item in items] == list(range(1, len(candidates) + 1))
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)


def test_tied_candidates_are_listed_in_the_order_scoring_ranked_them(
    client: TestClient, trained: Path, make_job: Factory, make_candidate: Factory, corpus
) -> None:
    """Identical on every field the matcher reads, so they tie; both endpoints put the lower
    id first."""
    twin = {
        "years_experience": 4,
        "seniority": "mid",
        "education_level": "bachelor",
        "current_title": "Software Engineer",
        "resume_text": "Shipped internal tools end to end with Python, SQL and some React.",
        "skills": ["Python", "SQL", "React"],
    }
    job = make_job(**corpus.jobs[0].api_payload())
    first, second = make_candidate(**twin), make_candidate(**twin)

    scored = client.post(f"/jobs/{job['id']}/score").json()["results"]
    listed = client.get(f"/jobs/{job['id']}/matches").json()["items"]

    assert scored[0]["score"] == scored[1]["score"]
    assert _ids(scored) == [first["id"], second["id"]]
    assert _ids(listed) == [first["id"], second["id"]]


def test_nothing_scored_yet_is_an_empty_listing(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool

    response = client.get(f"/jobs/{job['id']}/matches")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0


def test_the_listing_leaves_out_scores_from_another_matcher(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, candidates = pool
    chosen = [candidates[0]["id"], candidates[1]["id"]]
    client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": chosen})
    with Session(clean_db) as db:
        everyone = list(db.scalars(select(Candidate)))
        score_pool(db, db.get(Job, job["id"]), everyone, load_matcher(trained), "0" * 16)
        db.commit()
    assert len(_cached(clean_db, job["id"])) == 2 + len(candidates)

    body = client.get(f"/jobs/{job['id']}/matches").json()

    assert body["total"] == 2
    assert sorted(_ids(body["items"])) == sorted(chosen)


def test_listing_a_missing_job_is_a_404_even_without_a_matcher(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path
) -> None:
    use_model(tmp_path / "missing.joblib")

    assert client.get("/jobs/999999/matches").status_code == 404


def test_listing_without_a_trained_matcher_is_a_503(
    client: TestClient, use_model: Callable[[Path], None], tmp_path: Path, pool: Pool
) -> None:
    use_model(tmp_path / "missing.joblib")
    job, _ = pool

    response = client.get(f"/jobs/{job['id']}/matches")

    assert response.status_code == 503
    assert response.json()["code"] == "matcher_unavailable"


def test_listing_with_an_unreadable_matcher_is_a_503_too(
    client: TestClient,
    trained: Path,
    use_model: Callable[[Path], None],
    tmp_path: Path,
    pool: Pool,
) -> None:
    """Hashing alone would give the damaged file a version no score carries, and list nothing."""
    job, _ = pool
    client.post(f"/jobs/{job['id']}/score")
    damaged = tmp_path / "matcher.joblib"
    damaged.write_bytes(trained.read_bytes()[:-1])
    use_model(damaged)

    response = client.get(f"/jobs/{job['id']}/matches")

    assert response.status_code == 503
    assert response.json()["code"] == "matcher_unavailable"


# --- a run that loses a race ---------------------------------------------------


class _Interference:
    """Stands in for ``score_pool``: scores for real, then lets another "request" commit.

    ``action`` runs on the listed calls, after scoring and before the endpoint
    commits -- the window in which a concurrent request can change the rows.
    """

    def __init__(self, action: Callable[[int, Job, list[Candidate], str], None], on_calls=(1,)):
        self.action = action
        self.on_calls = on_calls
        self.calls = 0

    def __call__(self, db, job, candidates, matcher, model_version):
        self.calls += 1
        ranked = score_pool(db, job, candidates, matcher, model_version)
        if self.calls in self.on_calls:
            self.action(self.calls, job, candidates, model_version)
        return ranked


def _insert_a_score_first(engine: Engine) -> Callable[[int, Job, list[Candidate], str], None]:
    """Commits the row this run is about to insert, for a different candidate each call."""

    def action(call: int, job: Job, candidates: list[Candidate], version: str) -> None:
        with engine.begin() as conn:
            conn.execute(
                insert(MatchScore).values(
                    job_id=job.id,
                    candidate_id=candidates[call - 1].id,
                    model_version=version,
                    score=0.5,
                    rank=1,
                    features={},
                )
            )

    return action


def _delete(engine: Engine, model: type, row_id: int) -> Callable[..., None]:
    def action(*_: object) -> None:
        with engine.begin() as conn:
            conn.execute(delete(model).where(model.id == row_id))

    return action


def test_a_run_that_loses_an_insert_race_is_scored_again_and_succeeds(
    client: TestClient,
    trained: Path,
    pool: Pool,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, candidates = pool
    interference = _Interference(_insert_a_score_first(clean_db))
    monkeypatch.setattr(matches, "score_pool", interference)

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 200, response.text
    assert interference.calls == 2
    assert response.json()["scored_count"] == len(candidates)
    cached = _cached(clean_db, job["id"])
    assert len(cached) == len(candidates)
    # The winner's row was updated with this run's real score, not left as it was.
    assert all(set(row.features) == set(FEATURE_NAMES) for row in cached)


def test_a_run_that_loses_twice_is_a_409_and_writes_nothing(
    client: TestClient,
    trained: Path,
    pool: Pool,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, _ = pool
    interference = _Interference(_insert_a_score_first(clean_db), on_calls=(1, 2))
    monkeypatch.setattr(matches, "score_pool", interference)

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 409
    assert "Retry" in response.json()["detail"]
    assert interference.calls == 2
    assert [row.score for row in _cached(clean_db, job["id"])] == [0.5, 0.5]


def test_a_candidate_deleted_mid_run_drops_out_and_the_rest_are_scored(
    client: TestClient,
    trained: Path,
    pool: Pool,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scored once already, so this run updates rows -- and one of them is deleted under it."""
    job, candidates = pool
    client.post(f"/jobs/{job['id']}/score")
    gone = candidates[2]["id"]
    interference = _Interference(_delete(clean_db, Candidate, gone))
    monkeypatch.setattr(matches, "score_pool", interference)

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 200, response.text
    assert interference.calls == 2
    assert response.json()["scored_count"] == len(candidates) - 1
    assert gone not in _ids(response.json()["results"])


def test_a_candidate_asked_for_and_deleted_mid_run_is_a_404(
    client: TestClient,
    trained: Path,
    pool: Pool,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, candidates = pool
    chosen = [c["id"] for c in candidates[:3]]
    client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": chosen})
    before = {row.candidate_id: row.scored_at for row in _cached(clean_db, job["id"])}
    monkeypatch.setattr(
        matches, "score_pool", _Interference(_delete(clean_db, Candidate, chosen[0]))
    )

    response = client.post(f"/jobs/{job['id']}/score", json={"candidate_ids": chosen})

    assert response.status_code == 404
    assert response.json()["detail"] == f"Candidates not found: {chosen[0]}"
    # Untouched rather than rescored: a committed rescore would have moved scored_at.
    after = {row.candidate_id: row.scored_at for row in _cached(clean_db, job["id"])}
    assert after == {cid: at for cid, at in before.items() if cid != chosen[0]}


def test_a_job_deleted_mid_run_is_a_404(
    client: TestClient,
    trained: Path,
    pool: Pool,
    clean_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, _ = pool
    monkeypatch.setattr(matches, "score_pool", _Interference(_delete(clean_db, Job, job["id"])))

    response = client.post(f"/jobs/{job['id']}/score")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Job {job['id']} not found"
