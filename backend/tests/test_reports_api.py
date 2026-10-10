"""API tests for the PDF report endpoints: a job's shortlist and one candidate's scorecard.

As in the matching tests, the matcher is really trained on a small synthetic
corpus and found through ``RECRUITFLOW_MODEL_PATH``, and the reports are read
back with pypdf, so what is checked is the text a recruiter would see.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from sqlalchemy import Engine, delete, event, insert, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.artifact import save_matcher
from app.ml.corpus import CorpusConfig, generate_corpus
from app.ml.train import train_matcher, training_data
from app.models import MatchScore
from app.services.reports import MAX_SHORTLIST

Factory = Callable[..., dict[str, Any]]
Pool = tuple[dict[str, Any], list[dict[str, Any]]]


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(seed=5, n_jobs=40, n_candidates=1200))


@pytest.fixture(scope="module")
def artifact(corpus, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("matcher") / "matcher.joblib"
    return save_matcher(train_matcher(training_data(corpus)), manifest={}, path=path)


@pytest.fixture
def trained(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("RECRUITFLOW_MODEL_PATH", str(artifact))
    get_settings.cache_clear()
    yield artifact
    get_settings.cache_clear()


@pytest.fixture
def pool(make_job: Factory, make_candidate: Factory, corpus) -> Pool:
    """One job and six candidates from the corpus, created through the API."""
    job = make_job(**corpus.jobs[0].api_payload())
    candidates = [make_candidate(**c.api_payload()) for c in corpus.candidates[:6]]
    return job, candidates


def _text(response) -> str:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    reader = PdfReader(io.BytesIO(response.content))
    return "\n".join(page.extract_text() for page in reader.pages)


def _scored(client: TestClient, job_id: int) -> list[dict[str, Any]]:
    response = client.post(f"/jobs/{job_id}/score")
    assert response.status_code == 200, response.text
    return response.json()["results"]


# --- shortlist -------------------------------------------------------------


def test_the_shortlist_lists_the_cached_ranking_best_first(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool
    scored = _scored(client, job["id"])
    version = client.get(f"/jobs/{job['id']}/matches").json()["model_version"]

    response = client.get(f"/jobs/{job['id']}/report.pdf")
    text = _text(response)

    assert response.headers["content-disposition"] == (
        f'inline; filename="job-{job["id"]}-shortlist.pdf"'
    )
    assert f"Shortlist: {job['title']}" in text
    assert "top 6 of 6" in text
    positions = [text.index(row["candidate"]["full_name"]) for row in scored]
    assert positions == sorted(positions)
    for row in scored:
        assert f"{row['score']:.3f}" in text
    assert f"Matcher {version}" in text


def test_a_limit_trims_the_shortlist_but_not_the_total(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool
    scored = _scored(client, job["id"])

    text = _text(client.get(f"/jobs/{job['id']}/report.pdf", params={"limit": 2}))

    assert "top 2 of 6" in text
    assert all(row["candidate"]["full_name"] in text for row in scored[:2])
    assert not any(row["candidate"]["full_name"] in text for row in scored[2:])


@pytest.mark.parametrize("limit", [0, MAX_SHORTLIST + 1])
def test_a_limit_out_of_range_is_refused(
    client: TestClient, trained: Path, pool: Pool, limit: int
) -> None:
    job, _ = pool
    assert client.get(f"/jobs/{job['id']}/report.pdf", params={"limit": limit}).status_code == 422


def test_a_shortlist_of_an_unscored_job_is_a_conflict_naming_the_fix(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool

    response = client.get(f"/jobs/{job['id']}/report.pdf")

    assert response.status_code == 409
    assert f"POST /api/v1/jobs/{job['id']}/score" in response.json()["detail"]


def test_scores_from_another_matcher_do_not_make_a_report(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    """Rows a retrained matcher left behind are not mixed in, as the listing leaves them out."""
    job, candidates = pool
    _scored(client, job["id"])
    with Session(clean_db) as db:
        db.execute(update(MatchScore).values(model_version="0000000000000000"))
        db.commit()

    assert client.get(f"/jobs/{job['id']}/report.pdf").status_code == 409
    scorecard = client.get(f"/jobs/{job['id']}/candidates/{candidates[0]['id']}/scorecard.pdf")
    assert scorecard.status_code == 409


def _before_the_ranking_is_read(action: Callable[[], None]) -> Callable[..., None]:
    """A cursor hook that runs ``action`` once, just before the shortlist's ranked rows are read."""
    fired = []

    def hook(conn, cursor, statement, parameters, context, executemany) -> None:
        if not fired and "FROM match_scores" in statement and "ORDER BY" in statement:
            fired.append(True)
            action()

    return hook


def test_a_score_committed_while_the_shortlist_is_read_is_counted_in_its_total(
    client: TestClient, trained: Path, pool: Pool, make_candidate: Factory, clean_db: Engine
) -> None:
    """The total and the rows come from one snapshot, so the report never lists more than it
    says there are."""
    job, _ = pool
    _scored(client, job["id"])
    listed = client.get(f"/jobs/{job['id']}/matches").json()
    late = make_candidate()

    def commit_a_score() -> None:
        with clean_db.begin() as conn:
            conn.execute(
                insert(MatchScore).values(
                    job_id=job["id"],
                    candidate_id=late["id"],
                    model_version=listed["model_version"],
                    score=0.999,
                    rank=1,
                    features=listed["items"][0]["features"],
                    scored_at=datetime.now(UTC),
                )
            )

    hook = _before_the_ranking_is_read(commit_a_score)
    event.listen(Engine, "before_cursor_execute", hook)
    try:
        text = _text(client.get(f"/jobs/{job['id']}/report.pdf"))
    finally:
        event.remove(Engine, "before_cursor_execute", hook)

    assert "top 7 of 7" in text


def test_scores_deleted_while_the_shortlist_is_read_are_a_conflict_not_a_server_error(
    client: TestClient, trained: Path, pool: Pool, clean_db: Engine
) -> None:
    job, _ = pool
    _scored(client, job["id"])

    def delete_the_scores() -> None:
        with clean_db.begin() as conn:
            conn.execute(delete(MatchScore).where(MatchScore.job_id == job["id"]))

    hook = _before_the_ranking_is_read(delete_the_scores)
    event.listen(Engine, "before_cursor_execute", hook)
    try:
        response = client.get(f"/jobs/{job['id']}/report.pdf")
    finally:
        event.remove(Engine, "before_cursor_execute", hook)

    assert response.status_code == 409


def test_both_reports_need_the_matcher_and_say_how_to_regenerate_it(
    client: TestClient, pool: Pool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job, candidates = pool
    monkeypatch.setenv("RECRUITFLOW_MODEL_PATH", str(tmp_path / "missing.joblib"))
    get_settings.cache_clear()
    try:
        responses = [
            client.get(f"/jobs/{job['id']}/report.pdf"),
            client.get(f"/jobs/{job['id']}/candidates/{candidates[0]['id']}/scorecard.pdf"),
        ]
    finally:
        get_settings.cache_clear()

    for response in responses:
        assert response.status_code == 503
        assert response.json()["code"] == "matcher_unavailable"
        assert "scripts/train_matcher.py" in response.json()["detail"]


def test_a_shortlist_of_a_missing_job_is_not_found(client: TestClient, trained: Path) -> None:
    response = client.get("/jobs/999999/report.pdf")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job 999999 not found"


# --- scorecard -------------------------------------------------------------


def test_each_scorecard_gives_the_rank_and_score_the_listing_gives(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, _ = pool
    _scored(client, job["id"])
    listed = client.get(f"/jobs/{job['id']}/matches").json()["items"]

    for row in listed:
        candidate_id = row["candidate"]["id"]
        response = client.get(f"/jobs/{job['id']}/candidates/{candidate_id}/scorecard.pdf")
        text = _text(response)

        assert f"Rank {row['rank']} of {len(listed)}" in text
        assert f"score {row['score']:.3f}" in text
        assert f"Scorecard: {row['candidate']['full_name']}" in text
        assert response.headers["content-disposition"] == (
            f'inline; filename="job-{job["id"]}-candidate-{candidate_id}-scorecard.pdf"'
        )


def test_tied_candidates_take_the_ranks_the_listing_gives_them(
    client: TestClient, trained: Path, make_job: Factory, make_candidate: Factory, corpus
) -> None:
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
    scored = _scored(client, job["id"])
    assert scored[0]["score"] == scored[1]["score"]

    for rank, candidate in enumerate((first, second), start=1):
        url = f"/jobs/{job['id']}/candidates/{candidate['id']}/scorecard.pdf"
        assert f"Rank {rank} of 2" in _text(client.get(url))


def test_a_scorecard_says_when_the_candidate_changed_after_scoring(
    client: TestClient, trained: Path, pool: Pool
) -> None:
    job, candidates = pool
    _scored(client, job["id"])
    url = f"/jobs/{job['id']}/candidates/{candidates[0]['id']}/scorecard.pdf"
    assert "changed after this score" not in _text(client.get(url))

    edited = client.patch(f"/candidates/{candidates[0]['id']}", json={"skills": ["COBOL"]})
    assert edited.status_code == 200, edited.text

    text = _text(client.get(url))
    assert "The candidate changed after this score was taken" in text
    assert re.search(r"Lists 0 of \d+ required", text)


def test_a_candidate_with_no_score_for_the_job_is_a_conflict(
    client: TestClient, trained: Path, pool: Pool, make_candidate: Factory
) -> None:
    job, _ = pool
    _scored(client, job["id"])
    newcomer = make_candidate()

    response = client.get(f"/jobs/{job['id']}/candidates/{newcomer['id']}/scorecard.pdf")

    assert response.status_code == 409
    assert (
        f"Candidate {newcomer['id']} has no score for job {job['id']}"
        in (response.json()["detail"])
    )


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        ("/jobs/999999/candidates/{candidate}/scorecard.pdf", "Job 999999 not found"),
        ("/jobs/{job}/candidates/999999/scorecard.pdf", "Candidate 999999 not found"),
    ],
)
def test_a_scorecard_for_a_missing_job_or_candidate_is_not_found(
    client: TestClient, trained: Path, pool: Pool, path: str, detail: str
) -> None:
    job, candidates = pool
    _scored(client, job["id"])

    response = client.get(path.format(job=job["id"], candidate=candidates[0]["id"]))

    assert response.status_code == 404
    assert response.json()["detail"] == detail


def test_names_with_markup_and_letters_outside_the_font_still_make_a_report(
    client: TestClient, trained: Path, make_job: Factory, make_candidate: Factory, corpus
) -> None:
    job = make_job(**(corpus.jobs[0].api_payload() | {"title": "R&D <Lead>"}))
    candidate = make_candidate(
        **(corpus.candidates[0].api_payload() | {"full_name": "Łukasz <b>Šimić"})
    )
    _scored(client, job["id"])

    shortlist = _text(client.get(f"/jobs/{job['id']}/report.pdf"))
    scorecard = _text(client.get(f"/jobs/{job['id']}/candidates/{candidate['id']}/scorecard.pdf"))

    assert "Shortlist: R&D <Lead>" in shortlist
    assert "?ukasz <b>Šimic" in shortlist
    assert "Scorecard: ?ukasz <b>Šimic" in scorecard


# --- sample script ---------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample_reports.py"


def _load_script() -> ModuleType:
    """Not on the import path; load it from its file, as the other script tests do."""
    spec = importlib.util.spec_from_file_location("sample_reports_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sample_script_writes_both_reports_through_the_api(
    client: TestClient, trained: Path, tmp_path: Path
) -> None:
    written = _load_script().run(client, tmp_path / "samples", limit=5)

    assert [path.name for path in written] == ["shortlist.pdf", "scorecard.pdf"]
    shortlist, scorecard = (
        "\n".join(p.extract_text() for p in PdfReader(path).pages) for path in written
    )
    assert "top 5 of 50" in shortlist
    assert "Rank 1 of 50" in scorecard


def test_the_sample_script_reports_a_refusal_instead_of_writing_a_file(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No matcher where the app looks: scoring is a 503, and nothing is written."""
    monkeypatch.setenv("RECRUITFLOW_MODEL_PATH", str(tmp_path / "missing.joblib"))
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="gave 503"):
            _load_script().run(client, tmp_path / "samples", limit=5)
    finally:
        get_settings.cache_clear()
    assert not (tmp_path / "samples").exists()


@pytest.mark.parametrize("limit", ["0", str(MAX_SHORTLIST + 1)])
def test_the_sample_script_refuses_a_limit_before_creating_anything(
    limit: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checked up front: the report endpoint would refuse it only after the job, its slate
    and their scores had been written. Nothing listens on the port, so a request would fail
    differently."""
    status = _load_script().main(
        ["--base-url", "http://127.0.0.1:9/api/v1", "--out", str(tmp_path), "--limit", limit]
    )

    assert status == 2
    assert f"--limit must be 1 to {MAX_SHORTLIST}" in capsys.readouterr().err
