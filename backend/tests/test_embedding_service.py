"""Tests for app.services.embeddings: the batched, cached embedding writer.

A deterministic stand-in encoder makes each vector a function of its text, so
a test can say exactly which vector should be stored. The real model is tested
in test_encoder.py.

The race cases hold one write open on its own connection, start the other from
a thread, and let the first commit once the second is seen waiting for a lock,
or after ten seconds, so a case can fail but never hang.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

import numpy as np
import pytest
from fake_encoder import FakeEncoder, fake_vector
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from app.ml.encoder import candidate_document, job_document
from app.models.candidate import Candidate
from app.models.embedding import CandidateEmbedding
from app.models.job import Job
from app.schemas.common import MAX_ID
from app.services import embeddings
from app.services.embeddings import (
    CANDIDATES,
    JOBS,
    RefreshReport,
    load_embeddings,
    refresh_embeddings,
    text_sha256,
)

Factory = Callable[..., dict[str, Any]]

LOCK_WAITS = text(
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND backend_type = 'client backend' AND wait_event_type = 'Lock'"
)


def _waiting_for_a_lock(engine: Engine) -> bool:
    with engine.connect() as watcher:
        return watcher.scalar(LOCK_WAITS) > 0


def _wait_for_a_lock_wait(engine: Engine) -> bool:
    deadline = time.monotonic() + 10
    while not _waiting_for_a_lock(engine) and time.monotonic() < deadline:
        time.sleep(0.01)
    return _waiting_for_a_lock(engine)


@pytest.fixture
def candidates(clean_db: Engine, make_candidate: Factory) -> list[int]:
    resumes = [
        "Built reporting pipelines in SQL and Python for a retail team.",
        "Registered nurse with ICU experience.",
        "Frontend engineer, React and TypeScript.",
        "Data analyst: Tableau dashboards and weekly metrics.",
        "Backend engineer working in Go and PostgreSQL.",
    ]
    return [make_candidate(resume_text=resume)["id"] for resume in resumes]


def _stored(engine: Engine, encoder: str = "fake-encoder@1") -> dict[int, tuple]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT candidate_id, id, text_sha256, vector, encoded_at "
                "FROM candidate_embeddings WHERE encoder = :encoder"
            ),
            {"encoder": encoder},
        ).all()
    return {row[0]: tuple(row[1:]) for row in rows}


def _documents(engine: Engine, ids: Sequence[int]) -> dict[int, str]:
    with Session(engine) as db:
        rows = db.scalars(select(Candidate).where(Candidate.id.in_(ids)))
        return {row.id: candidate_document(row) for row in rows}


# --- refresh: what gets encoded --------------------------------------------


def test_a_first_refresh_encodes_and_stores_every_candidate(
    clean_db: Engine, candidates: list[int]
) -> None:
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder)

    assert report == RefreshReport(encoded=5, current=0, vanished=0, not_found=0)
    documents = _documents(clean_db, candidates)
    assert encoder.calls == [[documents[i] for i in candidates]]
    stored = _stored(clean_db)
    assert sorted(stored) == candidates
    for candidate_id, (_, sha, vector, _) in stored.items():
        assert sha == text_sha256(documents[candidate_id])
        assert bytes(vector) == fake_vector(documents[candidate_id], 8).astype("<f4").tobytes()


def test_a_second_refresh_encodes_nothing(clean_db: Engine, candidates: list[int]) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    before = _stored(clean_db)
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder)

    assert report == RefreshReport(encoded=0, current=5, vanished=0, not_found=0)
    assert encoder.calls == []
    assert _stored(clean_db) == before


def test_an_edited_resume_is_re_encoded_in_place(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    before = _stored(clean_db)
    edited = candidates[2]
    response = client.patch(f"/candidates/{edited}", json={"resume_text": "Now a Rust engineer."})
    assert response.status_code == 200, response.text

    encoder = FakeEncoder()
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder)

    assert report == RefreshReport(encoded=1, current=4, vanished=0, not_found=0)
    document = _documents(clean_db, [edited])[edited]
    assert "now a rust engineer." in document
    assert encoder.calls == [[document]]
    after = _stored(clean_db)
    row_id, sha, vector, encoded_at = after[edited]
    assert row_id == before[edited][0]  # updated, not a second row
    assert sha == text_sha256(document)
    assert bytes(vector) == fake_vector(document, 8).astype("<f4").tobytes()
    assert encoded_at > before[edited][3]
    assert {k: v for k, v in after.items() if k != edited} == {
        k: v for k, v in before.items() if k != edited
    }


def test_an_edit_outside_the_document_changes_nothing(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    response = client.patch(f"/candidates/{candidates[0]}", json={"phone": "555-0100"})
    assert response.status_code == 200, response.text

    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder())
    assert report.encoded == 0 and report.current == 5


def test_a_new_encoder_encodes_everything_and_keeps_the_old_rows(
    clean_db: Engine, candidates: list[int]
) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    old = _stored(clean_db)
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder(name="fake-encoder@2"))

    assert report.encoded == 5
    assert _stored(clean_db) == old
    assert sorted(_stored(clean_db, "fake-encoder@2")) == candidates


def test_batches_go_in_id_order_and_respect_the_batch_size(
    clean_db: Engine, candidates: list[int]
) -> None:
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, encoder, batch_size=2)
    documents = _documents(clean_db, candidates)
    assert encoder.calls == [
        [documents[candidates[0]], documents[candidates[1]]],
        [documents[candidates[2]], documents[candidates[3]]],
        [documents[candidates[4]]],
    ]


def test_each_batch_is_committed_before_the_next_is_encoded(
    clean_db: Engine, candidates: list[int]
) -> None:
    seen: list[int] = []
    encoder = FakeEncoder(on_encode=lambda call: seen.append(len(_stored(clean_db))))
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, encoder, batch_size=2)
        assert not db.in_transaction()
    assert seen == [0, 2, 4]


def test_a_refresh_can_be_limited_to_some_owners(clean_db: Engine, candidates: list[int]) -> None:
    encoder = FakeEncoder()
    wanted = [candidates[3], candidates[1], candidates[3], 999_999]
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder, wanted)

    assert report == RefreshReport(encoded=2, current=0, vanished=0, not_found=1)
    assert sorted(_stored(clean_db)) == [candidates[1], candidates[3]]
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, encoder, [candidates[1]]) == RefreshReport(
            encoded=0, current=1, vanished=0, not_found=0
        )
        assert refresh_embeddings(db, CANDIDATES, encoder, []) == RefreshReport(0, 0, 0, 0)


@pytest.mark.parametrize("impossible", [2**31, 2**63, 0, -1])
def test_an_id_no_row_can_have_is_not_found_rather_than_an_error(
    clean_db: Engine, candidates: list[int], impossible: int
) -> None:
    """2**31 used to fail the whole statement with a DataError (integer out of range)."""
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder, [impossible, candidates[0]])
        ids, _ = load_embeddings(db, CANDIDATES, encoder, [impossible, candidates[0]])
    assert report == RefreshReport(encoded=1, current=0, vanished=0, not_found=1)
    assert ids == [candidates[0]]


def test_the_largest_id_a_row_can_have_is_still_refreshed(clean_db: Engine) -> None:
    """Guards against over-correcting: the bound must admit every id a row can have."""
    with clean_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO candidates (id, full_name, email, resume_text, skills, "
                "years_experience, seniority, education_level) VALUES (:id, 'Max Id', "
                "'max@example.com', 'Built reporting pipelines in SQL.', '[]', 3, 'mid', "
                "'bachelor')"
            ),
            {"id": MAX_ID},
        )
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, encoder, [MAX_ID]).encoded == 1
        assert load_embeddings(db, CANDIDATES, encoder, [MAX_ID])[0] == [MAX_ID]


def test_jobs_are_embedded_from_the_job_document(clean_db: Engine, make_job: Factory) -> None:
    ids = [
        make_job(title="Data Analyst", required_skills=["SQL"])["id"],
        make_job(title="\uff30latform Engineer", description="Run the Kubernetes platform.")["id"],
    ]
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        assert refresh_embeddings(db, JOBS, encoder).encoded == 2
        documents = [job_document(db.get(Job, i)) for i in ids]
        assert documents[1].startswith("platform engineer\n")
        assert encoder.calls == [documents]
        found, matrix = load_embeddings(db, JOBS, encoder, ids)
    assert found == ids
    assert np.array_equal(matrix, np.stack([fake_vector(d, 8) for d in documents]))


def test_nothing_to_do_still_ends_the_read_transaction(clean_db: Engine) -> None:
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, FakeEncoder()) == RefreshReport(0, 0, 0, 0)
        assert not db.in_transaction()


# --- refresh: what is refused ----------------------------------------------


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda n: np.ones((n, 7), np.float32), r"returned shape \(2, 7\) for 2 texts"),
        (lambda n: np.ones((n + 1, 8), np.float32), r"returned shape \(3, 8\) for 2 texts"),
        (lambda n: np.ones(8, np.float32), r"returned shape \(8,\)"),
        (lambda n: np.full((n, 8), np.nan, np.float32), "finite"),
        (lambda n: np.zeros((n, 8), np.float32), "no direction"),
    ],
    ids=["too-narrow", "too-many-rows", "one-dimensional", "nan", "zero"],
)
def test_bad_encoder_output_fails_its_batch_and_keeps_the_ones_before(
    clean_db: Engine, candidates: list[int], output: Callable[[int], np.ndarray], message: str
) -> None:
    class BreaksOnSecondBatch(FakeEncoder):
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            good = super().encode(texts)
            return output(len(texts)) if len(self.calls) == 2 else good

    with Session(clean_db) as db, pytest.raises(ValueError, match=message):
        refresh_embeddings(db, CANDIDATES, BreaksOnSecondBatch(), batch_size=2)
    assert sorted(_stored(clean_db)) == candidates[:2]


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (FakeEncoder(name=""), "1 to 200 characters"),
        (FakeEncoder(name="x" * 201), "1 to 200 characters, got 201"),
        (FakeEncoder(dimensions=0), "at least one dimension"),
    ],
    ids=["empty-name", "long-name", "no-dimensions"],
)
def test_an_unusable_encoder_is_refused_before_reading(
    clean_db: Engine, encoder: FakeEncoder, message: str
) -> None:
    with Session(clean_db) as db, pytest.raises(ValueError, match=message):
        refresh_embeddings(db, CANDIDATES, encoder)


def test_the_longest_encoder_name_the_table_holds_is_accepted(
    clean_db: Engine, candidates: list[int]
) -> None:
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, FakeEncoder(name="x" * 200)).encoded == 5


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(clean_db: Engine, batch_size: int) -> None:
    with Session(clean_db) as db, pytest.raises(ValueError, match="at least 1"):
        refresh_embeddings(db, CANDIDATES, FakeEncoder(), batch_size=batch_size)


# --- load ------------------------------------------------------------------


def test_load_returns_stored_vectors_in_id_order_and_skips_missing(
    clean_db: Engine, candidates: list[int]
) -> None:
    encoder = FakeEncoder()
    with Session(clean_db) as db:
        # The last candidate first, so the rows' order on disk is not id order.
        refresh_embeddings(db, CANDIDATES, encoder, [candidates[4]])
        refresh_embeddings(db, CANDIDATES, encoder, candidates[1:4])
        ids, matrix = load_embeddings(
            db, CANDIDATES, encoder, [candidates[4], candidates[0], candidates[2], candidates[4]]
        )
    documents = _documents(clean_db, candidates)
    assert ids == [candidates[2], candidates[4]]
    assert np.array_equal(matrix, np.stack([fake_vector(documents[i], 8) for i in ids]))


def test_load_reads_only_the_named_encoder(clean_db: Engine, candidates: list[int]) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
        ids, matrix = load_embeddings(db, CANDIDATES, FakeEncoder(name="other@1"), candidates)
    assert ids == []
    assert matrix.shape == (0, 8)


def test_load_refuses_vectors_of_another_width(clean_db: Engine, candidates: list[int]) -> None:
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
        with pytest.raises(ValueError, match="do not have fake-encoder@1's 4 dimensions"):
            load_embeddings(db, CANDIDATES, FakeEncoder(dimensions=4), candidates)


# --- races -----------------------------------------------------------------


def _pause_before_first_commit(db: Session, during: Callable[[], None]) -> None:
    """Run ``during`` while the refresh's first batch is written but not committed."""
    real = db.commit
    paused = False

    def commit() -> None:
        nonlocal paused
        if not paused:
            paused = True
            during()
        real()

    db.commit = commit  # type: ignore[method-assign]


def test_a_delete_waits_for_an_uncommitted_batch_then_takes_its_vectors(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    doomed = candidates[1]
    pool = ThreadPoolExecutor(max_workers=1)
    pending: dict[str, Any] = {}

    def delete_while_paused() -> None:
        pending["response"] = pool.submit(client.delete, f"/candidates/{doomed}")
        pending["waited"] = _wait_for_a_lock_wait(clean_db)

    with Session(clean_db) as db:
        _pause_before_first_commit(db, delete_while_paused)
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder())
    response = pending["response"].result(timeout=30)
    pool.shutdown()

    assert pending["waited"], "the delete never waited for the refresh's batch"
    assert response.status_code == 204, response.text
    assert report.encoded == 5
    assert sorted(_stored(clean_db)) == [i for i in candidates if i != doomed]


def test_an_owner_deleted_while_its_batch_is_encoded_is_skipped(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    doomed = candidates[3]

    def delete_during_encode(call: int) -> None:
        if call == 2:
            assert client.delete(f"/candidates/{doomed}").status_code == 204

    with Session(clean_db) as db:
        report = refresh_embeddings(
            db, CANDIDATES, FakeEncoder(on_encode=delete_during_encode), batch_size=2
        )

    assert report == RefreshReport(encoded=4, current=0, vanished=1, not_found=0)
    assert sorted(_stored(clean_db)) == [i for i in candidates if i != doomed]


def test_a_batch_whose_every_owner_vanished_writes_nothing_and_carries_on(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    doomed = candidates[2]

    def delete_during_its_own_batch(call: int) -> None:
        if call == 3:  # batches of one: the third holds only the doomed candidate
            assert client.delete(f"/candidates/{doomed}").status_code == 204

    encoder = FakeEncoder(on_encode=delete_during_its_own_batch)
    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, encoder, batch_size=1)
        assert not db.in_transaction()

    assert len(encoder.calls) == 5
    assert report == RefreshReport(encoded=4, current=0, vanished=1, not_found=0)
    assert sorted(_stored(clean_db)) == [i for i in candidates if i != doomed]


def test_a_batch_waits_for_an_uncommitted_delete_then_skips_its_owner(
    clean_db: Engine, candidates: list[int]
) -> None:
    doomed = candidates[0]
    deleted = threading.Event()
    seen: dict[str, bool] = {}

    def hold_a_delete_open() -> None:
        with clean_db.connect() as other, other.begin():
            other.execute(text("DELETE FROM candidates WHERE id = :id"), {"id": doomed})
            deleted.set()
            seen["waited"] = _wait_for_a_lock_wait(clean_db)

    holder = threading.Thread(target=hold_a_delete_open)

    def start_the_delete(call: int) -> None:
        holder.start()
        assert deleted.wait(10)

    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder(on_encode=start_the_delete))
    holder.join(30)

    assert seen["waited"], "the batch never waited for the delete"
    assert report == RefreshReport(encoded=4, current=0, vanished=1, not_found=0)
    assert sorted(_stored(clean_db)) == candidates[1:]


def test_a_resume_edit_does_not_wait_for_a_batch_and_leaves_its_vector_stale(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    edited = candidates[2]
    outcome: dict[str, Any] = {}

    pool = ThreadPoolExecutor(max_workers=1)

    def edit_while_paused() -> None:
        sent = pool.submit(
            client.patch,
            f"/candidates/{edited}",
            json={"resume_text": "Changed meanwhile to a Rust role."},
        )
        try:
            outcome["status"] = sent.result(timeout=10).status_code
        except FutureTimeout:  # only if the edit waits for the batch
            # Not waiting on the pool here: the commit that releases the edit
            # runs once this returns.
            outcome["status"] = "waited for the batch"

    with Session(clean_db) as db:
        _pause_before_first_commit(db, edit_while_paused)
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    pool.shutdown()

    assert outcome["status"] == 200
    # The vector written is for the text read before the edit, so it is stale.
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, FakeEncoder()) == RefreshReport(
            encoded=1, current=4, vanished=0, not_found=0
        )
    document = _documents(clean_db, [edited])[edited]
    assert _stored(clean_db)[edited][1] == text_sha256(document)


def test_an_email_edit_waits_for_the_batch_then_lands(
    client: TestClient, clean_db: Engine, candidates: list[int]
) -> None:
    """The email has a unique index, so PostgreSQL treats it as a key: changing it
    takes FOR UPDATE, which conflicts with the batch's FOR KEY SHARE."""
    edited = candidates[1]
    pool = ThreadPoolExecutor(max_workers=1)
    pending: dict[str, Any] = {}

    def edit_while_paused() -> None:
        pending["response"] = pool.submit(
            client.patch, f"/candidates/{edited}", json={"email": "moved@example.com"}
        )
        pending["waited"] = _wait_for_a_lock_wait(clean_db)

    with Session(clean_db) as db:
        _pause_before_first_commit(db, edit_while_paused)
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder())
    response = pending["response"].result(timeout=30)
    pool.shutdown()

    assert pending["waited"], "the email edit never waited for the batch"
    assert response.status_code == 200, response.text
    assert response.json()["email"] == "moved@example.com"
    assert report.encoded == 5
    # The email is not part of the encoded document, so nothing went stale.
    with Session(clean_db) as db:
        assert refresh_embeddings(db, CANDIDATES, FakeEncoder()).current == 5


def test_a_batch_waits_for_an_uncommitted_email_change_then_writes(
    clean_db: Engine, candidates: list[int]
) -> None:
    """The batch then locks the updated row: the candidate is written, not counted
    as vanished."""
    edited = candidates[0]
    updated = threading.Event()
    seen: dict[str, bool] = {}

    def hold_an_email_change_open() -> None:
        with clean_db.connect() as other, other.begin():
            other.execute(
                text("UPDATE candidates SET email = 'moved@example.com' WHERE id = :id"),
                {"id": edited},
            )
            updated.set()
            seen["waited"] = _wait_for_a_lock_wait(clean_db)

    holder = threading.Thread(target=hold_an_email_change_open)

    def start_the_change(call: int) -> None:
        holder.start()
        assert updated.wait(10)

    with Session(clean_db) as db:
        report = refresh_embeddings(db, CANDIDATES, FakeEncoder(on_encode=start_the_change))
    holder.join(30)

    assert seen["waited"], "the batch never waited for the email change"
    assert report == RefreshReport(encoded=5, current=0, vanished=0, not_found=0)
    assert sorted(_stored(clean_db)) == candidates


def test_a_job_delete_waits_for_an_uncommitted_batch_then_takes_its_vectors(
    client: TestClient, clean_db: Engine, make_job: Factory
) -> None:
    jobs = [make_job(title=f"Data Analyst {n}")["id"] for n in range(3)]
    doomed = jobs[1]
    pool = ThreadPoolExecutor(max_workers=1)
    pending: dict[str, Any] = {}

    def delete_while_paused() -> None:
        pending["response"] = pool.submit(client.delete, f"/jobs/{doomed}")
        pending["waited"] = _wait_for_a_lock_wait(clean_db)

    with Session(clean_db) as db:
        _pause_before_first_commit(db, delete_while_paused)
        report = refresh_embeddings(db, JOBS, FakeEncoder())
    response = pending["response"].result(timeout=30)
    pool.shutdown()

    assert pending["waited"], "the job delete never waited for the refresh's batch"
    assert response.status_code == 204, response.text
    assert report.encoded == 3
    with clean_db.connect() as conn:
        left = conn.execute(text("SELECT job_id FROM job_embeddings ORDER BY job_id")).scalars()
        assert list(left) == [jobs[0], jobs[2]]


@pytest.mark.parametrize("batch_sizes", [(2, 2), (2, 3), (5, 1)])
def test_two_refreshes_of_the_same_owners_at_once_both_finish(
    clean_db: Engine, candidates: list[int], batch_sizes: tuple[int, int]
) -> None:
    """Both read before either writes, so both write every row; one row each survives."""
    both_read = threading.Barrier(2, timeout=10)

    def refresh(batch_size: int) -> RefreshReport:
        def wait_for_the_other(call: int) -> None:
            if call == 1:
                both_read.wait()

        with Session(clean_db) as db:
            return refresh_embeddings(
                db,
                CANDIDATES,
                FakeEncoder(on_encode=wait_for_the_other),
                batch_size=batch_size,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = [f.result(timeout=60) for f in [pool.submit(refresh, n) for n in batch_sizes]]

    assert [r.encoded for r in reports] == [5, 5]
    stored = _stored(clean_db)
    assert sorted(stored) == candidates
    with clean_db.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM candidate_embeddings")) == 5


def test_the_writer_locks_owners_for_key_share_only(
    clean_db: Engine, candidates: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the lock mode the module docstring's reasoning rests on."""
    statements: list[str] = []
    real = Session.scalars

    def recording(self: Session, statement: Any, *args: Any, **kwargs: Any) -> Any:
        statements.append(str(statement.compile(dialect=clean_db.dialect)))
        return real(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "scalars", recording)
    with Session(clean_db) as db:
        refresh_embeddings(db, CANDIDATES, FakeEncoder())
    locks = [s for s in statements if "FOR" in s.split("ORDER BY")[-1]]
    assert len(locks) == 1
    assert locks[0].rstrip().endswith("FOR KEY SHARE")
    assert "ORDER BY candidates.id" in locks[0]


def test_written_rows_go_in_id_order(clean_db: Engine) -> None:
    assert "ORDER BY batch.owner_id" in str(embeddings._upsert(CANDIDATES))


def test_the_model_classes_match_the_kinds() -> None:
    assert CANDIDATES.table == CandidateEmbedding.__tablename__ == "candidate_embeddings"
    assert JOBS.table == "job_embeddings"
