"""The candidate_embeddings and job_embeddings tables, on the real schema.

What the database itself enforces about a stored embedding: its length, its
hash, its key, and that it goes when its owner does. Each refusal is checked
by constraint name, since several constraints could refuse the same row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from app.ml.embeddings import from_bytes, to_bytes

ENCODER = "all-MiniLM-L6-v2@test"
TABLES = {"candidate_embeddings": "candidate_id", "job_embeddings": "job_id"}


def _sha(text_: str) -> str:
    return hashlib.sha256(text_.encode()).hexdigest()


@pytest.fixture
def owners(
    make_candidate: Callable[..., dict[str, Any]], make_job: Callable[..., dict[str, Any]]
) -> dict[str, int]:
    return {"candidate_embeddings": make_candidate()["id"], "job_embeddings": make_job()["id"]}


def _insert(engine: Engine, table: str, owner_id: int, **overrides: Any) -> None:
    row = {
        "owner": owner_id,
        "encoder": ENCODER,
        "dimensions": 3,
        "text_sha256": _sha("some text"),
        "vector": to_bytes([0.25, -1.5, 3.0]),
        **overrides,
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {table} ({TABLES[table]}, encoder, dimensions, text_sha256, vector) "
                "VALUES (:owner, :encoder, :dimensions, :text_sha256, :vector)"
            ),
            row,
        )


def _count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


@pytest.mark.parametrize("table", TABLES)
def test_a_stored_vector_reads_back_bit_for_bit(
    clean_db: Engine, owners: dict[str, int], table: str
) -> None:
    vector = np.random.default_rng(1).standard_normal(384).astype(np.float32)
    _insert(clean_db, table, owners[table], dimensions=384, vector=to_bytes(vector))
    with clean_db.connect() as conn:
        stored, dims, encoded_at = conn.execute(
            text(f"SELECT vector, dimensions, encoded_at FROM {table}")
        ).one()
    assert np.array_equal(from_bytes(stored, dims), vector)
    assert encoded_at is not None


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"vector": to_bytes([1.0, 2.0])}, "vector_length"),
        ({"vector": to_bytes([1.0, 2.0, 3.0, 4.0])}, "vector_length"),
        ({"vector": to_bytes([1.0, 2.0, 3.0])[:-1]}, "vector_length"),
        ({"vector": b""}, "vector_length"),
        ({"dimensions": 0, "vector": b""}, "dimensions_positive"),
        ({"dimensions": -1, "vector": b""}, "dimensions_positive"),
        ({"text_sha256": _sha("x").upper()}, "text_sha256_hex"),
        ({"text_sha256": _sha("x")[:63] + "g"}, "text_sha256_hex"),
        ({"text_sha256": _sha("x")[:63]}, "text_sha256_hex"),
        ({"encoder": ""}, "encoder_nonempty"),
    ],
    ids=[
        "too-short",
        "too-long",
        "partial-value",
        "empty",
        "zero-dims",
        "negative-dims",
        "uppercase-hash",
        "non-hex-hash",
        "short-hash",
        "empty-encoder",
    ],
)
def test_the_database_refuses_a_malformed_row(
    clean_db: Engine,
    owners: dict[str, int],
    table: str,
    overrides: dict[str, Any],
    constraint: str,
) -> None:
    with pytest.raises(IntegrityError, match=f'"ck_{table}_{constraint}"'):
        _insert(clean_db, table, owners[table], **overrides)
    assert _count(clean_db, table) == 0


@pytest.mark.parametrize("table", TABLES)
def test_one_row_per_owner_and_encoder(
    clean_db: Engine, owners: dict[str, int], table: str
) -> None:
    _insert(clean_db, table, owners[table])
    with pytest.raises(IntegrityError, match=f'"uq_{table}_owner_encoder"'):
        _insert(clean_db, table, owners[table], text_sha256=_sha("other text"))
    _insert(clean_db, table, owners[table], encoder="another-encoder@v2")
    assert _count(clean_db, table) == 2


@pytest.mark.parametrize("table", TABLES)
def test_an_embedding_needs_an_existing_owner(
    clean_db: Engine, owners: dict[str, int], table: str
) -> None:
    with pytest.raises(IntegrityError, match=f'"{table}_{TABLES[table]}_fkey"'):
        _insert(clean_db, table, owners[table] + 1000)


@pytest.mark.parametrize(
    ("table", "route"), [("candidate_embeddings", "/candidates"), ("job_embeddings", "/jobs")]
)
def test_deleting_an_owner_through_the_api_removes_only_its_embeddings(
    client: TestClient,
    clean_db: Engine,
    make_candidate: Callable[..., dict[str, Any]],
    make_job: Callable[..., dict[str, Any]],
    table: str,
    route: str,
) -> None:
    make = make_candidate if table == "candidate_embeddings" else make_job
    doomed, kept = make()["id"], make()["id"]
    for owner in (doomed, kept):
        _insert(clean_db, table, owner)
        _insert(clean_db, table, owner, encoder="another-encoder@v2")

    assert client.delete(f"{route}/{doomed}").status_code == 204

    with clean_db.connect() as conn:
        left = conn.execute(text(f"SELECT {TABLES[table]} FROM {table}")).scalars().all()
    assert left == [kept, kept]
