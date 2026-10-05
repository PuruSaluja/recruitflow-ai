"""Encoding candidates and jobs in batches, and caching the vectors in PostgreSQL.

:func:`refresh_embeddings` brings one encoder's stored vectors up to date for
the candidates or the jobs; :func:`load_embeddings` reads them back as one
matrix. A stored vector is current when it was made by the same encoder (its
name pins the weights) from the same text: each row keeps the SHA-256 of the
document it encoded, so a refresh re-encodes only owners that are new, or
whose document has changed since, without running the model to find out.

**A refresh commits as it goes.** It is a batch job, not part of a request:
each batch of ``batch_size`` owners is encoded with no transaction open, then
written and committed in a short transaction of its own. A long refresh
therefore never holds locks while the model runs, and an interrupted one keeps
the batches it finished. Pass a session with nothing pending.

**How a write takes turns with other writes.** Each batch's transaction first
locks its owners' rows ``FOR KEY SHARE``, in id order, then upserts their
vectors in id order:

* An edit of a candidate or job (``FOR NO KEY UPDATE``) does not conflict with
  that lock, so edits never wait for a refresh, nor a refresh for them. A
  vector written for text that an edit has since replaced carries the old
  text's hash, so the next refresh sees it as stale.
* A delete's final ``DELETE`` (``FOR UPDATE``) does conflict: it waits for the
  batch's commit, then takes the new vectors with it by ``ON DELETE CASCADE``.
  An owner deleted before the batch locks it is not returned by the lock, and
  is counted as vanished rather than failing the batch's foreign key check.
* Two refreshes of the same owners both get the lock; the second's upsert of
  each row waits for the first's commit and then updates it. Both go in id
  order, so neither can hold a row the other needs while waiting for one it
  holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import ARRAY, Integer, any_, bindparam, select, text
from sqlalchemy.orm import Session

from app.ml.embeddings import stack, to_bytes
from app.ml.encoder import Encoder, candidate_document, job_document
from app.models.candidate import Candidate
from app.models.embedding import ENCODER_MAX_LENGTH, CandidateEmbedding, JobEmbedding
from app.models.job import Job

#: Owners encoded and written per batch: one ``encode`` call, one transaction.
DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class EmbeddingKind:
    """What is being embedded: candidates or jobs."""

    owner: Any
    embedding: Any
    owner_column: str
    document: Callable[[Any], str]

    @property
    def table(self) -> str:
        return self.embedding.__tablename__


CANDIDATES = EmbeddingKind(Candidate, CandidateEmbedding, "candidate_id", candidate_document)
JOBS = EmbeddingKind(Job, JobEmbedding, "job_id", job_document)


@dataclass(frozen=True)
class RefreshReport:
    """What a refresh did, owner by owner."""

    #: Encoded and written by this refresh.
    encoded: int
    #: Already stored for this encoder and this exact text; not re-encoded.
    current: int
    #: Deleted after being read and before their batch was written.
    vanished: int
    #: Asked for by id but not found at all.
    not_found: int


def text_sha256(document: str) -> str:
    """The hash stored beside a vector: lowercase hex SHA-256 of the UTF-8 document."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _ids_param(name: str, ids: Sequence[int]) -> Any:
    # One array parameter, not one per id (psycopg refuses more than 65,535).
    return bindparam(name, list(ids), type_=ARRAY(Integer))


def _upsert(kind: EmbeddingKind) -> Any:
    owner = kind.owner_column
    return text(
        f"INSERT INTO {kind.table} ({owner}, encoder, dimensions, text_sha256, vector, encoded_at) "
        f"SELECT batch.owner_id, :encoder, :dimensions, batch.sha, batch.vector, :encoded_at "
        "FROM unnest("
        "CAST(:owner_ids AS integer[]), CAST(:shas AS text[]), CAST(:vectors AS bytea[])"
        ") AS batch(owner_id, sha, vector) "
        "ORDER BY batch.owner_id "
        f"ON CONFLICT ON CONSTRAINT uq_{kind.table}_owner_encoder DO UPDATE SET "
        "dimensions = excluded.dimensions, text_sha256 = excluded.text_sha256, "
        "vector = excluded.vector, encoded_at = excluded.encoded_at"
    )


def _check_encoder(encoder: Encoder) -> None:
    if not encoder.name or len(encoder.name) > ENCODER_MAX_LENGTH:
        raise ValueError(
            f"an encoder name must be 1 to {ENCODER_MAX_LENGTH} characters, got {len(encoder.name)}"
        )
    if encoder.dimensions < 1:
        raise ValueError(f"an encoder must have at least one dimension, got {encoder.dimensions}")


def _pending(
    db: Session, kind: EmbeddingKind, encoder: Encoder, owner_ids: Sequence[int] | None
) -> tuple[list[tuple[int, str, str]], int, int]:
    """Owners whose vector is missing or stale, as (id, document, hash) in id order."""
    owner_id = kind.owner.id
    owners = select(kind.owner).order_by(owner_id)
    stored = select(getattr(kind.embedding, kind.owner_column), kind.embedding.text_sha256).where(
        kind.embedding.encoder == encoder.name
    )
    if owner_ids is not None:
        owners = owners.where(owner_id == any_(_ids_param("owner_ids", owner_ids)))
        stored = stored.where(
            getattr(kind.embedding, kind.owner_column) == any_(_ids_param("stored_ids", owner_ids))
        )
    found = list(db.scalars(owners))
    hashes = dict(db.execute(stored).all())

    pending = []
    current = 0
    for owner in found:
        document = kind.document(owner)
        sha = text_sha256(document)
        if hashes.get(owner.id) == sha:
            current += 1
        else:
            pending.append((owner.id, document, sha))
    not_found = 0 if owner_ids is None else len(set(owner_ids)) - len(found)
    return pending, current, not_found


def refresh_embeddings(
    db: Session,
    kind: EmbeddingKind,
    encoder: Encoder,
    owner_ids: Sequence[int] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RefreshReport:
    """Encode and store every owner of ``kind`` whose vector is missing or stale.

    ``owner_ids`` limits the refresh to those owners (repeats are ignored);
    by default it covers every candidate or job. Owners are encoded in id
    order, ``batch_size`` at a time, and each batch is committed before the
    next is encoded (see the module docstring for how the writes take turns
    with edits, deletes and other refreshes).

    The encoder's output is checked before anything is written: one row of
    ``encoder.dimensions`` values per document, each storable by
    :func:`app.ml.embeddings.to_bytes` (finite and not all zero). A batch
    that fails the check raises ``ValueError``, and the batches before it stay
    committed.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    _check_encoder(encoder)

    pending, current, not_found = _pending(db, kind, encoder, owner_ids)
    # End the read before the model runs: no snapshot or transaction is held
    # open for the length of the encoding.
    db.rollback()

    encoded = vanished = 0
    upsert = _upsert(kind)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = np.asarray(encoder.encode([document for _, document, _ in batch]))
        if vectors.shape != (len(batch), encoder.dimensions):
            raise ValueError(
                f"{encoder.name} returned shape {vectors.shape} for {len(batch)} texts, "
                f"expected ({len(batch)}, {encoder.dimensions})"
            )
        blobs = {
            owner_id: to_bytes(vector)
            for (owner_id, _, _), vector in zip(batch, vectors, strict=True)
        }
        shas = {owner_id: sha for owner_id, _, sha in batch}

        locked = list(
            db.scalars(
                select(kind.owner.id)
                .where(kind.owner.id == any_(_ids_param("lock_ids", list(blobs))))
                .order_by(kind.owner.id)
                # read=True: key_share alone renders FOR NO KEY UPDATE, an edit's lock.
                .with_for_update(read=True, key_share=True)
            )
        )
        if locked:
            db.execute(
                upsert,
                {
                    "encoder": encoder.name,
                    "dimensions": encoder.dimensions,
                    "encoded_at": datetime.now(UTC),
                    "owner_ids": locked,
                    "shas": [shas[owner_id] for owner_id in locked],
                    "vectors": [blobs[owner_id] for owner_id in locked],
                },
            )
        db.commit()
        encoded += len(locked)
        vanished += len(batch) - len(locked)

    return RefreshReport(encoded=encoded, current=current, vanished=vanished, not_found=not_found)


def load_embeddings(
    db: Session, kind: EmbeddingKind, encoder: Encoder, owner_ids: Sequence[int]
) -> tuple[list[int], np.ndarray]:
    """The stored vectors of ``owner_ids`` for ``encoder``: (ids, matrix), in id order.

    Owners with no vector for this encoder are left out, so compare the ids
    returned with those asked for. Whether a vector is still current is the
    refresh's question, not this one's: call :func:`refresh_embeddings` first.
    A stored vector whose dimensions differ from the encoder's is refused.
    """
    owner = getattr(kind.embedding, kind.owner_column)
    rows = db.execute(
        select(owner, kind.embedding.dimensions, kind.embedding.vector)
        .where(kind.embedding.encoder == encoder.name)
        .where(owner == any_(_ids_param("owner_ids", sorted(set(owner_ids)))))
        .order_by(owner)
    ).all()
    wrong = [row[0] for row in rows if row[1] != encoder.dimensions]
    if wrong:
        raise ValueError(
            f"stored vectors for {wrong[:5]} do not have {encoder.name}'s "
            f"{encoder.dimensions} dimensions"
        )
    return [row[0] for row in rows], stack((row[2] for row in rows), encoder.dimensions)
