"""Stored text embeddings of candidates and jobs.

Kept as ``bytea`` of little-endian float32, the form the 2026-10-01 spike
chose over pgvector (``docs/embedding_storage.md``); :mod:`app.ml.embeddings`
writes and reads it. The database checks what it can of that form: the byte
length is four per dimension. It cannot check the byte order.

There is one row per owner *and encoder*, and the encoder's name carries its
version, the way ``match_scores`` carries a model version: after a change of
encoder the old rows are visibly another encoder's rather than silently mixed
with the new ones. ``text_sha256`` is the hash of the exact text that was
encoded, so an edited resume or job description shows up as a stale row
without re-encoding anything to find out.

There is no ORM relationship from ``Candidate`` or ``Job``: nothing loads
embeddings through them, and the foreign keys' ``ON DELETE CASCADE`` removes a
deleted owner's rows in the same statement that deletes the owner.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

#: Room for a Hub model id such as ``sentence-transformers/all-MiniLM-L6-v2``,
#: an ``@``, a 40-character revision hash and the chunking tag (the default
#: encoder's name is 89 characters).
ENCODER_MAX_LENGTH = 200


def _table_args(prefix: str, owner: str) -> tuple[UniqueConstraint | CheckConstraint, ...]:
    return (
        # Owner first, so its index also serves lookups by owner alone.
        UniqueConstraint(owner, "encoder", name=f"uq_{prefix}_owner_encoder"),
        CheckConstraint("dimensions > 0", name=f"ck_{prefix}_dimensions_positive"),
        CheckConstraint("octet_length(vector) = 4 * dimensions", name=f"ck_{prefix}_vector_length"),
        CheckConstraint("text_sha256 ~ '^[0-9a-f]{64}$'", name=f"ck_{prefix}_text_sha256_hex"),
        CheckConstraint("encoder <> ''", name=f"ck_{prefix}_encoder_nonempty"),
    )


class _EmbeddingColumns:
    """The columns both tables share; the owner's key is declared per table."""

    # An integer key like every other table's (test_regressions pins that).
    id: Mapped[int] = mapped_column(primary_key=True)
    #: Encoder name and version, e.g. ``all-MiniLM-L6-v2@<revision>``.
    encoder: Mapped[str] = mapped_column(String(ENCODER_MAX_LENGTH), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Lowercase hex SHA-256 of the UTF-8 text that was encoded.
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encoded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CandidateEmbedding(_EmbeddingColumns, Base):
    """One encoder's embedding of one candidate's matching text."""

    __tablename__ = "candidate_embeddings"
    __table_args__ = _table_args("candidate_embeddings", "candidate_id")

    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False
    )


class JobEmbedding(_EmbeddingColumns, Base):
    """One encoder's embedding of one job's matching text."""

    __tablename__ = "job_embeddings"
    __table_args__ = _table_args("job_embeddings", "job_id")

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
