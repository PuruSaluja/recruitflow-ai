"""Shared response envelopes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

#: The largest id any table can hold: every primary key is a PostgreSQL
#: ``INTEGER``. Every id the API accepts, in a path, a body or a query, is
#: bounded by it, because a larger one makes PostgreSQL raise "integer out of
#: range", which the database error handler would report as an outage.
MAX_ID = 2_147_483_647

#: The largest ``OFFSET`` PostgreSQL accepts (``BIGINT``), for the same reason.
MAX_OFFSET = 9_223_372_036_854_775_807


def contains_pattern(text: str) -> str:
    """Build a ``LIKE``/``ILIKE`` substring pattern from user-supplied text.

    ``%`` and ``_`` are wildcards, so an unescaped search box quietly lies:
    searching "100%" matches "1000 users" and a lone "_" matches everything.
    Callers must pair this with ``escape="\\\\"`` on the ilike() call.
    """
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def reject_null_fields(data: Any, fields: frozenset[str]) -> Any:
    """Reject an explicit ``null`` for a column the database declares NOT NULL.

    A PATCH body distinguishes "field omitted" from "field set to null", and
    for these fields the second is never meaningful. Letting it through is not
    merely untidy: on a JSONB column SQLAlchemy writes the JSON scalar ``null``
    rather than SQL NULL, so NOT NULL never fires, the row is stored unreadable,
    and every later response containing it fails to serialise.
    """
    if isinstance(data, dict):
        nulls = sorted(name for name in fields if name in data and data[name] is None)
        if nulls:
            raise ValueError(f"These fields cannot be set to null: {', '.join(nulls)}")
    return data


class Page(BaseModel, Generic[T]):
    """Offset-paginated collection response."""

    items: list[T]
    total: int = Field(description="Total rows matching the query, ignoring pagination")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class ErrorDetail(BaseModel):
    """Uniform error body returned by the API's exception handlers."""

    detail: str
    code: str | None = None
