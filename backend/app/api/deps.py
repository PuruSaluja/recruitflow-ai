"""Shared FastAPI dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query


@dataclass(frozen=True)
class Pagination:
    """Offset pagination, shared by every collection endpoint."""

    limit: int
    offset: int


def pagination(
    limit: int = Query(50, ge=1, le=200, description="Maximum rows to return"),
    offset: int = Query(0, ge=0, description="Rows to skip"),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)
