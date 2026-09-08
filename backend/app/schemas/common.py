"""Shared response envelopes."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


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
