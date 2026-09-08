"""Version 1 of the REST API."""

from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()
api_router.include_router(health.router)

__all__ = ["api_router"]
