"""Version 1 of the REST API."""

from fastapi import APIRouter

from app.api.v1 import applications, candidates, health, jobs, matches, reports

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(jobs.router)
api_router.include_router(candidates.router)
api_router.include_router(applications.router)
api_router.include_router(matches.router)
api_router.include_router(reports.router)

__all__ = ["api_router"]
