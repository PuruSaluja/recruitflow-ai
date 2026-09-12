"""Application settings, loaded from environment variables or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden with a ``RECRUITFLOW_``-prefixed environment
    variable, e.g. ``RECRUITFLOW_DATABASE_URL``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RECRUITFLOW_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_path` would otherwise collide with pydantic's protected
        # ``model_`` namespace.
        protected_namespaces=(),
    )

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/recruitflow"
    test_database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/recruitflow_test"
    )

    #: Seconds to wait for a TCP connection to PostgreSQL before giving up.
    #: Without this, an unreachable database makes requests hang indefinitely
    #: instead of failing fast, which defeats the point of /health.
    #:
    #: This bounds one connection attempt, not the response: a refused port
    #: comes back at once, and pool_pre_ping can spend a second attempt when
    #: the connection it is checking was already in the pool. Measured against
    #: an unroutable host on a cold pool, /jobs answered 503 in 3.1s at this
    #: setting -- and still in 3.0s at 6s, the network having given up first,
    #: so do not read the setting as a predictor of worst-case latency. Kept at
    #: 3s so a failing /health answers inside a 5-10s orchestrator probe.
    db_connect_timeout: int = 3
    #: Seconds to wait for a free connection when the pool is saturated.
    db_pool_timeout: int = 10

    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173"

    model_path: str = "backend/app/ml/artifacts/matcher.joblib"
    random_seed: int = 42

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list; the env var is a comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_model_path(self) -> Path:
        """Absolute path to the persisted matcher artifact."""
        path = Path(self.model_path)
        return path if path.is_absolute() else REPO_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the .env file is read once per process."""
    return Settings()
