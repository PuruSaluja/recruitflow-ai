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
