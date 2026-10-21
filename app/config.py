"""Application configuration for the predictive-maintenance service."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment-driven settings (12-factor)."""

    APP_ENV: str = "development"
    API_KEY: str = ""
    BACKEND_ORIGINS: str = "http://localhost:8501,http://localhost:3000"
    DATA_PATH: str = str(ROOT_DIR / "data" / "telemetry.csv")
    MODEL_PATH: str = str(ROOT_DIR / "models" / "pdm_bundle.joblib")
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
