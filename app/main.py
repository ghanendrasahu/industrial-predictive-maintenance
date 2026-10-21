"""FastAPI service for the predictive-maintenance engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from app.config import settings
from src.predict import assess_all, assess_machine, load_bundle, rank_machines

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_state: dict[str, Any] = {"bundle": None, "telemetry": None}


async def require_api_key(x_api_key: str | None = Security(api_key_header)) -> None:
    """Gate API routes when ``API_KEY`` is configured (empty disables it)."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _cors_origins() -> list[str]:
    return [origin.strip() for origin in settings.BACKEND_ORIGINS.split(",") if origin.strip()]


def _load_state() -> None:
    """Load telemetry + model bundle into process state."""
    try:
        telemetry = pd.read_csv(settings.DATA_PATH)
    except FileNotFoundError:
        telemetry = pd.DataFrame()
    try:
        bundle = load_bundle(settings.MODEL_PATH)
    except FileNotFoundError:
        bundle = None
    _state["telemetry"] = telemetry
    _state["bundle"] = bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _load_state()
    yield


app = FastAPI(
    title="Industrial Predictive Maintenance",
    description=(
        "Telemetry -> features -> anomaly + RUL + failure probability -> "
        "maintenance recommendation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> tuple[pd.DataFrame, dict[str, Any]]:
    telemetry = _state.get("telemetry")
    bundle = _state.get("bundle")
    if telemetry is None or bundle is None or telemetry.empty:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: telemetry and/or trained model bundle missing.",
        )
    return telemetry, bundle


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness + readiness probe."""
    telemetry = _state.get("telemetry")
    bundle = _state.get("bundle")
    return {
        "status": "healthy" if bundle is not None and telemetry is not None else "starting",
        "model_loaded": bundle is not None,
        "rows": int(len(telemetry)) if telemetry is not None else 0,
        "machines": int(telemetry["machine_id"].nunique())
        if telemetry is not None and not telemetry.empty
        else 0,
    }


@app.get("/api/machines", dependencies=[Depends(require_api_key)])
async def list_machines() -> dict[str, Any]:
    """Every machine with its latest risk snapshot, most urgent first."""
    telemetry, bundle = _require_ready()
    assessments = rank_machines(assess_all(telemetry, bundle))
    return {
        "count": len(assessments),
        "machines": [
            {
                "machine_id": item["machine_id"],
                "cycle": item["cycle"],
                "failure_probability": item["failure_probability"],
                "risk_band": item["risk_band"],
                "estimated_rul_days": item["estimated_rul_days"],
                "recommended_action": item["recommended_action"],
            }
            for item in assessments
        ],
    }


@app.get("/api/predict/{machine_id}", dependencies=[Depends(require_api_key)])
async def predict(machine_id: str) -> dict[str, Any]:
    """Full assessment for one machine (probability, RUL, indicators, action)."""
    telemetry, bundle = _require_ready()
    try:
        return assess_machine(telemetry, machine_id, bundle)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/refresh", dependencies=[Depends(require_api_key)])
async def refresh() -> dict[str, Any]:
    """Re-read telemetry + model bundle (picks up new training runs)."""
    _load_state()
    telemetry = _state.get("telemetry")
    return {
        "status": "reloaded",
        "rows": int(len(telemetry)) if telemetry is not None else 0,
    }
