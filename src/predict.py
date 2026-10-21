"""Machine assessment: failure probability, remaining life and indicators.

The output is the decision a maintenance planner actually needs:

    Machine M-17
      Failure probability : 83.4%
      Estimated failure    : 6.4 days
      Primary indicators   : vibration_mm_s (+3.9 sigma), temperature_c (+2.4 sigma)
      Recommended          : Inspect bearing within 48 hours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # allow `python src/predict.py`
    sys.path.insert(0, str(ROOT_DIR))

from src.features import build_window_features  # noqa: E402
from src.simulator import SENSOR_COLUMNS  # noqa: E402

DEFAULT_BUNDLE = ROOT_DIR / "models" / "pdm_bundle.joblib"

# One telemetry row == one operating hour.
CYCLE_HOURS = 1.0

CRITICAL_PROBABILITY = 0.75
WATCH_PROBABILITY = 0.35
CRITICAL_RUL_HOURS = 24 * 3  # within 3 days of predicted failure
WATCH_RUL_HOURS = 24 * 10  # within 10 days

# Sensor -> human-readable component name (drives the recommended action).
COMPONENT_LABELS = {
    "vibration_mm_s": "bearing / rotor vibration",
    "temperature_c": "temperature",
    "current_a": "motor current",
    "pressure_bar": "pressure",
    "rpm": "rotational speed",
    "voltage_v": "voltage",
}


def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    """Load the trained model bundle."""
    bundle_file = Path(path)
    if not bundle_file.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_file}. Run `python src/train.py` first."
        )
    return joblib.load(bundle_file)


def _risk_band(probability: float, rul_hours: float) -> str:
    if probability >= CRITICAL_PROBABILITY or rul_hours <= CRITICAL_RUL_HOURS:
        return "CRITICAL"
    if probability >= WATCH_PROBABILITY or rul_hours <= WATCH_RUL_HOURS:
        return "WATCH"
    return "HEALTHY"


def _indicators(raw: pd.DataFrame, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare the recent window against the fleet's healthy operating point."""
    window = int(bundle["window"])
    baselines = bundle.get("baselines", {})
    recent = raw.tail(window)

    indicators: list[dict[str, Any]] = []
    for sensor in SENSOR_COLUMNS:
        base = baselines.get(sensor, {"mean": 0.0, "std": 1.0})
        values = recent[sensor].astype(float)
        current_mean = float(values.mean())
        z_score = (current_mean - base["mean"]) / (base["std"] or 1.0)
        direction = "up" if z_score > 0.5 else "down" if z_score < -0.5 else "stable"
        indicators.append(
            {
                "sensor": sensor,
                "component": COMPONENT_LABELS.get(sensor, sensor),
                "current_mean": round(current_mean, 3),
                "healthy_mean": round(float(base["mean"]), 3),
                "sigma": round(float(z_score), 2),
                "direction": direction,
                "anomalous": bool(abs(z_score) >= 3.0),
            }
        )
    indicators.sort(key=lambda item: abs(item["sigma"]), reverse=True)
    return indicators


def _recommended_action(band: str, indicators: list[dict[str, Any]]) -> str:
    abnormal = [item for item in indicators if item["anomalous"]]
    drivers = ", ".join(
        f"{item['sensor']} ({item['direction']} {abs(item['sigma']):.1f} sigma)" for item in abnormal[:3]
    )
    if band == "CRITICAL":
        base = "Schedule emergency inspection within 24 hours."
    elif band == "WATCH":
        base = "Inspect within 48 hours and raise monitoring frequency."
    else:
        base = "Continue routine monitoring - no action required."
    if drivers:
        return f"{base} Primary indicators: {drivers}."
    return base


def assess_machine(raw: pd.DataFrame, machine_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
    """Score one machine's most recent state."""
    subset = raw[raw["machine_id"] == machine_id]
    if subset.empty:
        raise KeyError(f"Unknown machine: {machine_id}")
    subset = subset.sort_values("cycle")

    window = int(bundle["window"])
    horizon = int(bundle.get("horizon", 10))
    frame = build_window_features(subset, window=window, horizon=horizon)
    if frame.empty:
        raise ValueError(f"Machine {machine_id} needs at least {window} telemetry rows")

    latest = frame.iloc[[-1]]
    X = latest[bundle["features"]]

    probability = float(bundle["failure"].predict_proba(X)[0][1])
    rul_cycles = max(float(bundle["rul"].predict(X)[0]), 0.0)
    rul_hours = rul_cycles * CYCLE_HOURS
    anomaly_score = float(-bundle["anomaly"].score_samples(X)[0])

    indicators = _indicators(subset, bundle)
    band = _risk_band(probability, rul_hours)

    return {
        "machine_id": machine_id,
        "cycle": int(latest["cycle"].iloc[0]),
        "failure_probability": round(probability, 4),
        "risk_band": band,
        "estimated_rul_hours": round(rul_hours, 1),
        "estimated_rul_days": round(rul_hours / 24.0, 2),
        "anomaly_score": round(anomaly_score, 4),
        "primary_indicators": indicators[:3],
        "all_indicators": indicators,
        "recommended_action": _recommended_action(band, indicators),
        "model_version": {
            "window": window,
            "horizon": horizon,
            "features": len(bundle["features"]),
        },
    }


def assess_all(raw: pd.DataFrame, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Score every machine present in the telemetry."""
    return [assess_machine(raw, machine_id, bundle) for machine_id in sorted(raw["machine_id"].unique())]


def rank_machines(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order assessments by urgency (highest failure probability first)."""
    return sorted(assessments, key=lambda item: item["failure_probability"], reverse=True)


def to_json(assessment: dict[str, Any]) -> str:
    """Serialise an assessment for logs or API payloads."""
    return json.dumps(assessment, indent=2)


__all__ = [
    "assess_all",
    "assess_machine",
    "load_bundle",
    "rank_machines",
    "to_json",
]
