"""Rolling-window feature builder.

One training row per (machine, cycle) pair, built from the last ``window``
hours of telemetry:

- ``{sensor}_mean``  average level over the window (drift)
- ``{sensor}_std``   variability over the window (instability)
- ``{sensor}_slope`` least-squares trend per cycle (degradation rate)
- ``{sensor}_last``  most recent reading (immediate state)

Labels (training only):
- ``rul``               remaining useful life, capped at ``MAX_RUL``
- ``failed_within``     1 when the machine fails within the next N cycles
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulator import MAX_RUL, SENSOR_COLUMNS

IDENTIFIER_COLUMNS = ["machine_id", "cycle"]
LABEL_COLUMNS = ["rul", "failed_within"]


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against their index."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_var = x.var()
    if x_var == 0:
        return 0.0
    return float(np.cov(x, values, bias=True)[0, 1] / x_var)


def build_window_features(
    df: pd.DataFrame,
    window: int = 16,
    horizon: int = 10,
) -> pd.DataFrame:
    """Expand raw telemetry into rolling-window feature rows.

    Rows before a machine has ``window`` observations are dropped (cold start).
    """
    missing = [column for column in SENSOR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Telemetry is missing required sensors: {missing}")

    records: list[dict[str, float | int | str]] = []
    for machine_id, group in df.sort_values(["machine_id", "cycle"]).groupby("machine_id"):
        rows = group.to_dict("records")
        cycles = [int(row["cycle"]) for row in rows]
        failed = [int(row.get("failed", 0)) for row in rows]
        for index in range(window - 1, len(rows)):
            slice_rows = rows[index - window + 1 : index + 1]
            record: dict[str, float | int | str] = {
                "machine_id": machine_id,
                "cycle": cycles[index],
            }
            for sensor in SENSOR_COLUMNS:
                series = np.asarray([float(row[sensor]) for row in slice_rows], dtype=float)
                record[f"{sensor}_mean"] = float(series.mean())
                record[f"{sensor}_std"] = float(series.std(ddof=0))
                record[f"{sensor}_slope"] = _slope(series)
                record[f"{sensor}_last"] = float(series[-1])

            raw_rul = rows[index].get("rul")
            record["rul"] = float(min(raw_rul, MAX_RUL)) if raw_rul is not None else MAX_RUL

            # "Needs intervention now or within the horizon": inclusive of the
            # current cycle, so the failing cycle itself is a positive sample.
            # The scheduled next failure drives the label - it may lie beyond
            # the observed telemetry, which is exactly what we must predict.
            next_fault = rows[index].get("next_fault_cycle")
            if next_fault is not None and int(next_fault) >= 0:
                record["failed_within"] = int(0 <= int(next_fault) - int(cycles[index]) <= horizon)
            else:
                horizon_slice = failed[index : index + horizon]
                record["failed_within"] = int(1 if any(horizon_slice) else 0)
            records.append(record)

    if not records:
        return pd.DataFrame(columns=[*IDENTIFIER_COLUMNS, *feature_names(), *LABEL_COLUMNS])
    return pd.DataFrame(records)


def feature_names() -> list[str]:
    """Ordered model feature names (stable across runs)."""
    return [f"{sensor}_{stat}" for sensor in SENSOR_COLUMNS for stat in ("mean", "std", "slope", "last")]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns present in ``frame`` (in canonical order)."""
    return [name for name in feature_names() if name in frame.columns]
