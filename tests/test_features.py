"""Rolling-window feature builder: shape, labels and cold-start handling."""

from __future__ import annotations

import pandas as pd

from src.features import build_window_features, feature_columns, feature_names
from src.simulator import MAX_RUL


def test_one_row_per_machine_cycle_after_warmup(fleet_frame):
    frame = build_window_features(fleet_frame, window=8, horizon=6)
    # One row per machine/cycle once the window is full. Faulted machines stop
    # producing telemetry at their failure, so expectations follow the history.
    expected = sum(len(group) - 8 + 1 for _, group in fleet_frame.groupby("machine_id"))
    assert len(frame) == expected
    assert frame.groupby("machine_id")["cycle"].min().eq(8).all()


def test_features_are_finite(fleet_frame):
    frame = build_window_features(fleet_frame, window=8, horizon=6)
    assert frame[feature_columns(frame)].notna().all().all()
    assert frame[feature_columns(frame)].apply(lambda col: col.abs().max() < 1e9).all()


def test_feature_names_cover_every_sensor_statistic():
    names = feature_names()
    assert len(names) == 24
    assert "vibration_mm_s_slope" in names
    assert "temperature_c_mean" in names


def test_failed_within_label_marks_the_pre_failure_window(fleet_frame):
    frame = build_window_features(fleet_frame, window=8, horizon=6)
    failing = frame[(frame["machine_id"] == "M-01") & (frame["cycle"] == 95)]
    assert failing["failed_within"].iloc[0] == 1  # failure at cycle 100 is 5 steps away
    safe = frame[(frame["machine_id"] == "M-01") & (frame["cycle"] == 50)]
    assert safe["failed_within"].iloc[0] == 0


def test_rul_label_is_capped(fleet_frame):
    frame = build_window_features(fleet_frame, window=8, horizon=6)
    assert frame["rul"].max() <= MAX_RUL
    healthy = frame[frame["machine_id"] == "M-03"]
    assert healthy["rul"].iloc[0] == MAX_RUL  # censored machine saturates the cap


def test_missing_sensor_is_rejected():
    frame = pd.DataFrame({"machine_id": ["M-01"], "cycle": [1], "temperature_c": [60.0]})
    try:
        build_window_features(frame, window=4)
    except ValueError as exc:
        assert "missing required sensors" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing sensors")
