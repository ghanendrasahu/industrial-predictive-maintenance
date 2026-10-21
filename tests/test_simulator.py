"""Telemetry simulator: determinism, schema and degradation physics."""

from __future__ import annotations

import pandas as pd

from src.simulator import SENSOR_COLUMNS, generate_fleet, generate_machine


def test_generation_is_deterministic():
    first = generate_machine("M-01", 60, fault_cycle=40, seed=7)
    second = generate_machine("M-01", 60, fault_cycle=40, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_diverge():
    first = generate_machine("M-01", 60, fault_cycle=40, seed=7)
    second = generate_machine("M-01", 60, fault_cycle=40, seed=8)
    assert not first["vibration_mm_s"].equals(second["vibration_mm_s"])


def test_schema_and_types():
    frame = generate_machine("M-01", 30, fault_cycle=None, seed=3)
    assert list(frame.columns) == [
        "machine_id",
        "cycle",
        "rul",
        "failed",
        "next_fault_cycle",
        *SENSOR_COLUMNS,
    ]
    assert frame["machine_id"].nunique() == 1
    assert frame["cycle"].tolist() == list(range(1, 31))
    assert frame["failed"].dtype.kind in "iu"


def test_machine_is_repaired_and_keeps_running():
    frame = generate_machine("M-01", 60, fault_cycle=25, seed=3)
    assert frame["cycle"].max() == 60  # telemetry continues after the repair
    failure_row = frame[frame["failed"] == 1].iloc[0]
    assert failure_row["cycle"] == 25
    assert failure_row["rul"] == 0
    # The repaired machine runs on and can fail again later.
    assert frame["failed"].sum() >= 1
    assert frame.iloc[-1]["failed"] == 0


def test_degradation_raises_vibration_and_temperature():
    frame = generate_machine("M-01", 200, fault_cycle=150, seed=11)
    early = frame.head(20)
    pre_fault = frame[(frame["cycle"] >= 131) & (frame["cycle"] <= 150)]
    assert pre_fault["vibration_mm_s"].mean() > early["vibration_mm_s"].mean() * 1.5
    assert pre_fault["temperature_c"].mean() > early["temperature_c"].mean()
    assert pre_fault["current_a"].mean() > early["current_a"].mean()


def test_healthy_machine_never_fails():
    frame = generate_machine("M-09", 100, fault_cycle=None, seed=5)
    assert frame["failed"].sum() == 0
    assert (frame["rul"] > 0).all()


def test_rul_counts_down_to_zero_at_failure():
    frame = generate_machine("M-01", 100, fault_cycle=80, seed=13)
    failing_row = frame[frame["failed"] == 1].iloc[0]
    assert failing_row["cycle"] == 80
    assert failing_row["rul"] == 0
    earlier = frame[frame["cycle"] == 79].iloc[0]
    assert earlier["rul"] == 1


def test_generate_fleet_mixes_healthy_and_failing_units():
    fleet = generate_fleet(n_machines=6, cycles=200, seed=21)
    assert fleet["machine_id"].nunique() == 6
    assert fleet["failed"].sum() > 0
    healthy = fleet.groupby("machine_id")["failed"].sum()
    assert (healthy == 0).any()
