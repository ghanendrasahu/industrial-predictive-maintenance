"""Seeded synthetic telemetry generator with degradation physics.

A fleet of rotating machines emits one record per operating hour ("cycle").
Healthy baselines vary per unit; once a fault is seeded (e.g. bearing wear),
sensors drift in physically-plausible directions as failure approaches:

- vibration rises super-linearly (bearing wear)
- temperature climbs (friction)
- current increases (motor load)
- pressure sags
- rpm droops and flutters

Every machine is reproducible from its seed, which makes the whole pipeline
testable offline and CI-stable. In production these rows would come from an
OPC-UA / MQTT historian; the label columns (``rul``, ``failed``) would come
from maintenance records instead of the seeder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "voltage_v",
    "current_a",
]
OUTPUT_COLUMNS = [
    "machine_id",
    "cycle",
    "rul",
    "failed",
    "next_fault_cycle",
    *SENSOR_COLUMNS,
]

# RUL training cap - rows far from any failure teach the model nothing and
# blow up targets for censored (never-failed) machines. 400 cycles ~= 16.7 days
# at one row per operating hour.
MAX_RUL = 400

# Nominal healthy operating points. Units are realistic for a mid-size
# three-phase induction machine on a pumping/compressor duty cycle.
_HEALTHY = {
    "temperature_c": (65.0, 0.8),
    "vibration_mm_s": (2.5, 0.15),
    "pressure_bar": (5.2, 0.05),
    "rpm": (1780.0, 6.0),
    "voltage_v": (231.0, 0.8),
    "current_a": (10.4, 0.12),
}


def _fault_schedule(n_cycles: int, first_fault: int | None, rng: np.random.Generator) -> list[int]:
    """Failure cycles of one machine.

    A machine that fails is repaired and returns to service; its next life is
    shorter than the previous one (wear does not reset to factory condition).
    The schedule therefore contains several failures per machine, and the
    telemetry keeps flowing after each repair - exactly how a real fleet
    behaves, and what makes the final telemetry a *prediction* point rather
    than a known failure.
    """
    if first_fault is None:
        return []
    faults = [int(first_fault)]
    life = int(first_fault * rng.uniform(0.6, 0.8))
    nxt = faults[-1] + life
    while nxt < n_cycles:
        faults.append(int(nxt))
        life = max(int(life * rng.uniform(0.6, 0.8)), 10)
        nxt = faults[-1] + life
    return faults


def generate_machine(
    machine_id: str,
    n_cycles: int,
    fault_cycle: int | None,
    seed: int,
    snapshot_cycle: int | None = None,
) -> pd.DataFrame:
    """Telemetry history of one machine.

    ``fault_cycle = None`` renders a permanently healthy (censored) unit.
    ``snapshot_cycle`` truncates the *visible* history while the failure
    schedule keeps running in the future - that is what a live fleet looks
    like: telemetry up to now, failure still ahead of us.
    """
    rng = np.random.default_rng(seed)

    # Unit-to-unit variation: healthy machines differ a little from fleet norms.
    unit_shift = {
        sensor: rng.normal(0.0, scale * 0.6) for sensor, (_, scale) in _HEALTHY.items()
    }

    faults = _fault_schedule(n_cycles, fault_cycle, rng)
    visible_cycles = snapshot_cycle or n_cycles
    rows: list[dict[str, float | int | str]] = []

    for cycle in range(1, visible_cycles + 1):
        failed = int(cycle in faults)

        next_fault = next((fault for fault in faults if fault >= cycle), None)
        previous_fault = max((fault for fault in faults if fault < cycle), default=0)

        if next_fault is None:
            # No failure scheduled ahead: healthy drift, censored RUL.
            severity = 0.08 * (cycle / max(n_cycles, 1))
            rul = MAX_RUL
        else:
            # Severity ramps towards the next failure, restarting from the last
            # repair - the degradation signature the models must learn.
            progress = (cycle - previous_fault) / max(next_fault - previous_fault, 1)
            severity = 0.85 * max(progress, 0.0) ** 1.6
            rul = max(next_fault - cycle, 0)

        flutter = 1.0 + 2.5 * severity
        row: dict[str, float | int | str] = {
            "machine_id": machine_id,
            "cycle": cycle,
            "rul": rul,
            "failed": failed,
            "next_fault_cycle": next_fault if next_fault is not None else -1,
            "temperature_c": _HEALTHY["temperature_c"][0] + unit_shift["temperature_c"]
            + severity**1.4 * 12.0
            + rng.normal(0, _HEALTHY["temperature_c"][1]),
            "vibration_mm_s": _HEALTHY["vibration_mm_s"][0] + unit_shift["vibration_mm_s"]
            + severity**1.8 * 3.2
            + rng.normal(0, _HEALTHY["vibration_mm_s"][1] * flutter),
            "pressure_bar": _HEALTHY["pressure_bar"][0] + unit_shift["pressure_bar"]
            - severity * 0.35
            + rng.normal(0, _HEALTHY["pressure_bar"][1]),
            "rpm": _HEALTHY["rpm"][0] + unit_shift["rpm"]
            - severity * 35.0
            + rng.normal(0, _HEALTHY["rpm"][1] * flutter),
            "voltage_v": _HEALTHY["voltage_v"][0] + unit_shift["voltage_v"]
            + rng.normal(0, _HEALTHY["voltage_v"][1]),
            "current_a": _HEALTHY["current_a"][0] + unit_shift["current_a"]
            + severity * 0.9
            + rng.normal(0, _HEALTHY["current_a"][1]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# Cycles of warning before the upcoming failure, cycled across at-risk units so
# the fleet spans several risk bands (from "inspect today" to "plan ahead").
LEAD_TIMES = [35, 60, 120, 154]


def generate_fleet(
    n_machines: int = 8,
    cycles: int = 900,
    seed: int = 42,
    unhealthy_share: float = 0.75,
) -> pd.DataFrame:
    """A fleet whose units sit at different points of their degradation curve.

    At-risk units have their visible telemetry truncated a few cycles before
    their scheduled failure - that is the live situation: the failure is still
    ahead of us and has to be predicted.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    at_risk = round(n_machines * unhealthy_share)

    for index in range(n_machines):
        machine_seed = int(rng.integers(1_000_000))
        machine_id = f"M-{index + 1:02d}"
        horizon = cycles if index % 2 == 0 else int(cycles * rng.uniform(0.75, 1.0))

        if index < at_risk:
            lead = LEAD_TIMES[index % len(LEAD_TIMES)]
            first_fault = int(horizon * rng.uniform(0.35, 0.6))
            schedule = _fault_schedule(horizon, first_fault, np.random.default_rng(machine_seed))
            upcoming = schedule[-1] if schedule else first_fault
            snapshot = max(upcoming - lead, 40)
            frames.append(
                generate_machine(
                    machine_id, horizon, first_fault, machine_seed, snapshot_cycle=snapshot
                )
            )
        else:
            frames.append(generate_machine(machine_id, horizon, None, machine_seed))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic machine telemetry.")
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "telemetry.csv"))
    args = parser.parse_args()

    df = generate_fleet(args.machines, args.cycles, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(out, index=False)
    print(f"wrote {len(df):,} rows for {args.machines} machines -> {out}")


if __name__ == "__main__":
    main()
