"""Shared fixtures: a small seeded fleet and a trained (tiny) model bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulator import generate_machine  # noqa: E402
from src.train import TrainingConfig, train  # noqa: E402


@pytest.fixture(scope="session")
def fleet_frame() -> pd.DataFrame:
    """Three units: two that eventually fail, one that never does."""
    frames = [
        generate_machine("M-01", 140, fault_cycle=100, seed=1),
        generate_machine("M-02", 140, fault_cycle=120, seed=2),
        generate_machine("M-03", 140, fault_cycle=None, seed=3),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, fleet_frame) -> dict:
    """Train a small model set on the fixture fleet and expose its artifacts."""
    workdir = tmp_path_factory.mktemp("pdm")
    data_csv = workdir / "telemetry.csv"
    fleet_frame.to_csv(data_csv, index=False)

    bundle_path = workdir / "pdm_bundle.joblib"
    metadata_path = workdir / "metadata.json"
    bundle, metadata = train(
        data_path=data_csv,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        config=TrainingConfig(
            window=8, horizon=6, n_estimators=20, holdout_fraction=0.34, seed=42
        ),
    )
    return {
        "bundle": bundle,
        "metadata": metadata,
        "data_csv": data_csv,
        "bundle_path": bundle_path,
        "metadata_path": metadata_path,
    }
