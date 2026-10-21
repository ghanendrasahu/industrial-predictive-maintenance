"""Train the predictive-maintenance model set.

Three complementary models over the same window features:

1. ``anomaly``   IsolationForest - unsupervised novelty of the current state
2. ``rul``       RandomForestRegressor - remaining useful life in cycles
3. ``failure``   GradientBoostingClassifier - P(failure within the horizon)

Machines are split by unit (never by row) so holdout evaluation reflects
genuine generalization to unseen equipment, not to unseen rows of a machine
the model already learned.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # allow `python src/train.py`
    sys.path.insert(0, str(ROOT_DIR))

from src.features import build_window_features, feature_columns  # noqa: E402
from src.simulator import MAX_RUL, SENSOR_COLUMNS  # noqa: E402

DEFAULT_BUNDLE = ROOT_DIR / "models" / "pdm_bundle.joblib"
DEFAULT_META = ROOT_DIR / "models" / "metadata.json"
DEFAULT_DATA = ROOT_DIR / "data" / "telemetry.csv"


@dataclass
class TrainingConfig:
    """Hyper-parameters for a training run."""

    window: int = 16
    horizon: int = 168  # planning horizon: 7 days of operation (1 row = 1 hour)
    n_estimators: int = 200
    holdout_fraction: float = 0.25
    contamination: float = 0.02
    seed: int = 42
    baseline_rul_margin: int = field(default=60)

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ValueError("window must be >= 4 to form meaningful statistics")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def _split_machines(machines: list[str], holdout_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ordered = sorted(machines)
    shuffled = list(rng.permutation(ordered))
    holdout_count = max(1, int(round(len(ordered) * holdout_fraction))) if len(ordered) > 1 else 0
    holdout = sorted(shuffled[:holdout_count])
    train = sorted(m for m in ordered if m not in holdout)
    return train, holdout


def _healthy_baselines(raw: pd.DataFrame, margin: int) -> dict[str, dict[str, float]]:
    """Fleet-wide healthy operating point per sensor (from far-from-failure rows)."""
    baselines: dict[str, dict[str, float]] = {}
    healthy = raw[raw.get("rul", pd.Series(dtype=float)) >= margin] if "rul" in raw.columns else raw
    if healthy.empty:
        healthy = raw
    for sensor in SENSOR_COLUMNS:
        values = healthy[sensor].astype(float)
        baselines[sensor] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)) or 1.0,
        }
    return baselines


def train(
    data_path: str | Path = DEFAULT_DATA,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    metadata_path: str | Path = DEFAULT_META,
    config: TrainingConfig | None = None,
) -> tuple[dict, dict]:
    """Train all models, persist them and return ``(bundle, metadata)``."""
    cfg = config or TrainingConfig()
    raw = pd.read_csv(data_path)
    if raw.empty:
        raise ValueError(f"No telemetry found in {data_path}")

    frame = build_window_features(raw, window=cfg.window, horizon=cfg.horizon)
    if frame.empty:
        raise ValueError("Not enough telemetry to build window features; lower --window.")

    features = feature_columns(frame)
    if not features:
        raise ValueError("No feature columns produced - check the telemetry schema.")

    train_machines, holdout_machines = _split_machines(
        sorted(frame["machine_id"].unique().tolist()), cfg.holdout_fraction, cfg.seed
    )
    train_frame = frame[frame["machine_id"].isin(train_machines)]
    holdout_frame = frame[frame["machine_id"].isin(holdout_machines)]

    X_train = train_frame[features]
    y_rul = train_frame["rul"].astype(float)
    y_fail = train_frame["failed_within"].astype(int)

    anomaly = IsolationForest(
        contamination=cfg.contamination, random_state=cfg.seed, n_jobs=-1
    ).fit(X_train)

    rul_model = RandomForestRegressor(
        n_estimators=cfg.n_estimators, min_samples_leaf=2, random_state=cfg.seed, n_jobs=-1
    ).fit(X_train, y_rul)

    failure_model: Any
    if y_fail.nunique() > 1:
        base_classifier = GradientBoostingClassifier(random_state=cfg.seed)
        # Calibrate so the reported probability is trustworthy, not merely
        # separable - a raw GBM is over-confident on clean degradation curves.
        if int(y_fail.sum()) >= 3 and int((1 - y_fail).sum()) >= 3:
            failure_model = CalibratedClassifierCV(
                base_classifier, method="sigmoid", cv=3
            ).fit(X_train, y_fail)
        else:
            failure_model = base_classifier.fit(X_train, y_fail)
    else:
        # No imminent failure in the training window: keep the pipeline alive
        # with a constant prior instead of crashing on a single-class target.
        failure_model = DummyClassifier(strategy="prior").fit(X_train, y_fail)

    metrics: dict[str, float] = {}
    if not holdout_frame.empty:
        X_hold = holdout_frame[features]
        rul_pred = rul_model.predict(X_hold)
        metrics["rul_mae_cycles"] = round(float(mean_absolute_error(holdout_frame["rul"], rul_pred)), 3)
        holdout_labels = holdout_frame["failed_within"].astype(int)
        if holdout_labels.nunique() > 1:
            probabilities = failure_model.predict_proba(X_hold)[:, 1]
            metrics["failure_auc"] = round(float(roc_auc_score(holdout_labels, probabilities)), 4)
    metrics["train_rows"] = int(len(train_frame))
    metrics["holdout_rows"] = int(len(holdout_frame))

    bundle = {
        "anomaly": anomaly,
        "rul": rul_model,
        "failure": failure_model,
        "features": features,
        "window": cfg.window,
        "horizon": cfg.horizon,
        "baselines": _healthy_baselines(raw, cfg.baseline_rul_margin),
        "max_rul": MAX_RUL,
    }

    metadata = {
        "window": cfg.window,
        "horizon": cfg.horizon,
        "features": features,
        "train_machines": train_machines,
        "holdout_machines": holdout_machines,
        "metrics": metrics,
        "trained_rows": int(len(frame)),
        "trained_machines": int(frame["machine_id"].nunique()),
    }

    bundle_file = Path(bundle_path)
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_file)
    meta_file = Path(metadata_path)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the predictive-maintenance models.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "models"))
    parser.add_argument("--window", type=int, default=TrainingConfig.window)
    parser.add_argument("--horizon", type=int, default=TrainingConfig.horizon)
    parser.add_argument("--trees", type=int, default=TrainingConfig.n_estimators)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    bundle, metadata = train(
        data_path=args.data,
        bundle_path=out_dir / "pdm_bundle.joblib",
        metadata_path=out_dir / "metadata.json",
        config=TrainingConfig(
            window=args.window, horizon=args.horizon, n_estimators=args.trees, seed=args.seed
        ),
    )
    print(f"trained on {metadata['trained_rows']:,} rows from {metadata['trained_machines']} machines")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"bundle  -> {out_dir / 'pdm_bundle.joblib'}")
    print(f"metadata-> {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
