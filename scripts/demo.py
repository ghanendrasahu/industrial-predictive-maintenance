"""Print a fleet risk report from the trained models.

Usage:
    python scripts/demo.py [--data data/telemetry.csv] [--machine M-03]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from src.predict import assess_all, load_bundle, rank_machines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fleet risk report.")
    parser.add_argument("--data", default=str(ROOT_DIR / "data" / "telemetry.csv"))
    parser.add_argument("--bundle", default=str(ROOT_DIR / "models" / "pdm_bundle.joblib"))
    parser.add_argument("--machine", default=None, help="Show the full payload for one machine")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    telemetry = pd.read_csv(args.data)
    bundle = load_bundle(args.bundle)
    assessments = rank_machines(assess_all(telemetry, bundle))

    if args.machine:
        match = next((item for item in assessments if item["machine_id"] == args.machine), None)
        if match is None:
            raise SystemExit(f"unknown machine: {args.machine}")
        print(json.dumps(match, indent=2))
        return

    print(f"{'machine':<10}{'P(failure)':>12}{'band':>10}{'RUL (days)':>13}  next action")
    print("-" * 88)
    for item in assessments:
        print(
            f"{item['machine_id']:<10}{item['failure_probability']:>11.1%}"
            f"{item['risk_band']:>10}{item['estimated_rul_days']:>13.1f}  "
            f"{item['recommended_action']}"
        )

    if args.json:
        print(json.dumps(assessments, indent=2))


if __name__ == "__main__":
    main()
