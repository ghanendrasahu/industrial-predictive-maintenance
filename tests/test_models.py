"""Model training + machine assessment behaviour."""

from __future__ import annotations

import json

import pytest

from src.predict import assess_all, assess_machine, rank_machines


def test_training_persists_bundle_and_metadata(trained):
    assert trained["bundle_path"].exists()
    assert trained["metadata_path"].exists()
    metadata = json.loads(trained["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["window"] == 8
    assert metadata["features"]
    assert metadata["metrics"]["train_rows"] > 0


def test_holdout_split_is_machine_level(trained):
    metadata = trained["metadata"]
    train_ids = set(metadata["train_machines"])
    holdout_ids = set(metadata["holdout_machines"])
    assert holdout_ids, "expected at least one held-out machine"
    assert not (train_ids & holdout_ids)


def test_assessment_payload_is_complete(trained, fleet_frame):
    result = assess_machine(fleet_frame, "M-01", trained["bundle"])
    assert result["machine_id"] == "M-01"
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_band"] in {"CRITICAL", "WATCH", "HEALTHY"}
    assert result["estimated_rul_days"] >= 0
    assert result["recommended_action"]
    assert len(result["primary_indicators"]) == 3


def test_indicators_are_ranked_by_deviation(trained, fleet_frame):
    result = assess_machine(fleet_frame, "M-01", trained["bundle"])
    sigmas = [abs(item["sigma"]) for item in result["all_indicators"]]
    assert sigmas == sorted(sigmas, reverse=True)
    assert all(item["direction"] in {"up", "down", "stable"} for item in result["all_indicators"])


def test_failing_machine_scores_worse_than_healthy_one(trained, fleet_frame):
    # Score M-01 at the moment of its first failure, against a unit that never
    # fails: the degrading machine must carry the higher failure probability.
    at_failure = fleet_frame[
        (fleet_frame["machine_id"] == "M-01") & (fleet_frame["cycle"] <= 100)
    ]
    failing = assess_machine(at_failure, "M-01", trained["bundle"])
    healthy = assess_machine(fleet_frame, "M-03", trained["bundle"])
    assert failing["failure_probability"] > healthy["failure_probability"]
    assert healthy["risk_band"] != "CRITICAL"


def test_assess_all_covers_every_machine(trained, fleet_frame):
    assessments = assess_all(fleet_frame, trained["bundle"])
    assert len(assessments) == 3
    ranked = rank_machines(assessments)
    probabilities = [item["failure_probability"] for item in ranked]
    assert probabilities == sorted(probabilities, reverse=True)


def test_unknown_machine_raises_keyerror(trained, fleet_frame):
    with pytest.raises(KeyError):
        assess_machine(fleet_frame, "M-99", trained["bundle"])


def test_machine_with_too_little_history_is_rejected(trained, fleet_frame):
    truncated = fleet_frame[fleet_frame["cycle"] <= 3]
    with pytest.raises(ValueError):
        assess_machine(truncated, "M-01", trained["bundle"])
