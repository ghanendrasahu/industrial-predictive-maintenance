"""API contract: readiness, assessment payload and the API-key gate."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setattr(main.settings, "DATA_PATH", str(trained["data_csv"]))
    monkeypatch.setattr(main.settings, "MODEL_PATH", str(trained["bundle_path"]))
    monkeypatch.setattr(main.settings, "API_KEY", "")
    with TestClient(main.app) as live:
        yield live


def test_health_reports_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["machines"] == 3


def test_predict_returns_decision_payload(client):
    response = client.get("/api/predict/M-01")
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == "M-01"
    assert body["risk_band"] in {"CRITICAL", "WATCH", "HEALTHY"}
    assert body["recommended_action"]
    assert body["primary_indicators"]


def test_fleet_endpoint_is_ranked(client):
    response = client.get("/api/machines")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    probabilities = [item["failure_probability"] for item in body["machines"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_unknown_machine_returns_404(client):
    assert client.get("/api/predict/M-99").status_code == 404


def test_api_key_gate(client, monkeypatch):
    monkeypatch.setattr(main.settings, "API_KEY", "secret-123")
    assert client.get("/api/predict/M-01").status_code == 401
    ok = client.get("/api/predict/M-01", headers={"X-API-Key": "secret-123"})
    assert ok.status_code == 200
