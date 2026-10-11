"""API contract tests against the committed model (models/model.joblib)."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from heartrisk import api
from heartrisk.monitoring import JsonFormatter
from heartrisk.schemas import EXAMPLE_PATIENT


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:  # context manager runs the lifespan (model load)
        yield c


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"]


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=EXAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("disease" if body["prediction"] else "no disease")
    assert 0 <= body["probability_disease"] <= 1
    assert body["confidence"] == pytest.approx(max(body["probability_disease"], 1 - body["probability_disease"]),
                                               abs=1e-4)
    assert body["risk_band"] in ("low", "moderate", "high")
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_api_matches_offline_pipeline(client, clean_df):
    """Serving must give exactly the probabilities of the saved pipeline."""
    row = clean_df.dropna().iloc[0]
    payload = {k: (int(v) if float(v).is_integer() and k != "oldpeak" else float(v))
               for k, v in row.drop("target").items()}
    expected = api.service.pipeline.predict_proba(clean_df.dropna().iloc[[0]].drop(columns="target"))[0, 1]
    assert client.post("/predict", json=payload).json()["probability_disease"] == pytest.approx(expected, abs=1e-4)


def test_optional_fields_can_be_omitted(client):
    payload = {k: v for k, v in EXAMPLE_PATIENT.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_clear_risk_profiles(client):
    low = {**EXAMPLE_PATIENT, "age": 40, "sex": 0, "cp": 3, "thalach": 180, "exang": 0, "oldpeak": 0.0,
           "slope": 1, "ca": 0, "thal": 3}
    high = {**EXAMPLE_PATIENT, "cp": 4, "thalach": 110, "exang": 1, "oldpeak": 3.5, "slope": 2, "ca": 3, "thal": 7}
    assert client.post("/predict", json=low).json()["risk_band"] == "low"
    assert client.post("/predict", json=high).json()["risk_band"] == "high"


@pytest.mark.parametrize("field, value", [("age", 630), ("cp", 7), ("chol", -5), ("sex", "male")])
def test_invalid_values_are_rejected(client, field, value):
    r = client.post("/predict", json={**EXAMPLE_PATIENT, field: value})
    assert r.status_code == 422
    assert field in str(r.json()["detail"])


def test_missing_required_field_rejected(client):
    payload = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**EXAMPLE_PATIENT, "weight": 80}).status_code == 422


def test_batch_prediction(client):
    r = client.post("/predict/batch", json={"patients": [EXAMPLE_PATIENT] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
    assert client.post("/predict/batch", json={"patients": [EXAMPLE_PATIENT] * 101}).status_code == 422


def test_model_endpoint_exposes_metadata(client):
    body = client.get("/model").json()
    assert body["model_name"] and "test_roc_auc" in body["metrics"]


def test_request_id_is_propagated(client):
    assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_response_names_the_serving_replica(client):
    import socket

    assert client.get("/health").headers["X-Served-By"] == socket.gethostname()


def test_metrics_endpoint_counts_requests(client):
    client.post("/predict", json=EXAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert 'heartrisk_http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "heartrisk_predictions_total" in text
    assert "heartrisk_prediction_probability_bucket" in text
    assert "heartrisk_model_info" in text


def test_requests_are_logged_as_json(client, caplog):
    caplog.set_level(logging.INFO, logger="heartrisk.access")
    client.get("/health")
    record = [r for r in caplog.records if r.name == "heartrisk.access"][-1]
    entry = json.loads(JsonFormatter().format(record))
    assert entry["msg"] == "request" and entry["path"] == "/health" and entry["status"] == 200
    assert {"request_id", "latency_ms", "method"} <= entry.keys()


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(api.service, "pipeline", None)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(api.service, "pipeline", None)  # lifespan reloads; unset again
        assert c.post("/predict", json=EXAMPLE_PATIENT).status_code == 503
        assert c.get("/health").json()["status"] == "degraded"
