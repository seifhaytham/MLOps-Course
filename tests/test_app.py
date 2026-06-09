"""Tests for predictor logic and API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

from src.predictor import _hash_features, is_loaded, load_artifacts, reset
from src.schemas import PredictRequest, PredictResponse
from src.app import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PAYLOAD: dict = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}


@pytest.fixture(autouse=True)
def clear_state():
    """Reset the module-level artifact cache before every test."""
    reset()
    yield
    reset()


@pytest.fixture()
def sample_request() -> PredictRequest:
    return PredictRequest(**SAMPLE_PAYLOAD)


# ── Unit tests (function-level) ───────────────────────────────────────────────

class TestLoadArtifacts:
    def test_raises_when_model_missing(self, tmp_path: Path):
        """RuntimeError when model.pkl does not exist."""
        with pytest.raises(RuntimeError, match="Model artifact not found"):
            load_artifacts(model_path=tmp_path / "missing.pkl", transformer_path=tmp_path / "t.pkl")

    def test_raises_when_transformer_missing(self, tmp_path: Path):
        """RuntimeError when transformer.pkl does not exist."""
        # Create a dummy model file so the model check passes
        fake_model = tmp_path / "model.pkl"
        fake_model.write_bytes(b"")
        with pytest.raises(RuntimeError, match="Transformer artifact not found"):
            load_artifacts(model_path=fake_model, transformer_path=tmp_path / "missing.pkl")

    def test_loads_real_artifacts(self):
        """Happy path: real pkl files load without error."""
        repo_root = Path(__file__).resolve().parents[1]
        load_artifacts(
            model_path=repo_root / "model.pkl",
            transformer_path=repo_root / "transformer.pkl",
        )
        assert is_loaded()


class TestHashFeatures:
    def test_deterministic(self, sample_request: PredictRequest):
        """Same input always produces same hash."""
        assert _hash_features(sample_request) == _hash_features(sample_request)

    def test_different_inputs_differ(self, sample_request: PredictRequest):
        """Different customer data produces different hash."""
        other = sample_request.model_copy(update={"Age": 99})
        assert _hash_features(sample_request) != _hash_features(other)

    def test_hash_length(self, sample_request: PredictRequest):
        """Hash is exactly 12 hex characters."""
        assert len(_hash_features(sample_request)) == 12


class TestPredict:
    def test_returns_valid_response(self, sample_request: PredictRequest):
        """predict() returns a PredictResponse with correct types."""
        repo_root = Path(__file__).resolve().parents[1]
        load_artifacts(
            model_path=repo_root / "model.pkl",
            transformer_path=repo_root / "transformer.pkl",
        )
        result = predict(sample_request)
        assert isinstance(result, PredictResponse)
        assert result.prediction in (0, 1)
        assert 0.0 <= result.probability <= 1.0


# ── Endpoint tests (integration-level) ───────────────────────────────────────

@pytest.fixture()
def client():
    """TestClient with artifacts pre-loaded."""
    from app import app  # import here so logging config is applied
    repo_root = Path(__file__).resolve().parents[1]
    load_artifacts(
        model_path=repo_root / "model.pkl",
        transformer_path=repo_root / "transformer.pkl",
    )
    with TestClient(app=app) as c:
        yield c


class TestHomeEndpoint:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_response_shape(self, client: TestClient):
        body = resp = client.get("/").json()
        assert body["service"] == "churn-prediction-api"
        assert "docs" in body


class TestHealthEndpoint:
    def test_ok_when_loaded(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True

    def test_degraded_when_not_loaded(self):
        """Returns 'degraded' when artifacts are not loaded."""
        reset()  # ensure not loaded for this test
        with TestClient(app=app) as c:
            resp = c.get("/health")
        assert resp.json()["status"] == "degraded"


class TestPredictEndpoint:
    def test_valid_payload_returns_prediction(self, client: TestClient):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0

    def test_invalid_payload_returns_422(self, client: TestClient):
        """Missing required fields → validation error."""
        resp = client.post("/predict", json={"CreditScore": 500})
        assert resp.status_code == 422

    def test_invalid_geography_returns_422(self, client: TestClient):
        bad = {**SAMPLE_PAYLOAD, "Geography": "Egypt"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422