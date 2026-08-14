from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import LoanInference, app

client = TestClient(app)


@pytest.fixture
def mock_predictor(monkeypatch):
    """Fixture to mock the predictor."""
    mock = MagicMock(spec=LoanInference)
    mock.predict.return_value = {
        "default_prediction": 0,
        "default_probability": 0.1234,
        "risk_status": "Low Risk (Auto-Approve)",
    }
    monkeypatch.setattr("src.api.predictor", mock)
    return mock


def test_predict_happy_path(mock_predictor):
    payload = {
        "credit_score": 750,
        "annual_income": 80000,
        "loan_amount": 20000,
        "dti_ratio": 0.25,
        "employment_tenure": 5,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"]["default_prediction"] == 0
    assert data["decision"]["default_probability"] == 0.1234


def test_predict_failure_path_invalid_input(mock_predictor):
    payload = {
        "credit_score": 150,  # Invalid, below 300
        "annual_income": 80000,
        "loan_amount": 20000,
        "dti_ratio": 0.25,
        "employment_tenure": 5,
    }
    response = client.post("/predict", json=payload)

    # FastAPI should return 422 Unprocessable Entity for Pydantic validation errors
    assert response.status_code == 422


def test_predict_failure_path_uninitialized_model(monkeypatch):
    monkeypatch.setattr("src.api.predictor", None)
    payload = {
        "credit_score": 750,
        "annual_income": 80000,
        "loan_amount": 20000,
        "dti_ratio": 0.25,
        "employment_tenure": 5,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model core uninitialized"
