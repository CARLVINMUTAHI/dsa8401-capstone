"""
tests/test_api.py
=================
Unit and integration tests for the FastAPI prediction service (app/main.py).
Uses httpx TestClient — no live models required; models are monkey-patched
with tiny sklearn pipelines trained on synthetic data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import app.main as api_module  # noqa: E402

FEATURE_COLS = api_module.FEATURE_COLS
N_FEATURES = len(FEATURE_COLS)   # must be 19


def _make_tiny_pipeline(seed: int = 0) -> dict:
    """Train a 20-sample logistic regression — fast but realistic pipeline."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, N_FEATURES))
    y = rng.integers(0, 2, 40)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=seed)),
    ])
    pipe.fit(X, y)
    return {"pipeline": pipe, "model_name": "LogisticRegression_test"}


@pytest.fixture(autouse=True)
def patch_pipelines():
    """Replace loaded pkl models with toy pipelines for every test."""
    original = api_module.pipelines.copy()
    api_module.pipelines.update({
        "stunted": _make_tiny_pipeline(0),
        "wasted": _make_tiny_pipeline(1),
        "underweight": _make_tiny_pipeline(2),
    })
    yield
    api_module.pipelines.clear()
    api_module.pipelines.update(original)


@pytest.fixture
def client():
    return TestClient(api_module.app)


# ── VALID CHILD INPUT FIXTURE ─────────────────────────────────────────────────

@pytest.fixture
def valid_child():
    return {
        "child_age_months": 24,
        "child_sex": 1,
        "birth_order": 2,
        "prev_birth_interval": 30.0,
        "size_at_birth": 3,
        "birth_weight_kg": 3.1,
        "diarrhea_2wk": 0,
        "fever_2wk": 1,
        "breastfeeding": 1,
        "maternal_edu": 1,
        "maternal_bmi": 22.5,
        "wealth_index": 2,
        "under5_in_hh": 3,
        "hh_head_sex": 1,
        "hh_head_age": 35.0,
        "electricity": 0,
        "residence_urban": 0,
        "wash_score": 1,
        "county": 23,
    }


# ── TESTS: GET /health ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_all_models_ready(self, client):
        data = client.get("/health").json()
        assert data["all_models_ready"] is True

    def test_health_three_models_listed(self, client):
        data = client.get("/health").json()
        assert len(data["models_loaded"]) == 3

    def test_health_model_names(self, client):
        data = client.get("/health").json()
        assert set(data["models_loaded"]) == {"stunted", "wasted", "underweight"}


# ── TESTS: GET / (root) ───────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_message(self, client):
        assert "message" in client.get("/").json()


# ── TESTS: POST /predict — valid input ───────────────────────────────────────

class TestPredictValid:
    def test_200_response(self, client, valid_child):
        resp = client.post("/predict", json=valid_child)
        assert resp.status_code == 200

    def test_response_has_predictions(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        assert "predictions" in data

    def test_predictions_has_all_outcomes(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        preds = data["predictions"]
        assert "stunted" in preds
        assert "wasted" in preds
        assert "underweight" in preds

    def test_each_outcome_has_probability_and_flag(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        for outcome in ["stunted", "wasted", "underweight"]:
            assert "probability" in data["predictions"][outcome]
            assert "flag" in data["predictions"][outcome]

    def test_probabilities_between_0_and_1(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        for outcome in ["stunted", "wasted", "underweight"]:
            p = data["predictions"][outcome]["probability"]
            assert 0.0 <= p <= 1.0, f"{outcome} probability {p} out of range"

    def test_flags_are_bool(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        for outcome in ["stunted", "wasted", "underweight"]:
            assert isinstance(data["predictions"][outcome]["flag"], bool)

    def test_county_echoed(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        assert data["county"] == valid_child["county"]

    def test_model_version_present(self, client, valid_child):
        data = client.post("/predict", json=valid_child).json()
        assert "model_version" in data

    def test_deterministic(self, client, valid_child):
        """Same input produces same output."""
        r1 = client.post("/predict", json=valid_child).json()
        r2 = client.post("/predict", json=valid_child).json()
        for outcome in ["stunted", "wasted", "underweight"]:
            assert (r1["predictions"][outcome]["probability"] ==
                    r2["predictions"][outcome]["probability"])

    def test_different_county(self, client, valid_child):
        """Changing county produces a valid response."""
        child2 = dict(valid_child)
        child2["county"] = 1
        r2 = client.post("/predict", json=child2).json()
        for outcome in ["stunted", "wasted", "underweight"]:
            p = r2["predictions"][outcome]["probability"]
            assert 0.0 <= p <= 1.0


# ── TESTS: POST /predict — invalid input ─────────────────────────────────────

class TestPredictInvalid:
    def test_missing_field_422(self, client, valid_child):
        payload = dict(valid_child)
        del payload["child_age_months"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_age_above_max_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["child_age_months"] = 60
        assert client.post("/predict", json=payload).status_code == 422

    def test_county_out_of_range_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["county"] = 48
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_age_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["child_age_months"] = -1
        assert client.post("/predict", json=payload).status_code == 422

    def test_wealth_index_zero_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["wealth_index"] = 0
        assert client.post("/predict", json=payload).status_code == 422

    def test_empty_body_422(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_child_sex_zero_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["child_sex"] = 0
        assert client.post("/predict", json=payload).status_code == 422

    def test_breastfeeding_zero_422(self, client, valid_child):
        payload = dict(valid_child)
        payload["breastfeeding"] = 0
        assert client.post("/predict", json=payload).status_code == 422


# ── TESTS: POST /predict/batch ────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_single_child(self, client, valid_child):
        resp = client.post("/predict/batch", json=[valid_child])
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_batch_multiple_children(self, client, valid_child):
        payload = [valid_child, dict(valid_child | {"county": 5})]
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_batch_all_have_scores(self, client, valid_child):
        payload = [dict(valid_child | {"county": i + 1}) for i in range(5)]
        results = client.post("/predict/batch", json=payload).json()
        for r in results:
            assert "predictions" in r
            assert 0.0 <= r["predictions"]["stunted"]["probability"] <= 1.0

    def test_batch_over_limit_400(self, client, valid_child):
        payload = [valid_child] * 1001
        assert client.post("/predict/batch", json=payload).status_code == 400

    def test_batch_empty_list_200(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 200
        assert resp.json() == []

    def test_batch_one_invalid_422(self, client, valid_child):
        bad_child = dict(valid_child)
        bad_child["county"] = 99
        resp = client.post("/predict/batch", json=[valid_child, bad_child])
        assert resp.status_code == 422
