"""Phase 4.2 Backend Tests for FastAPI Transfer Inference Endpoint.

Covers all 20 required backend test cases:
1. health endpoint
2. valid Cover Test payload
3. missing sampleId
4. invalid cycle
5. invalid coveredEye
6. invalid trackedEye
7. coveredEye == trackedEye
8. non-monotonic timestamp
9. NaN coordinate
10. Infinity coordinate
11. invalid trackingQuality
12. missing samples
13. feature count == 30
14. exact feature order
15. no protocol features
16. no target leakage
17. model inference
18. class probability
19. domainShiftWarning == true
20. clinicalMeaning == null
"""

import copy
import json
import math
import os
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

from app.main import app
from app.services.korean_transfer import (
    FEATURE_ORDER,
    get_korean_transfer_service,
)
from app.services.shared_feature_contract import EXCLUDED_FEATURES

client = TestClient(app)


@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def valid_payload(project_root):
    sample_path = os.path.join(project_root, "data", "raw", "sample.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 1. HEALTH CHECK ENDPOINT
# =============================================================================

def test_health_endpoint():
    """1. Test GET /health endpoint returns status ok and service name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "remicare-strabismus-ai"


# =============================================================================
# 2. VALID COVER TEST PAYLOAD
# =============================================================================

def test_valid_cover_test_payload(valid_payload):
    """2. Test POST /api/v1/transfer/strabismus with valid Cover Test payload."""
    response = client.post("/api/v1/transfer/strabismus", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sampleId"] == valid_payload["sampleId"]
    assert data["status"] == "TRANSFER_EXPERIMENT"
    assert data["inputCompatible"] is True
    assert data["prediction"] in ["NORMAL", "STRABISMUS"]
    assert "classProbability" in data
    assert data["domainShiftWarning"] is True
    assert data["clinicalMeaning"] is None


# =============================================================================
# 3 - 12. DATA INTEGRITY & VALIDATION TESTS
# =============================================================================

def test_missing_sample_id(valid_payload):
    """3. Missing sampleId must trigger 422 Unprocessable Entity."""
    payload = copy.deepcopy(valid_payload)
    del payload["sampleId"]
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_invalid_cycle(valid_payload):
    """4. Empty or missing cycles list must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"] = []
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_invalid_covered_eye(valid_payload):
    """5. Invalid coveredEye (not LEFT or RIGHT) must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["coveredEye"] = "CENTER"
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_invalid_tracked_eye(valid_payload):
    """6. Invalid trackedEye (not LEFT or RIGHT) must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["trackedEye"] = "UNKNOWN"
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_covered_eye_equals_tracked_eye(valid_payload):
    """7. coveredEye == trackedEye must be rejected with 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["coveredEye"] = "LEFT"
    payload["cycles"][0]["trackedEye"] = "LEFT"
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_non_monotonic_timestamp(valid_payload):
    """8. Non-monotonic timestamp within cycle must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    # Reverse timestamps for cycle 0
    payload["cycles"][0]["samples"][1]["t"] = payload["cycles"][0]["samples"][0]["t"] - 50.0
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_nan_coordinate(valid_payload):
    """9. NaN coordinate on valid frame must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["samples"][0]["leftValid"] = True
    payload["cycles"][0]["samples"][0]["leftX"] = float("nan")
    raw_content = json.dumps(payload, allow_nan=True)
    response = client.post("/api/v1/transfer/strabismus", content=raw_content, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_infinity_coordinate(valid_payload):
    """10. Infinity coordinate on valid frame must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["samples"][0]["leftValid"] = True
    payload["cycles"][0]["samples"][0]["leftX"] = float("inf")
    raw_content = json.dumps(payload, allow_nan=True)
    response = client.post("/api/v1/transfer/strabismus", content=raw_content, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_invalid_tracking_quality(valid_payload):
    """11. Out-of-range or invalid trackingQuality must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["samples"][0]["trackingQuality"] = 999.0
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


def test_missing_samples_in_cycle(valid_payload):
    """12. Cycle with empty samples list must trigger 422."""
    payload = copy.deepcopy(valid_payload)
    payload["cycles"][0]["samples"] = []
    response = client.post("/api/v1/transfer/strabismus", json=payload)
    assert response.status_code == 422


# =============================================================================
# 13 - 16. FEATURE CONTRACT & LEAKAGE ASSERTIONS
# =============================================================================

def test_feature_count_equals_30():
    """13. Ensure canonical FEATURE_ORDER contains exactly 30 features."""
    assert len(FEATURE_ORDER) == 30


def test_exact_feature_order():
    """14. Ensure exact canonical feature ordering matches specification."""
    expected_order = [
        "leftValidRatio", "rightValidRatio", "bothValidRatio",
        "meanDeltaX", "medianDeltaX", "stdDeltaX", "minDeltaX", "maxDeltaX", "rangeDeltaX", "meanAbsDeltaX",
        "meanDeltaY", "medianDeltaY", "stdDeltaY", "minDeltaY", "maxDeltaY", "rangeDeltaY", "meanAbsDeltaY",
        "meanLeftX", "stdLeftX", "meanLeftY", "stdLeftY",
        "meanRightX", "stdRightX", "meanRightY", "stdRightY",
        "meanLeftVelocity", "peakLeftVelocity", "meanRightVelocity", "peakRightVelocity", "velocityDisparity"
    ]
    assert FEATURE_ORDER == expected_order


def test_no_protocol_features():
    """15. Ensure protocol/hardware dependent features are strictly excluded."""
    for excluded in EXCLUDED_FEATURES:
        assert excluded not in FEATURE_ORDER


def test_no_target_leakage():
    """16. Ensure target labels and diagnoses are excluded from feature contract."""
    forbidden = ["label", "label_name", "target", "diagnosis", "clinicalLabel", "groundTruth"]
    for f in forbidden:
        assert f not in FEATURE_ORDER


# =============================================================================
# 17 - 20. INFERENCE & RESPONSE SCHEMA TESTS
# =============================================================================

def test_model_inference(valid_payload):
    """17. Test model executes inference returning valid binary classification."""
    svc = get_korean_transfer_service()
    from app.schemas import ScreeningRequest
    req = ScreeningRequest.model_validate(valid_payload)
    res = svc.predict_transfer(req)

    assert res.prediction in ["NORMAL", "STRABISMUS"]
    assert res.status == "TRANSFER_EXPERIMENT"


def test_class_probability_schema(valid_payload):
    """18. Test classProbability returned with NORMAL and STRABISMUS keys summing to ~1.0."""
    response = client.post("/api/v1/transfer/strabismus", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert "classProbability" in data
    prob = data["classProbability"]
    assert "NORMAL" in prob
    assert "STRABISMUS" in prob
    assert 0.0 <= prob["NORMAL"] <= 1.0
    assert 0.0 <= prob["STRABISMUS"] <= 1.0
    assert math.isclose(prob["NORMAL"] + prob["STRABISMUS"], 1.0, abs_tol=0.01)


def test_domain_shift_warning_flag(valid_payload):
    """19. Test domainShiftWarning is explicitly true with source and target tags."""
    response = client.post("/api/v1/transfer/strabismus", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["domainShiftWarning"] is True
    assert "domainShift" in data
    assert data["domainShift"]["source"] == "KOREAN_INFRARED_EYE_TRACKER"
    assert data["domainShift"]["target"] == "REMICARE_WEBCAM_MEDIAPIPE"
    assert data["domainShift"]["warning"] is True


def test_clinical_meaning_is_null(valid_payload):
    """20. Test clinicalMeaning is explicitly null/None."""
    response = client.post("/api/v1/transfer/strabismus", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["clinicalMeaning"] is None
    assert "not a diagnosis" in data["notice"].lower()
