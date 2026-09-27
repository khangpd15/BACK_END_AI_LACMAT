"""Unit tests for Feature Contract and Shared Feature Extraction (Phase 3).

Verifies:
1. Feature contract specifications integrity (REQUIRED, OPTIONAL, UNAVAILABLE)
2. Contract feature extraction mathematical correctness
3. Feature vector validation against contract requirements
4. Incomplete feature vector detection (returns missing feature names)
5. RemiCare processed features compatibility JSON schema
"""

import os
import pytest

from app.services.feature_contract import (
    FEATURE_SPECS,
    OPTIONAL_FEATURES,
    REQUIRED_SHARED_FEATURES,
    UNAVAILABLE_FEATURES,
    extract_contract_features,
    validate_feature_vector,
)


def test_feature_contract_specifications():
    """Verify that all features in FEATURE_SPECS have required metadata."""
    assert len(REQUIRED_SHARED_FEATURES) == 30
    assert len(OPTIONAL_FEATURES) >= 2
    assert len(UNAVAILABLE_FEATURES) >= 4

    for name, spec in FEATURE_SPECS.items():
        assert spec.name == name
        assert len(spec.formula) > 0
        assert len(spec.source) > 0
        assert spec.dtype in ("float64", "int64")
        assert len(spec.description) > 0
        assert spec.category in ("REQUIRED", "OPTIONAL", "UNAVAILABLE")


def test_contract_feature_extraction():
    """Verify mathematical calculation of contract features on known synthetic values."""
    samples = [
        {"t": 0.00, "leftX": 0.40, "leftY": 0.50, "leftValid": True, "rightX": 0.55, "rightY": 0.52, "rightValid": True},
        {"t": 0.03, "leftX": 0.41, "leftY": 0.51, "leftValid": True, "rightX": 0.57, "rightY": 0.53, "rightValid": True},
        {"t": 0.06, "leftX": 0.40, "leftY": 0.50, "leftValid": True, "rightX": 0.56, "rightY": 0.52, "rightValid": True},
    ]

    feats = extract_contract_features(samples)

    # Disparity checks: dx values are [0.15, 0.16, 0.16]
    assert feats["meanDeltaX"] == pytest.approx(0.156667, abs=1e-4)
    assert feats["medianDeltaX"] == pytest.approx(0.16, abs=1e-4)
    assert feats["minDeltaX"] == pytest.approx(0.15, abs=1e-4)
    assert feats["maxDeltaX"] == pytest.approx(0.16, abs=1e-4)
    assert feats["rangeDeltaX"] == pytest.approx(0.01, abs=1e-4)

    # Validity checks
    assert feats["leftValidRatio"] == pytest.approx(1.0)
    assert feats["rightValidRatio"] == pytest.approx(1.0)
    assert feats["bothValidRatio"] == pytest.approx(1.0)


def test_validate_feature_vector_all_present():
    """Verify that complete feature vector passes contract validation."""
    samples = [
        {"t": 0.00, "leftX": 0.40, "leftY": 0.50, "leftValid": True, "rightX": 0.55, "rightY": 0.52, "rightValid": True},
        {"t": 0.03, "leftX": 0.40, "leftY": 0.50, "leftValid": True, "rightX": 0.55, "rightY": 0.52, "rightValid": True},
    ]
    feats = extract_contract_features(samples)
    is_compatible, available, missing = validate_feature_vector(feats)

    assert is_compatible is True
    assert len(missing) == 0
    assert len(available) == len(REQUIRED_SHARED_FEATURES)


def test_validate_feature_vector_missing_detected():
    """Verify that an incomplete feature dictionary is flagged with missing names."""
    partial_feats = {
        "meanDeltaX": 0.15,
        "meanDeltaY": 0.02,
        # Intentionally missing the rest
    }
    is_compatible, available, missing = validate_feature_vector(partial_feats)

    assert is_compatible is False
    assert "meanDeltaX" in available
    assert "stdDeltaX" in missing
    assert "meanLeftVelocity" in missing
    assert len(missing) > 0


def test_remicare_features_json_file():
    """Verify data/processed/remicare_features.json if generated."""
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "remicare_features.json")
    if os.path.exists(json_path):
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["source"] == "REMICARE_WEBCAM"
        assert data["deviceType"] == "MEDIAPIPE_WEBCAM"
        assert data["compatibility"]["compatible"] is True
        assert data["compatibility"]["missingFeatureCount"] == 0
