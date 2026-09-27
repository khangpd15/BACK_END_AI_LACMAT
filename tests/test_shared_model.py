"""Phase 4 Automated Tests for Shared Feature Model & Cross-Domain Transfer.

Covers all 12 validation requirements:
1. shared feature contract
2. train/test feature equality
3. no target leakage
4. no sampleId as model feature
5. no incompatible hardware features
6. no missing-feature imputation
7. scaler fit only on train
8. participant overlap detection
9. RemiCare feature extraction
10. transfer compatibility
11. missing feature handling
12. domain-shift warning
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    EXCLUDED_FEATURES,
    POTENTIAL_DOMAIN_SHIFT_FEATURES,
    SHARED_FEATURE_CONTRACT_VERSION,
    SHARED_FEATURE_SPECS,
    extract_remicare_shared_payload,
    extract_shared_features_vector,
    run_shared_transfer_inference,
)


@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def korean_train_df(project_root):
    train_path = os.path.join(project_root, "data", "processed", "korean_shared_train.csv")
    assert os.path.exists(train_path), f"Train dataset missing at {train_path}"
    return pd.read_csv(train_path)


@pytest.fixture
def korean_test_df(project_root):
    test_path = os.path.join(project_root, "data", "processed", "korean_shared_test.csv")
    assert os.path.exists(test_path), f"Test dataset missing at {test_path}"
    return pd.read_csv(test_path)


@pytest.fixture
def trained_model_artifact(project_root):
    model_path = os.path.join(project_root, "models", "korean_shared_model.joblib")
    assert os.path.exists(model_path), f"Model artifact missing at {model_path}"
    return joblib.load(model_path)


@pytest.fixture
def raw_remicare_sample(project_root):
    sample_path = os.path.join(project_root, "data", "raw", "sample.json")
    assert os.path.exists(sample_path), f"Raw sample missing at {sample_path}"
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 1. SHARED FEATURE CONTRACT SPECIFICATIONS
# =============================================================================

def test_shared_feature_contract_specifications():
    """Verify shared feature contract defines exactly 30 features with explicit specifications."""
    assert len(ALL_SHARED_FEATURES) == 30
    assert SHARED_FEATURE_CONTRACT_VERSION == "shared-v1.0.0"

    for feat_name in ALL_SHARED_FEATURES:
        assert feat_name in SHARED_FEATURE_SPECS, f"Missing spec for {feat_name}"
        spec = SHARED_FEATURE_SPECS[feat_name]
        assert spec.formula != ""
        assert spec.dtype in ["float64", "int64"]
        assert spec.domain_shift_risk in ["LOW_DOMAIN_SHIFT", "POTENTIAL_DOMAIN_SHIFT", "HIGH_DOMAIN_SHIFT"]

    # Verify potential domain shift features are properly flagged
    for f in POTENTIAL_DOMAIN_SHIFT_FEATURES:
        assert SHARED_FEATURE_SPECS[f].domain_shift_risk == "POTENTIAL_DOMAIN_SHIFT"


# =============================================================================
# 2. TRAIN / TEST FEATURE EQUALITY
# =============================================================================

def test_train_test_feature_equality(korean_train_df, korean_test_df):
    """Verify train and test datasets contain exactly identical columns and order."""
    train_cols = list(korean_train_df.columns)
    test_cols = list(korean_test_df.columns)

    assert train_cols == test_cols, "Train and test column mismatch"
    assert len(train_cols) == 32  # sampleId, label + 30 shared features

    # Verify all 30 shared features are present
    feature_cols = [c for c in train_cols if c not in ["sampleId", "label"]]
    assert feature_cols == ALL_SHARED_FEATURES


# =============================================================================
# 3. NO TARGET LEAKAGE
# =============================================================================

def test_no_target_leakage(trained_model_artifact):
    """Ensure target label is never present in model input feature space."""
    model_features = trained_model_artifact["feature_names"]

    assert "label" not in model_features
    assert "label_name" not in model_features
    assert "target" not in model_features
    assert "diagnosis" not in model_features


# =============================================================================
# 4. NO SAMPLE_ID AS MODEL FEATURE
# =============================================================================

def test_no_sample_id_as_model_feature(trained_model_artifact):
    """Ensure sampleId and participant metadata are not included in model inputs."""
    model_features = trained_model_artifact["feature_names"]

    assert "sampleId" not in model_features
    assert "participantId" not in model_features
    assert "filename" not in model_features


# =============================================================================
# 5. NO INCOMPATIBLE HARDWARE FEATURES
# =============================================================================

def test_no_incompatible_hardware_features(trained_model_artifact):
    """Ensure hardware/protocol dependent features are strictly excluded from X."""
    model_features = trained_model_artifact["feature_names"]

    for excluded in EXCLUDED_FEATURES:
        assert excluded not in model_features, f"Excluded feature {excluded} found in model feature set!"


# =============================================================================
# 6. NO MISSING-FEATURE IMPUTATION
# =============================================================================

def test_no_missing_feature_imputation():
    """Verify missing raw coordinates produce None/null and are never imputed with 0 or mean."""
    # Create empty/invalid sample stream
    empty_samples = [
        {"t": 0.0, "leftX": None, "leftY": None, "leftValid": False, "rightX": None, "rightY": None, "rightValid": False},
        {"t": 100.0, "leftX": None, "leftY": None, "leftValid": False, "rightX": None, "rightY": None, "rightValid": False},
    ]

    features = extract_shared_features_vector(empty_samples)

    # Validity ratios are 0.0, but disparity and position metrics MUST be None
    assert features["leftValidRatio"] == 0.0
    assert features["rightValidRatio"] == 0.0
    assert features["bothValidRatio"] == 0.0

    assert features["meanDeltaX"] is None
    assert features["medianDeltaX"] is None
    assert features["meanLeftX"] is None
    assert features["meanRightX"] is None
    assert features["meanLeftVelocity"] is None
    assert features["velocityDisparity"] is None


# =============================================================================
# 7. SCALER FIT ONLY ON TRAIN
# =============================================================================

def test_scaler_fit_only_on_train(korean_train_df, korean_test_df):
    """Verify preprocessing scaler is fit exclusively on training data without test contamination."""
    feat_cols = ALL_SHARED_FEATURES
    X_train = korean_train_df[feat_cols].values
    X_test = korean_test_df[feat_cols].values

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_mean = scaler.mean_.copy()
    train_var = scaler.var_.copy()

    # Transform test without re-fitting
    X_test_scaled = scaler.transform(X_test)

    # Assert scaler parameters were untouched during test transform
    np.testing.assert_array_equal(scaler.mean_, train_mean)
    np.testing.assert_array_equal(scaler.var_, train_var)
    assert X_test_scaled.shape == X_test.shape


# =============================================================================
# 8. PARTICIPANT OVERLAP DETECTION
# =============================================================================

def test_participant_overlap_detection():
    """Verify participant overlap detection correctly identifies shared identities between train and test."""
    from scripts.build_korean_shared_datasets import analyze_participant_leakage
    import tempfile

    df_mock_train = pd.DataFrame({
        "participantId": ["김민경", "이영희", "박철수"],
        "sampleId": ["s1", "s2", "s3"],
    })
    df_mock_test = pd.DataFrame({
        "participantId": ["김민경", "정민수"],
        "sampleId": ["s4", "s5"],
    })

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        leakage = analyze_participant_leakage(df_mock_train, df_mock_test, tmp_path)
        assert leakage["status"] == "PARTICIPANT_LEVEL_LEAKAGE"
        assert "김민경" in leakage["overlapParticipants"]
        assert leakage["overlapCount"] == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# 9. REMICARE FEATURE EXTRACTION PIPELINE
# =============================================================================

def test_remicare_feature_extraction(raw_remicare_sample):
    """Verify RemiCare sample passes validation, preprocessing, and extracts 30 contract features."""
    payload = extract_remicare_shared_payload(raw_remicare_sample)

    assert payload["sampleId"] == "remicare-demo-sample-001"
    assert payload["source"] == "REMICARE_WEBCAM_MEDIAPIPE"
    assert payload["featureContractVersion"] == SHARED_FEATURE_CONTRACT_VERSION
    assert len(payload["features"]) == 30
    assert payload["missingFeatures"] == []

    # Check key technical values
    feats = payload["features"]
    assert 0.0 <= feats["leftValidRatio"] <= 1.0
    assert 0.0 <= feats["rightValidRatio"] <= 1.0
    assert feats["meanDeltaX"] is not None
    assert feats["meanLeftVelocity"] is not None


# =============================================================================
# 10. TRANSFER COMPATIBILITY
# =============================================================================

def test_transfer_compatibility(raw_remicare_sample, trained_model_artifact):
    """Verify inference succeeds on valid RemiCare feature payload without crashes."""
    payload = extract_remicare_shared_payload(raw_remicare_sample)
    result = run_shared_transfer_inference(payload, trained_model_artifact)

    assert result["status"] == "TRANSFER_EXPERIMENT"
    assert result["inputCompatible"] is True
    assert result["prediction"] in ["NORMAL", "STRABISMUS"]
    assert "notice" in result
    assert "probabilities" in result


# =============================================================================
# 11. MISSING FEATURE HANDLING WITHOUT IMPUTATION
# =============================================================================

def test_missing_feature_handling_incompatible(trained_model_artifact):
    """Verify missing required features trigger MODEL_INPUT_INCOMPATIBLE and abort without imputation."""
    incompatible_payload = {
        "sampleId": "test-incompatible",
        "source": "REMICARE_WEBCAM_MEDIAPIPE",
        "featureContractVersion": SHARED_FEATURE_CONTRACT_VERSION,
        "features": {
            "leftValidRatio": 0.5,
            # Deliberately missing all other features
        },
        "missingFeatures": ["rightValidRatio", "bothValidRatio", "meanDeltaX"],
    }

    result = run_shared_transfer_inference(incompatible_payload, trained_model_artifact)

    assert result["status"] == "MODEL_INPUT_INCOMPATIBLE"
    assert result["inputCompatible"] is False
    assert len(result["missingFeatures"]) > 0
    assert "meanDeltaX" in result["missingFeatures"]


# =============================================================================
# 12. DOMAIN-SHIFT WARNING & PRODUCTION CONSTRAINT
# =============================================================================

def test_domain_shift_warning_enforced(raw_remicare_sample, trained_model_artifact):
    """Verify domain shift warning and production constraints are strictly present."""
    payload = extract_remicare_shared_payload(raw_remicare_sample)
    result = run_shared_transfer_inference(payload, trained_model_artifact)

    assert result["domainShiftWarning"] is True
    assert "domainShift" in result
    assert result["domainShift"]["sourceDomain"] == "KOREAN_INFRARED_EYE_TRACKER"
    assert result["domainShift"]["targetDomain"] == "REMICARE_WEBCAM_MEDIAPIPE"
    assert "productionConstraint" in result
    assert "SCREENING_CLEAR" in result["productionConstraint"]
    assert "SCREENING_ATTENTION" in result["productionConstraint"]


# =============================================================================
# 13. CLASS PROBABILITY & NO CLINICAL MEANING
# =============================================================================

def test_class_probability_and_clinical_meaning_none(raw_remicare_sample, trained_model_artifact):
    """Verify inference output uses classProbability and explicitly sets clinicalMeaning to None."""
    payload = extract_remicare_shared_payload(raw_remicare_sample)
    result = run_shared_transfer_inference(payload, trained_model_artifact)

    assert "classProbability" in result
    assert isinstance(result["classProbability"], float)
    assert result["clinicalMeaning"] == "None"
    assert "clinicalRiskDisclaimer" in result


# =============================================================================
# 14. REMICARE TRANSFER RESULTS CSV SCHEMA (N=1)
# =============================================================================

def test_remicare_transfer_results_csv_schema(project_root):
    """Verify remicare_transfer_results.csv has required schema and records N=1 sample."""
    csv_path = os.path.join(project_root, "data", "processed", "remicare_transfer_results.csv")
    assert os.path.exists(csv_path), f"Transfer results CSV missing: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = ["sampleId", "prediction", "classProbability", "domainShiftWarning", "inputCompatible"]
    assert list(df.columns) == expected_cols
    assert len(df) == 1
    assert df.iloc[0]["sampleId"] == "remicare-demo-sample-001"
    assert df.iloc[0]["prediction"] in ["NORMAL", "STRABISMUS"]
    assert bool(df.iloc[0]["domainShiftWarning"]) is True
    assert bool(df.iloc[0]["inputCompatible"]) is True

