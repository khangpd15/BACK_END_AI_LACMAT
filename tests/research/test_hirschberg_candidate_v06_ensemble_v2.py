from pathlib import Path

import joblib
import numpy as np
import pytest

from app.services.eye_crop_geometry_service import (
    GEOMETRIC_FEATURE_NAMES,
    extract_eye_crop_geometric_features,
)


MODEL_PATH = Path("app/models/research/hirschberg_candidate_v0.6_ensemble_v2.joblib")
BRANCH_V02_PATH = Path("app/models/research/hirschberg_branch_v02_eyecrop.joblib")
EVAL_PATH = Path("reports/hirschberg_candidate_v0.6_ensemble_v2_eval.json")
CARD_PATH = Path("reports/hirschberg_candidate_v0.6_ensemble_v2_model_card.md")


def test_candidate_v06_v2_artifact_contract():
    assert MODEL_PATH.is_file(), f"Model artifact missing at {MODEL_PATH}"
    bundle = joblib.load(MODEL_PATH)

    assert bundle["model_id"] == "hirschberg-candidate-v0.6-ensemble-v2"
    assert bundle["status"] in {"production_runtime", "research_candidate"}
    assert bundle["version"] == "v0.6_ensemble_v2"
    assert bundle["is_production"] is True
    assert bundle["classes"] == ["esotropia", "exotropia", "normal"]
    assert bundle["primary_pipeline"] is not None
    assert bundle["branch_v02_pipeline"] is not None
    assert bundle["fusion_policy"]["type"] == "prior_normalized_weighted"
    assert bundle["fusion_policy"]["conf_threshold"] > 0.0
    assert bundle["fusion_policy"]["margin_threshold"] >= 0.0
    assert bundle["fusion_policy"]["abstain_status"] == "UNCERTAIN"
    assert "non_clinical_declaration" in bundle


def test_predict_hirschberg_production_service():
    from app.services.hirschberg_ai_service import predict_hirschberg, DEFAULT_MODEL_PATH
    import cv2
    assert str(DEFAULT_MODEL_PATH).endswith("hirschberg_candidate_v0.6_ensemble_v2.joblib")

    # Test with synthetic eye crop image
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    res = predict_hirschberg(img)

    assert res.get("status") in {"PREDICTED", "INCONCLUSIVE"}
    assert res.get("modelId") in {
        "hirschberg-candidate-v0.6-ensemble-v2",
        "hirschberg-candidate-v0.5-safe",
    }
    if res.get("fallbackReason"):
        assert res["modelId"] == "hirschberg-candidate-v0.5-safe"
        assert res["fallbackFromModelId"] == "hirschberg-candidate-v0.6-ensemble-v2"
    assert "evaluationSummary" not in res
    probs = res.get("probabilities")
    assert probs is not None
    assert set(probs.keys()) == {"esotropia", "exotropia", "normal"}



def test_branch_v02_eyecrop_contract():
    assert BRANCH_V02_PATH.is_file(), f"Branch v0.2 artifact missing at {BRANCH_V02_PATH}"
    branch = joblib.load(BRANCH_V02_PATH)

    assert branch["model_id"] == "hirschberg-branch-v0.2-eyecrop"
    assert branch["classes"] == ["esotropia", "exotropia", "normal"]
    assert len(branch["geometric_feature_names"]) == 19
    assert branch["pipeline"] is not None


def test_eye_crop_geometry_service_synthetic():
    # Test on a synthetic 224x224 image
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Draw two dark circles for pupils/irises
    cv2_circle = True
    import cv2
    cv2.circle(img, (56, 112), 20, (30, 30, 30), -1)
    cv2.circle(img, (168, 112), 20, (30, 30, 30), -1)
    # Draw light glints
    cv2.circle(img, (58, 112), 3, (255, 255, 255), -1)
    cv2.circle(img, (166, 112), 3, (255, 255, 255), -1)

    feats = extract_eye_crop_geometric_features(img)
    assert isinstance(feats, dict)
    for name in GEOMETRIC_FEATURE_NAMES:
        assert name in feats
        if feats[name] is not None:
            assert np.isfinite(feats[name])
    assert feats["dx_left_mm"] is None
    assert feats["intercanthal_distance_mm"] is None
    assert feats["is_likely_pseudostrabismus_flag"] is None


def test_geometry_uses_explicit_measured_scale_only():
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    import cv2
    cv2.circle(img, (56, 112), 20, (30, 30, 30), -1)
    cv2.circle(img, (168, 112), 20, (30, 30, 30), -1)
    cv2.circle(img, (58, 112), 3, (255, 255, 255), -1)
    cv2.circle(img, (166, 112), 3, (255, 255, 255), -1)
    feats = extract_eye_crop_geometric_features(img, iris_diameter_mm=11.0)
    assert feats["intercanthal_distance_mm"] is not None


def test_effnet_refuses_random_initialized_fallback(monkeypatch):
    import sys
    import types
    from app.services.hirschberg_ai_service import EfficientNetFeatureExtractor

    calls = []
    fake_timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: calls.append(kwargs) or (_ for _ in ()).throw(OSError("weights missing"))
    )
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    fake_transforms = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.__path__ = []
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=object()))
    with pytest.raises(RuntimeError, match="refusing random initialization"):
        EfficientNetFeatureExtractor()
    assert calls == [{"pretrained": True, "num_classes": 0}]


def test_candidate_v06_v2_reports_exist():
    assert EVAL_PATH.is_file(), f"Eval report missing at {EVAL_PATH}"
    assert CARD_PATH.is_file(), f"Model card missing at {CARD_PATH}"
