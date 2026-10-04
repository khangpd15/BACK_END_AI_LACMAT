from pathlib import Path
import joblib
import numpy as np
import pytest

pytest.importorskip("lightgbm")

from scripts.train_hirschberg_research_model import (
    ALL_FEATURE_NAMES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_MODEL_CARD_OUTPUT,
    DEFAULT_EVAL_OUTPUT,
)


def test_candidate_v02_artifact_exists_and_loads():
    assert DEFAULT_MODEL_OUTPUT.is_file(), f"Model artifact missing at {DEFAULT_MODEL_OUTPUT}"
    bundle = joblib.load(DEFAULT_MODEL_OUTPUT)

    assert isinstance(bundle, dict)
    assert bundle["model_id"] == "hirschberg-candidate-v0.2"
    assert bundle["status"] == "research_candidate"
    assert bundle["is_production"] is False
    assert bundle["classes"] == ["normal", "esotropia", "exotropia", "pseudostrabismus", "poor_quality"]
    assert bundle["vision_backbone"] == "efficientnet_b0"
    assert len(bundle["feature_names"]) == len(ALL_FEATURE_NAMES)
    assert "pipeline" in bundle
    assert "metrics" in bundle

    pipeline = bundle["pipeline"]
    # Test inference on dummy feature vectors
    dummy_x = np.random.randn(3, len(ALL_FEATURE_NAMES)).astype(np.float32)
    preds = pipeline.predict(dummy_x)
    assert len(preds) == 3
    assert all(p in range(5) for p in preds)

    probs = pipeline.predict_proba(dummy_x)
    assert probs.shape == (3, 5)
    assert np.allclose(np.sum(probs, axis=1), [1.0, 1.0, 1.0])


def test_candidate_v02_reports_exist():
    assert DEFAULT_MODEL_CARD_OUTPUT.is_file(), f"Model card missing at {DEFAULT_MODEL_CARD_OUTPUT}"
    card_content = DEFAULT_MODEL_CARD_OUTPUT.read_text(encoding="utf-8")
    assert "hirschberg-candidate-v0.2" in card_content
    assert "Esotropia Sensitivity" in card_content
    assert "Severe Confusion Rate" in card_content

    assert DEFAULT_EVAL_OUTPUT.is_file(), f"Eval JSON missing at {DEFAULT_EVAL_OUTPUT}"
