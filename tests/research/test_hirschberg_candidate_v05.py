from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path("app/models/research/hirschberg_candidate_v0.5_safe.joblib")


def test_candidate_v05_safe_artifact_contract():
    assert MODEL_PATH.is_file(), f"Model artifact missing at {MODEL_PATH}"
    bundle = joblib.load(MODEL_PATH)

    assert bundle["model_id"] == "hirschberg-candidate-v0.5-safe"
    assert bundle["status"] == "research_candidate"
    assert bundle["is_production"] is False
    assert bundle["feature_contract"] == "hirschberg_pair_features_v0.5"
    assert bundle["classes"] == ["esotropia", "exotropia", "normal"]
    assert len(bundle["feature_names"]) == 365
    assert bundle["decision_policy"]["type"] == "confidence_margin_abstention"
    assert bundle["decision_policy"]["threshold"] >= 0.0
    assert bundle["decision_policy"]["margin_threshold"] >= 0.0

    pipeline = bundle["pipeline"]
    dummy_x = np.zeros((2, len(bundle["feature_names"])), dtype=np.float32)
    probs = pipeline.predict_proba(dummy_x)
    assert probs.shape == (2, 3)
    assert np.allclose(np.sum(probs, axis=1), [1.0, 1.0])
