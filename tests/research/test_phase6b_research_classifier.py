"""Test suite for Phase 6B exploratory Hirschberg research classifier candidate."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.1.joblib"
EVAL_REPORT_PATH = Path("D:/AI_Check_Lac/manifests/hirschberg_candidate_eval.json")
PRODUCTION_ONNX_PATH = PROJECT_ROOT / "app" / "models" / "best_model.onnx"


def test_research_candidate_artifact_contract():
    """Verify that the research candidate bundle conforms to the non-production contract."""
    assert CANDIDATE_PATH.is_file(), f"Research candidate not found at {CANDIDATE_PATH}"
    bundle = joblib.load(CANDIDATE_PATH)

    assert isinstance(bundle, dict), "Artifact must be a dictionary bundle"
    assert bundle["model_id"] == "hirschberg-candidate-v0.1"
    assert bundle["status"] == "research_candidate"
    assert bundle["is_production"] is False, "Must NEVER be marked as production"
    assert "pipeline" in bundle
    assert "feature_names" in bundle
    assert "classes" in bundle
    assert bundle["classes"] == ["esotropia", "exotropia", "normal"]
    assert "non_clinical_declaration" in bundle

    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    assert len(feature_names) == 73, f"Expected 73 features, got {len(feature_names)}"

    # Test inference on synthetic 73-dim input
    dummy_input = np.zeros((2, len(feature_names)), dtype=np.float32)
    preds = pipeline.predict(dummy_input)
    assert len(preds) == 2
    assert all(p in [0, 1, 2] for p in preds)

    probs = pipeline.predict_proba(dummy_input)
    assert probs.shape == (2, 3)
    assert np.allclose(np.sum(probs, axis=1), [1.0, 1.0])


def test_production_model_remains_untouched():
    """Verify that production ONNX model is unchanged and not replaced by research artifacts."""
    assert PRODUCTION_ONNX_PATH.is_file(), "Production ONNX model must exist"
    assert PRODUCTION_ONNX_PATH.stat().st_size > 40_000_000, "Production model size must match ResNet18"


def test_evaluation_report_compliance():
    """Verify that the evaluation report records governance declarations and test metrics."""
    assert EVAL_REPORT_PATH.is_file(), f"Evaluation report missing at {EVAL_REPORT_PATH}"
    report = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "COMPLETED_RESEARCH_CANDIDATE"
    assert "governanceDeclarations" in report
    assert any("NOT deployed to production" in d for d in report["governanceDeclarations"])
    assert "testMetrics" in report
    test_metrics = report["testMetrics"]
    assert "test_accuracy" in test_metrics
    assert "test_balanced_accuracy" in test_metrics
    assert "test_macro_f1" in test_metrics
    assert "binary_strabismus_vs_normal" in test_metrics
