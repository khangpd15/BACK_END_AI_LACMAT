from pathlib import Path

import joblib
import numpy as np
import pytest


MODEL_PATH = Path("app/models/research/hirschberg_candidate_v0.6_ensemble.joblib")
EVAL_PATH = Path("reports/hirschberg_candidate_v0.6_ensemble_eval.json")
CARD_PATH = Path("reports/hirschberg_candidate_v0.6_ensemble_model_card.md")


def test_candidate_v06_ensemble_artifact_contract():
    assert MODEL_PATH.is_file(), f"Model artifact missing at {MODEL_PATH}"
    bundle = joblib.load(MODEL_PATH)

    assert bundle["model_id"] == "hirschberg-candidate-v0.6-ensemble"
    assert bundle["status"] == "research_candidate"
    assert bundle["is_production"] is False
    assert bundle["classes"] == ["esotropia", "exotropia", "normal"]
    assert bundle["fusion_policy"]["type"] == "agreement_confidence_weighted"
    assert bundle["fusion_policy"]["conf_threshold"] > 0.0
    assert bundle["fusion_policy"]["margin_threshold"] >= 0.0
    assert bundle["fusion_policy"]["abstain_status"] == "UNCERTAIN"
    assert "non_clinical_declaration" in bundle


def test_candidate_v06_reports_exist():
    assert EVAL_PATH.is_file(), f"Eval report missing at {EVAL_PATH}"
    assert CARD_PATH.is_file(), f"Model card missing at {CARD_PATH}"
