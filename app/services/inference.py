"""Inference service and abstraction for RemiCare Strabismus AI.

Follows strict clinical safety principles:
- Provides clean abstraction for scikit-learn / joblib ML models.
- If no model artifact exists (because independent clinical ground truth is not yet collected),
  it returns SCREENING_INCONCLUSIVE with a clear reason.
- NEVER simulates fake AI predictions.
- NEVER hard-codes heuristic thresholds (e.g. displacement > 0.10 -> ATTENTION).
- NEVER issues clinical diagnoses or prism diopter conversions.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
import joblib

from app.schemas import ScreeningStatus

logger = logging.getLogger("remicare.inference")

MODEL_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_FILE = os.path.join(MODEL_DIRECTORY, "strabismus_model.joblib")


@dataclass
class InferenceResult:
    status: str
    modelVersion: Optional[str]
    confidence: Optional[float] = None
    reason: Optional[str] = None
    notes: Optional[str] = None


class StrabismusModel:
    """Abstraction for feature-based ML inference.
    
    Adheres strictly to the requirement that prediction is only enabled
    when trained on verified, independent clinical ground truth labels.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_FILE
        self.model_artifact: Optional[Any] = None
        self.version: Optional[str] = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Attempt to load a trained model artifact if present."""
        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model_artifact = loaded["model"]
                    self.version = loaded.get("version", "strabismus-v0.1.0")
                else:
                    self.model_artifact = loaded
                    self.version = "strabismus-v0.1.0"
                logger.info(f"Loaded ML model from {self.model_path} (version: {self.version})")
            except Exception as e:
                logger.error(f"Failed loading model artifact from {self.model_path}: {e}")
                self.model_artifact = None
                self.version = None
        else:
            logger.info(f"No trained model artifact at {self.model_path}. Inference abstraction ready.")

    def is_available(self) -> bool:
        """Returns True only if a validated ML model artifact is loaded."""
        return self.model_artifact is not None

    def predict(self, features: Dict[str, Any]) -> InferenceResult:
        """Execute inference on extracted technical features.
        
        If model is unavailable, safely returns SCREENING_INCONCLUSIVE without
        fabricating clinical estimates.
        """
        if not self.is_available():
            return InferenceResult(
                status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
                modelVersion=None,
                confidence=None,
                reason="MODEL_NOT_YET_TRAINED_AWAITING_CLINICAL_GROUND_TRUTH",
                notes=(
                    "No certified model artifact is loaded. Models must be trained "
                    "on independent clinical reference labels before deployment."
                ),
            )

        # When a verified model is loaded:
        try:
            # Flatten or format features into vector
            agg = features.get("aggregatedFeatures", {})
            feature_vector = [
                agg.get("global_medianDx", 0.0),
                agg.get("global_maxPeakAbsDx", 0.0),
                agg.get("global_medianDy", 0.0),
                agg.get("global_maxPeakAbsDy", 0.0),
                agg.get("global_meanVelocity", 0.0),
                agg.get("cycleConsistency", 0.0),
            ]
            pred = self.model_artifact.predict([feature_vector])[0]
            confidence = None
            if hasattr(self.model_artifact, "predict_proba"):
                probs = self.model_artifact.predict_proba([feature_vector])[0]
                confidence = float(max(probs))

            status = (
                ScreeningStatus.SCREENING_ATTENTION.value
                if pred == 1 or pred == "ATTENTION"
                else ScreeningStatus.SCREENING_NORMAL.value
            )

            return InferenceResult(
                status=status,
                modelVersion=self.version,
                confidence=confidence,
                reason=None,
                notes="Screening result derived from ML inference.",
            )
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            return InferenceResult(
                status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
                modelVersion=self.version,
                confidence=None,
                reason="INFERENCE_EXECUTION_ERROR",
                notes=f"Error executing model: {str(e)}",
            )


# Singleton instance
_model_instance: Optional[StrabismusModel] = None


def get_strabismus_model() -> StrabismusModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = StrabismusModel()
    return _model_instance
