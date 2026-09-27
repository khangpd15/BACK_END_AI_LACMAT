"""Korean Shared Feature Model Transfer Service for RemiCare.

Executes cross-domain transfer inference on raw Cover Test time-series:
- Enforces strict FEATURE_ORDER (30 features)
- Reuses verified formulas from shared_feature_contract.py
- Rejects protocol-dependent features (sampleCount, durationMs, estimatedHz, meanIntervalMs)
- Loads model ONCE at startup (singleton pattern)
- Returns structured TransferExperimentResponse with domainShiftWarning=True
- Strictly avoids diagnostic or clinical risk claims
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from app.config import MODEL_PATH
from app.schemas import (
    ScreeningRequest,
    TransferDomainShiftInfo,
    TransferExperimentResponse,
    TransferModelMetadata,
)
from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    EXCLUDED_FEATURES,
    extract_shared_features_vector,
)

# =============================================================================
# 1. CANONICAL FEATURE ORDER (Single Source of Truth)
# =============================================================================

FEATURE_ORDER: List[str] = [
    # Validity Ratios (3)
    "leftValidRatio",
    "rightValidRatio",
    "bothValidRatio",
    # Horizontal Inter-Ocular Disparity (7)
    "meanDeltaX",
    "medianDeltaX",
    "stdDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "rangeDeltaX",
    "meanAbsDeltaX",
    # Vertical Inter-Ocular Disparity (7)
    "meanDeltaY",
    "medianDeltaY",
    "stdDeltaY",
    "minDeltaY",
    "maxDeltaY",
    "rangeDeltaY",
    "meanAbsDeltaY",
    # Monocular Left Eye Coordinates & Dispersion (4)
    "meanLeftX",
    "stdLeftX",
    "meanLeftY",
    "stdLeftY",
    # Monocular Right Eye Coordinates & Dispersion (4)
    "meanRightX",
    "stdRightX",
    "meanRightY",
    "stdRightY",
    # Kinematic Velocities & Velocity Disparity (5)
    "meanLeftVelocity",
    "peakLeftVelocity",
    "meanRightVelocity",
    "peakRightVelocity",
    "velocityDisparity",
]

# Static assertions to guarantee feature contract integrity
assert len(FEATURE_ORDER) == 30, f"FEATURE_ORDER must contain exactly 30 features, got {len(FEATURE_ORDER)}"
for _excluded in EXCLUDED_FEATURES:
    assert _excluded not in FEATURE_ORDER, f"Protocol-dependent feature {_excluded} must NOT be in FEATURE_ORDER"


# =============================================================================
# 2. TRANSFER SERVICE IMPLEMENTATION (Singleton Loader)
# =============================================================================

class KoreanTransferService:
    """Manages Korean Shared Model lifecycle and cross-domain transfer inference."""

    _instance: Optional["KoreanTransferService"] = None

    def __init__(self, model_file_path: Optional[str] = None):
        self.model_path = model_file_path or MODEL_PATH
        self.artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load joblib artifact once at initialization."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Korean shared model artifact not found at: {self.model_path}. "
                "Ensure Phase 4 model training was executed."
            )

        try:
            self.artifact = joblib.load(self.model_path)
            self.model = self.artifact.get("model")
            self.feature_names = self.artifact.get("feature_names", FEATURE_ORDER)
            if self.model is None:
                raise ValueError("Model artifact does not contain a valid 'model' object.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Korean shared model artifact: {e}") from e

    @classmethod
    def get_instance(cls, model_file_path: Optional[str] = None) -> "KoreanTransferService":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_file_path)
        return cls._instance

    def extract_features(self, request: ScreeningRequest) -> Tuple[Dict[str, Optional[float]], List[str]]:
        """Flatten request samples across all cycles and extract 30 contract features."""
        all_samples: List[Dict[str, Any]] = []
        sample_idx = 0
        for cycle in request.cycles:
            for s in cycle.samples:
                all_samples.append({
                    "index": sample_idx,
                    "t": s.t,
                    "phase": s.phase,
                    "leftX": s.leftX,
                    "leftY": s.leftY,
                    "leftValid": s.leftValid,
                    "rightX": s.rightX,
                    "rightY": s.rightY,
                    "rightValid": s.rightValid,
                    "trackingQuality": s.trackingQuality,
                })
                sample_idx += 1

        features = extract_shared_features_vector(all_samples)
        missing_features = [f for f in FEATURE_ORDER if features.get(f) is None]
        return features, missing_features

    def predict_transfer(self, request: ScreeningRequest) -> TransferExperimentResponse:
        """Execute transfer inference on validated ScreeningRequest."""
        features, missing = self.extract_features(request)

        if missing:
            raise ValueError(
                f"Sample cannot compute required shared features: {missing}. "
                "Insufficient valid eye tracking frames."
            )

        # Build feature vector strictly in FEATURE_ORDER
        feature_vector: List[float] = []
        for name in FEATURE_ORDER:
            val = features[name]
            if val is None or not np.isfinite(val):
                raise ValueError(f"Feature '{name}' has non-finite value: {val}")
            feature_vector.append(float(val))

        X = np.array([feature_vector], dtype=float)

        # Run inference
        pred_idx = int(self.model.predict(X)[0])
        prediction_label = "NORMAL" if pred_idx == 0 else "STRABISMUS"

        class_probabilities = {"NORMAL": 0.0, "STRABISMUS": 0.0}
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X)[0]
                class_probabilities["NORMAL"] = round(float(probs[0]), 4)
                class_probabilities["STRABISMUS"] = round(float(probs[1]), 4)
            except Exception:
                class_probabilities[prediction_label] = 1.0

        model_meta = TransferModelMetadata(
            name=self.artifact.get("name", "korean_shared_model"),
            version=self.artifact.get("version", "shared-v1.0.0"),
        )

        domain_shift = TransferDomainShiftInfo(
            source="KOREAN_INFRARED_EYE_TRACKER",
            target="REMICARE_WEBCAM_MEDIAPIPE",
            warning=True,
            potentialShiftFeatures=["meanLeftX", "meanLeftY", "meanRightX", "meanRightY"],
        )

        return TransferExperimentResponse(
            sampleId=request.sampleId,
            status="TRANSFER_EXPERIMENT",
            inputCompatible=True,
            prediction=prediction_label,
            classProbability=class_probabilities,
            domainShiftWarning=True,
            clinicalMeaning=None,
            model=model_meta,
            domainShift=domain_shift,
            features=features,
            notice="Research transfer experiment only — not a diagnosis.",
        )


# Global access function
def get_korean_transfer_service(model_file_path: Optional[str] = None) -> KoreanTransferService:
    return KoreanTransferService.get_instance(model_file_path)
