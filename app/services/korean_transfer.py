"""RemiCare Transfer Inference Service — Phase 5.

Loads the best domain-adapted transfer model (remicare_transfer_model.joblib).
Falls back to the Korean baseline model if the new artifact is not found.

Key improvement in Phase 5:
- Model artifact includes experiment metadata (B1 or B2)
- If artifact.coordinate_rescaling == True:
    Apply IPD scale factor to horizontal disparity features BEFORE inference.
    This compensates for coordinate-space mismatch:
        Korean infrared viewport  -> meanDeltaX ~ 0.33
        RemiCare MediaPipe frame  -> meanDeltaX ~ 0.13
    The model was trained with Korean data divided by the scale factor (mapped to RemiCare scale).
    At inference time, RemiCare features go in directly without modification.

Singleton pattern:
- Model is loaded once at startup.
- Hot-reload: reset the singleton by calling KoreanTransferService.reset()

Clinical constraint:
- Output is a research transfer experiment result.
- NOT a clinical screening result, NOT a diagnosis.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from app.schemas import (
    ScreeningRequest,
    TransferDomainShiftInfo,
    TransferExperimentResponse,
    TransferComparisonModelResult,
    TransferModelMetadata,
)
from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    EXCLUDED_FEATURES,
    VIEWPORT_POSITION_FEATURES,
    extract_shared_features_vector,
)

# =============================================================================
# CANONICAL FEATURE ORDER (Single Source of Truth)
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

assert len(FEATURE_ORDER) == 30
for _excl in EXCLUDED_FEATURES:
    assert _excl not in FEATURE_ORDER

# Horizontal disparity features subject to coordinate-space rescaling
HORIZONTAL_DISPARITY_FEATURES: List[str] = [
    "meanDeltaX",
    "medianDeltaX",
    "stdDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "rangeDeltaX",
    "meanAbsDeltaX",
]

# Model paths — priority order
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REMICARE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "app", "models", "remicare_transfer_model.joblib")
KOREAN_FALLBACK_PATH = os.path.join(_PROJECT_ROOT, "app", "models", "korean_shared_model.joblib")
FIFTEEN_FPS_MODEL_PATH = os.path.join(_PROJECT_ROOT, "app", "models", "remicare_15fps_candidate.joblib")


# =============================================================================
# TRANSFER SERVICE
# =============================================================================

class KoreanTransferService:
    """Manages transfer model lifecycle and cross-domain inference for RemiCare."""

    _instance: Optional["KoreanTransferService"] = None

    def __init__(self, model_file_path: Optional[str] = None):
        self.artifact: Optional[Dict[str, Any]] = None
        self.model = None
        self.feature_names: List[str] = []
        self.coordinate_rescaling: bool = False
        self.ipd_scale_factor: Optional[float] = None
        self.experiment: str = "BASELINE"
        self.model_path: str = ""
        self.fifteen_fps_artifact: Optional[Dict[str, Any]] = None
        self._load_model(model_file_path)
        self._load_fifteen_fps_candidate()

    def _load_fifteen_fps_candidate(self) -> None:
        """Load the optional sampling-robust comparison model without replacing B2."""
        if not os.path.exists(FIFTEEN_FPS_MODEL_PATH):
            self.fifteen_fps_artifact = None
            return
        artifact = joblib.load(FIFTEEN_FPS_MODEL_PATH)
        if artifact.get("model") is None or not artifact.get("feature_names"):
            raise ValueError("Sampling-robust candidate artifact is missing model or feature_names")
        self.fifteen_fps_artifact = artifact

    def _load_model(self, override_path: Optional[str] = None) -> None:
        """Load the best available model artifact.

        Priority:
        1. override_path (for testing)
        2. remicare_transfer_model.joblib (Phase 5 domain-adapted)
        3. korean_shared_model.joblib (Phase 4 baseline fallback)
        """
        candidates = []
        if override_path:
            candidates.append(override_path)
        candidates.append(REMICARE_MODEL_PATH)
        candidates.append(KOREAN_FALLBACK_PATH)

        loaded_path: Optional[str] = None
        for path in candidates:
            if os.path.exists(path):
                loaded_path = path
                break

        if loaded_path is None:
            raise FileNotFoundError(
                "No model artifact found. Checked:\n"
                f"  1. {REMICARE_MODEL_PATH}\n"
                f"  2. {KOREAN_FALLBACK_PATH}\n"
                "Ensure training scripts have been executed (Phase 4 or Phase 5)."
            )

        try:
            self.artifact = joblib.load(loaded_path)
            self.model = self.artifact.get("model")
            if self.model is None:
                raise ValueError("Model artifact does not contain a valid 'model' object.")

            # Read feature list from artifact, fall back to canonical FEATURE_ORDER
            self.feature_names = self.artifact.get("feature_names", FEATURE_ORDER)

            # Phase 5 coordinate rescaling metadata
            self.coordinate_rescaling = bool(self.artifact.get("coordinate_rescaling", False))
            self.ipd_scale_factor = self.artifact.get("ipd_scale_factor", None)
            self.experiment = self.artifact.get("experiment", "BASELINE")
            self.model_path = loaded_path

        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact from {loaded_path}: {e}") from e

    @classmethod
    def get_instance(cls, model_file_path: Optional[str] = None) -> "KoreanTransferService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_file_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (used for hot-reload or testing with a different model path)."""
        cls._instance = None

    def _apply_coordinate_rescaling(
        self,
        features: Dict[str, Optional[float]],
    ) -> Dict[str, Optional[float]]:
        """No-op for RemiCare input in B2 experiment.

        The B2 model was trained with Korean data rescaled DOWN to RemiCare IPD scale.
        Therefore, RemiCare features are used AS-IS — no rescaling needed at inference.
        This method is kept for documentation clarity and future extension.
        """
        return features

    def extract_features(self, request: ScreeningRequest) -> Tuple[Dict[str, Optional[float]], List[str]]:
        """Flatten all cycle samples and extract shared features."""
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

        # For B2: RemiCare features go in as-is (model was trained in RemiCare scale)
        if self.coordinate_rescaling:
            features = self._apply_coordinate_rescaling(features)

        # Check which required features are missing
        required = self.feature_names if self.feature_names else FEATURE_ORDER
        missing_features = [f for f in required if features.get(f) is None]
        return features, missing_features

    def predict_transfer(self, request: ScreeningRequest) -> TransferExperimentResponse:
        """Execute transfer inference on a validated ScreeningRequest."""
        features, missing = self.extract_features(request)

        if missing:
            raise ValueError(
                f"Cannot compute required features: {missing}. "
                "Insufficient valid eye tracking frames."
            )

        # Build feature vector in the order the model expects
        feature_order = self.feature_names if self.feature_names else FEATURE_ORDER
        feature_vector: List[float] = []
        for name in feature_order:
            val = features.get(name)
            if val is None or not np.isfinite(val):
                raise ValueError(f"Feature '{name}' has non-finite value: {val}")
            feature_vector.append(float(val))

        X = np.array([feature_vector], dtype=float)

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

        # Build response
        model_meta = TransferModelMetadata(
            name="Korean 10–15 FPS Transfer Research Model",
            version="remicare-transfer-10to15fps-candidate-v1.1.0",
        )

        experiment_label = self.experiment
        if self.coordinate_rescaling and self.ipd_scale_factor:
            shift_note = (
                f"B2 IPD Coordinate Rescaling applied (scale={self.ipd_scale_factor:.3f}). "
                f"Korean train data was rescaled to match RemiCare MediaPipe coordinate space."
            )
        else:
            shift_note = "B1 Invariant features only (4 viewport position features dropped)."

        domain_shift = TransferDomainShiftInfo(
            source="KOREAN_INFRARED_EYE_TRACKER",
            target="REMICARE_WEBCAM_MEDIAPIPE",
            warning=True,
            potentialShiftFeatures=HORIZONTAL_DISPARITY_FEATURES if self.coordinate_rescaling else VIEWPORT_POSITION_FEATURES,
        )

        if self.fifteen_fps_artifact is not None:
            candidate_names = list(self.fifteen_fps_artifact["feature_names"])
            candidate_vector = []
            for name in candidate_names:
                value = features.get(name)
                if value is None or not np.isfinite(value):
                    raise ValueError(f"10-15 FPS candidate feature '{name}' is non-finite: {value}")
                candidate_vector.append(float(value))
            candidate_x = np.asarray([candidate_vector], dtype=float)
            candidate_model = self.fifteen_fps_artifact["model"]
            candidate_index = int(candidate_model.predict(candidate_x)[0])
            candidate_label = "NORMAL" if candidate_index == 0 else "STRABISMUS"
            candidate_probability = candidate_model.predict_proba(candidate_x)[0]
            candidate_probabilities = {
                "NORMAL": round(float(candidate_probability[0]), 4),
                "STRABISMUS": round(float(candidate_probability[1]), 4),
            }

            candidate_label_name = self.fifteen_fps_artifact.get("ui_label", "Korean 10–15 FPS candidate")
            candidate_sampling_profile = self.fifteen_fps_artifact.get(
                "sampling_profile",
                "Korean recordings augmented across fixed and variable 10–15 FPS with simulated frame drops",
            )
            candidate_meta = TransferModelMetadata(
                name=self.fifteen_fps_artifact.get("name", "Korean 10-15 FPS robust transfer candidate"),
                version=self.fifteen_fps_artifact.get("version", "remicare-transfer-10to15fps-candidate-v1.1.0"),
            )

            primary_prediction = candidate_label
            primary_probabilities = candidate_probabilities
            active_model_meta = candidate_meta

            comparison_models = [
                TransferComparisonModelResult(
                    key="korean_15fps_candidate",
                    label=candidate_label_name,
                    prediction=candidate_label,
                    classProbability=candidate_probabilities,
                    model=candidate_meta,
                    samplingProfile=candidate_sampling_profile,
                    domainShiftWarning=True,
                    clinicalMeaning=None,
                    notice=(
                        "Research-only sampling-matched output. No RemiCare labels were used; "
                        "this is not a medical diagnosis or validated clinical probability."
                    ),
                ),
                TransferComparisonModelResult(
                    key="korean_b2",
                    label="Korean B2 (60 Hz source)",
                    prediction=prediction_label,
                    classProbability=class_probabilities,
                    model=TransferModelMetadata(
                        name="Korean B2 (60 Hz source)",
                        version="remicare-transfer-b2-v1.0.0",
                    ),
                    samplingProfile="Korean source approximately 60 Hz; B2 coordinate-rescaled transfer",
                    domainShiftWarning=True,
                    clinicalMeaning=None,
                    notice="Research-only Korean transfer output - not a medical diagnosis.",
                ),
            ]
        else:
            primary_prediction = prediction_label
            primary_probabilities = class_probabilities
            active_model_meta = model_meta

            comparison_models = [
                TransferComparisonModelResult(
                    key="korean_15fps_candidate",
                    label="Korean 10–15 FPS candidate",
                    prediction=prediction_label,
                    classProbability=class_probabilities,
                    model=model_meta,
                    samplingProfile="Korean recordings augmented across fixed and variable 10–15 FPS with simulated frame drops",
                    domainShiftWarning=True,
                    clinicalMeaning=None,
                    notice="Research-only output. Not a medical diagnosis.",
                )
            ]

        return TransferExperimentResponse(
            sampleId=request.sampleId,
            status="TRANSFER_EXPERIMENT",
            inputCompatible=True,
            prediction=primary_prediction,
            classProbability=primary_probabilities,
            domainShiftWarning=True,
            clinicalMeaning=None,
            model=active_model_meta,
            domainShift=domain_shift,
            features=features,
            comparisonModels=comparison_models,
            notice=(
                f"Research transfer experiment only – not a diagnosis. "
                f"Experiment: {experiment_label}. {shift_note}"
            ),
        )


# Global access function
def get_korean_transfer_service(model_file_path: Optional[str] = None) -> KoreanTransferService:
    return KoreanTransferService.get_instance(model_file_path)
