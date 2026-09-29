"""Computer Vision services for RemiCare Strabismus AI.

Includes:
- MediaPipe Face & Iris Landmark Extractor (468/478 anatomical landmarks)
- Advanced Quality Gate (Head pose, blur, illumination, occlusions, blink, distance)
- Eye ROI Alignment & Cropping (roll-invariant horizontal canthal alignment)
- Gaze & Fixation Tracking (vector projections, velocity, I-DT fixation detection)
- Temporal Filtering (One Euro Filter for ocular trajectory smoothing)
- Canthal & Subpixel Geometry Normalizer (IPD & palpebral fissure scaling)
- Temporal Aggregation & Majority Voting (multi-frame consensus for stable inference)
"""

from app.services.cv.canthal_normalizer import CanthalGeometryNormalizer
from app.services.cv.eye_roi import EyeRoiAligner, EyeRoiCrop
from app.services.cv.gaze_tracker import GazeFixationTracker, GazeMetrics
from app.services.cv.mediapipe_eye_extractor import (
    EyeLandmarksData,
    MediaPipeEyeExtractor,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
    LEFT_INNER_CANTHUS,
    LEFT_OUTER_CANTHUS,
    RIGHT_INNER_CANTHUS,
    RIGHT_OUTER_CANTHUS,
)
from app.services.cv.one_euro_filter import (
    EyeTrajectoryOneEuroFilter,
    LowPassFilter,
    OneEuroFilter1D,
)
from app.services.cv.quality_gate import AdvancedQualityGate, QualityGateResult
from app.services.cv.temporal_aggregator import (
    FrameInferenceCandidate,
    TemporalAggregationResult,
    TemporalConsensusAggregator,
)

__all__ = [
    "OneEuroFilter1D",
    "LowPassFilter",
    "EyeTrajectoryOneEuroFilter",
    "CanthalGeometryNormalizer",
    "AdvancedQualityGate",
    "QualityGateResult",
    "MediaPipeEyeExtractor",
    "EyeLandmarksData",
    "LEFT_IRIS_CENTER",
    "RIGHT_IRIS_CENTER",
    "LEFT_INNER_CANTHUS",
    "LEFT_OUTER_CANTHUS",
    "RIGHT_INNER_CANTHUS",
    "RIGHT_OUTER_CANTHUS",
    "EyeRoiAligner",
    "EyeRoiCrop",
    "GazeFixationTracker",
    "GazeMetrics",
    "TemporalConsensusAggregator",
    "FrameInferenceCandidate",
    "TemporalAggregationResult",
]
