"""Computer Vision services for RemiCare Strabismus AI.

Includes:
- Quality Gate (Head pose, blur, blink, face/eye sizing)
- Temporal filtering (One Euro Filter for ocular trajectory)
- Canthal & subpixel geometry normalizer
"""

from app.services.cv.one_euro_filter import (
    EyeTrajectoryOneEuroFilter,
    LowPassFilter,
    OneEuroFilter1D,
)
from app.services.cv.canthal_normalizer import CanthalGeometryNormalizer
from app.services.cv.quality_gate import AdvancedQualityGate, QualityGateResult

__all__ = [
    "OneEuroFilter1D",
    "LowPassFilter",
    "EyeTrajectoryOneEuroFilter",
    "CanthalGeometryNormalizer",
    "AdvancedQualityGate",
    "QualityGateResult",
]
