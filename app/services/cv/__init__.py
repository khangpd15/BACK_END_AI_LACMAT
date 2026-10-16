"""Computer Vision services for RemiCare Hirschberg & Strabismus AI.

Includes:
- Bilateral quality gate for ocular image screening and landmark validation
"""

from app.services.cv.bilateral_quality_gate import (
    QualityResult,
    evaluate_quality,
)

__all__ = [
    "QualityResult",
    "evaluate_quality",
]
