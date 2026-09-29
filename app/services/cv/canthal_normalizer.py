"""Canthal geometry and inter-pupillary distance (IPD) normalization module.

Normalizes raw frame-level pupil/iris coordinates relative to anatomical eye landmarks:
- Inner Canthus (Medial) & Outer Canthus (Lateral)
- Upper Eyelid & Lower Eyelid
- Inter-Pupillary Distance (IPD)

This completely decouples ocular deviation from user-to-camera distance, facial size,
and video resolution differences.
"""

import math
from typing import Dict, Optional, Tuple


class CanthalGeometryNormalizer:
    """Transforms full-frame eye coordinates to dimensionless anatomical ocular units."""

    @staticmethod
    def normalize_iris_coordinates(
        iris_xy: Tuple[float, float],
        inner_canthus: Tuple[float, float],
        outer_canthus: Tuple[float, float],
        eyelid_top: Optional[Tuple[float, float]] = None,
        eyelid_bottom: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Normalizes iris center relative to inner and outer canthus and palpebral fissure.
        
        Args:
            iris_xy: (x, y) coordinates of iris center.
            inner_canthus: (x, y) of medial canthus.
            outer_canthus: (x, y) of lateral canthus.
            eyelid_top: Optional (x, y) of superior palpebral margin.
            eyelid_bottom: Optional (x, y) of inferior palpebral margin.

        Returns:
            Dict containing normalized displacement ratios (dimensionless),
            eye width, and estimated aspect ratio.
        """
        # Eye anatomical center defined as midpoint between canthi
        eye_cx = (inner_canthus[0] + outer_canthus[0]) / 2.0
        eye_cy = (inner_canthus[1] + outer_canthus[1]) / 2.0

        # Palpebral fissure horizontal width
        eye_w = math.hypot(
            outer_canthus[0] - inner_canthus[0],
            outer_canthus[1] - inner_canthus[1],
        )
        eye_w = max(eye_w, 1e-4)

        # Palpebral fissure vertical height if provided, else anatomically approximated (approx 0.35 * width)
        if eyelid_top and eyelid_bottom:
            eye_h = math.hypot(
                eyelid_bottom[0] - eyelid_top[0],
                eyelid_bottom[1] - eyelid_top[1],
            )
            eye_h = max(eye_h, 1e-4)
        else:
            eye_h = eye_w * 0.35

        # Dimensionless normalized coordinates
        # normDx = 0.0 means iris is perfectly centered between inner and outer canthi
        norm_dx = (iris_xy[0] - eye_cx) / eye_w
        norm_dy = (iris_xy[1] - eye_cy) / eye_h

        return {
            "normDx": float(norm_dx),
            "normDy": float(norm_dy),
            "eyeWidth": float(eye_w),
            "eyeHeight": float(eye_h),
            "eyeAspectRatio": float(eye_h / eye_w),
        }

    @staticmethod
    def compute_ipd_scale(
        left_iris_xy: Tuple[float, float],
        right_iris_xy: Tuple[float, float],
    ) -> float:
        """Computes Euclidean Inter-Pupillary Distance (IPD) across the video frame."""
        return float(
            math.hypot(
                right_iris_xy[0] - left_iris_xy[0],
                right_iris_xy[1] - left_iris_xy[1],
            )
        )
