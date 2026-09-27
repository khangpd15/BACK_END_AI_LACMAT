"""Formal Feature Contract for RemiCare and Korean Eye-Tracking Data Spaces.

Defines the explicit shared mathematical specifications, categorizing features into:
- REQUIRED_FEATURES (Core shared features that MUST be present and computed identically)
- OPTIONAL_FEATURES (Available but protocol/device dependent)
- UNAVAILABLE_FEATURES (Modality-specific features that cannot be transferred)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    formula: str
    source: str
    dtype: str
    description: str
    category: str  # 'REQUIRED', 'OPTIONAL', 'UNAVAILABLE'


# =============================================================================
# FEATURE SPECIFICATIONS
# =============================================================================

FEATURE_SPECS: Dict[str, FeatureSpec] = {
    # -------------------------------------------------------------------------
    # 1. REQUIRED SHARED FEATURES (Computed identically on both domains)
    # -------------------------------------------------------------------------
    # Horizontal Inter-Ocular Disparity (RightX - LeftX)
    "meanDeltaX": FeatureSpec(
        name="meanDeltaX",
        formula="mean(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Average horizontal inter-ocular distance between pupils/irises",
        category="REQUIRED",
    ),
    "medianDeltaX": FeatureSpec(
        name="medianDeltaX",
        formula="median(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Median horizontal inter-ocular disparity; robust against jitter",
        category="REQUIRED",
    ),
    "stdDeltaX": FeatureSpec(
        name="stdDeltaX",
        formula="std(rightX - leftX) on mutually valid frames (ddof=0)",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Standard deviation of horizontal inter-ocular disparity",
        category="REQUIRED",
    ),
    "minDeltaX": FeatureSpec(
        name="minDeltaX",
        formula="min(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Minimum horizontal distance observed between eyes",
        category="REQUIRED",
    ),
    "maxDeltaX": FeatureSpec(
        name="maxDeltaX",
        formula="max(rightX - leftX) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Maximum horizontal distance observed between eyes",
        category="REQUIRED",
    ),
    "rangeDeltaX": FeatureSpec(
        name="rangeDeltaX",
        formula="maxDeltaX - minDeltaX",
        source="maxDeltaX, minDeltaX",
        dtype="float64",
        description="Peak-to-peak amplitude range of horizontal disparity",
        category="REQUIRED",
    ),
    "meanAbsDeltaX": FeatureSpec(
        name="meanAbsDeltaX",
        formula="mean(abs(rightX - leftX)) on mutually valid frames",
        source="rightX, leftX (bothValid)",
        dtype="float64",
        description="Mean absolute horizontal distance between eyes",
        category="REQUIRED",
    ),

    # Vertical Inter-Ocular Disparity (RightY - LeftY)
    "meanDeltaY": FeatureSpec(
        name="meanDeltaY",
        formula="mean(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Average vertical inter-ocular pupil/iris offset",
        category="REQUIRED",
    ),
    "medianDeltaY": FeatureSpec(
        name="medianDeltaY",
        formula="median(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Median vertical inter-ocular disparity",
        category="REQUIRED",
    ),
    "stdDeltaY": FeatureSpec(
        name="stdDeltaY",
        formula="std(rightY - leftY) on mutually valid frames (ddof=0)",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Standard deviation of vertical inter-ocular offset",
        category="REQUIRED",
    ),
    "minDeltaY": FeatureSpec(
        name="minDeltaY",
        formula="min(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Minimum vertical distance between eyes",
        category="REQUIRED",
    ),
    "maxDeltaY": FeatureSpec(
        name="maxDeltaY",
        formula="max(rightY - leftY) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Maximum vertical distance between eyes",
        category="REQUIRED",
    ),
    "rangeDeltaY": FeatureSpec(
        name="rangeDeltaY",
        formula="maxDeltaY - minDeltaY",
        source="maxDeltaY, minDeltaY",
        dtype="float64",
        description="Peak-to-peak excursion range of vertical disparity",
        category="REQUIRED",
    ),
    "meanAbsDeltaY": FeatureSpec(
        name="meanAbsDeltaY",
        formula="mean(abs(rightY - leftY)) on mutually valid frames",
        source="rightY, leftY (bothValid)",
        dtype="float64",
        description="Mean absolute vertical disparity between pupils",
        category="REQUIRED",
    ),

    # Coordinate Center and Dispersion
    "meanLeftX": FeatureSpec(
        name="meanLeftX",
        formula="mean(leftX) on leftValid frames",
        source="leftX (leftValid)",
        dtype="float64",
        description="Mean horizontal coordinate of left eye",
        category="REQUIRED",
    ),
    "stdLeftX": FeatureSpec(
        name="stdLeftX",
        formula="std(leftX) on leftValid frames",
        source="leftX (leftValid)",
        dtype="float64",
        description="Dispersion / standard deviation of left eye horizontal coordinate",
        category="REQUIRED",
    ),
    "meanLeftY": FeatureSpec(
        name="meanLeftY",
        formula="mean(leftY) on leftValid frames",
        source="leftY (leftValid)",
        dtype="float64",
        description="Mean vertical coordinate of left eye",
        category="REQUIRED",
    ),
    "stdLeftY": FeatureSpec(
        name="stdLeftY",
        formula="std(leftY) on leftValid frames",
        source="leftY (leftValid)",
        dtype="float64",
        description="Standard deviation of left eye vertical coordinate",
        category="REQUIRED",
    ),
    "meanRightX": FeatureSpec(
        name="meanRightX",
        formula="mean(rightX) on rightValid frames",
        source="rightX (rightValid)",
        dtype="float64",
        description="Mean horizontal coordinate of right eye",
        category="REQUIRED",
    ),
    "stdRightX": FeatureSpec(
        name="stdRightX",
        formula="std(rightX) on rightValid frames",
        source="rightX (rightValid)",
        dtype="float64",
        description="Standard deviation of right eye horizontal coordinate",
        category="REQUIRED",
    ),
    "meanRightY": FeatureSpec(
        name="meanRightY",
        formula="mean(rightY) on rightValid frames",
        source="rightY (rightValid)",
        dtype="float64",
        description="Mean vertical coordinate of right eye",
        category="REQUIRED",
    ),
    "stdRightY": FeatureSpec(
        name="stdRightY",
        formula="std(rightY) on rightValid frames",
        source="rightY (rightValid)",
        dtype="float64",
        description="Standard deviation of right eye vertical coordinate",
        category="REQUIRED",
    ),

    # Movement Dynamics and Velocity
    "meanLeftVelocity": FeatureSpec(
        name="meanLeftVelocity",
        formula="mean(hypot(dx, dy) / dt) for dt > 0.001s",
        source="leftX, leftY, t (leftValid)",
        dtype="float64",
        description="Mean frame-to-frame velocity of left eye (coord/s)",
        category="REQUIRED",
    ),
    "peakLeftVelocity": FeatureSpec(
        name="peakLeftVelocity",
        formula="max(hypot(dx, dy) / dt)",
        source="leftX, leftY, t (leftValid)",
        dtype="float64",
        description="Maximum frame-to-frame velocity of left eye (coord/s)",
        category="REQUIRED",
    ),
    "meanRightVelocity": FeatureSpec(
        name="meanRightVelocity",
        formula="mean(hypot(dx, dy) / dt) for dt > 0.001s",
        source="rightX, rightY, t (rightValid)",
        dtype="float64",
        description="Mean frame-to-frame velocity of right eye (coord/s)",
        category="REQUIRED",
    ),
    "peakRightVelocity": FeatureSpec(
        name="peakRightVelocity",
        formula="max(hypot(dx, dy) / dt)",
        source="rightX, rightY, t (rightValid)",
        dtype="float64",
        description="Maximum frame-to-frame velocity of right eye (coord/s)",
        category="REQUIRED",
    ),
    "velocityDisparity": FeatureSpec(
        name="velocityDisparity",
        formula="abs(meanRightVelocity - meanLeftVelocity)",
        source="meanRightVelocity, meanLeftVelocity",
        dtype="float64",
        description="Absolute difference between right and left eye mean velocities",
        category="REQUIRED",
    ),

    # Validity Ratios
    "leftValidRatio": FeatureSpec(
        name="leftValidRatio",
        formula="count(leftValid) / totalSamples",
        source="leftValid, sampleCount",
        dtype="float64",
        description="Ratio of frames with valid left eye detection",
        category="REQUIRED",
    ),
    "rightValidRatio": FeatureSpec(
        name="rightValidRatio",
        formula="count(rightValid) / totalSamples",
        source="rightValid, sampleCount",
        dtype="float64",
        description="Ratio of frames with valid right eye detection",
        category="REQUIRED",
    ),
    "bothValidRatio": FeatureSpec(
        name="bothValidRatio",
        formula="count(bothValid) / totalSamples",
        source="leftValid, rightValid, sampleCount",
        dtype="float64",
        description="Ratio of frames with both eyes simultaneously valid",
        category="REQUIRED",
    ),

    # -------------------------------------------------------------------------
    # 2. OPTIONAL FEATURES (Protocol / Session dependent)
    # -------------------------------------------------------------------------
    "sampleCount": FeatureSpec(
        name="sampleCount",
        formula="len(samples)",
        source="time-series length",
        dtype="int64",
        description="Total sample frame count in time-series",
        category="OPTIONAL",
    ),
    "durationMs": FeatureSpec(
        name="durationMs",
        formula="(t_end - t_start) * 1000.0",
        source="t",
        dtype="float64",
        description="Total elapsed recording duration in milliseconds",
        category="OPTIONAL",
    ),

    # -------------------------------------------------------------------------
    # 3. UNAVAILABLE / INCOMPATIBLE FEATURES
    # -------------------------------------------------------------------------
    "estimatedHz": FeatureSpec(
        name="estimatedHz",
        formula="1000.0 / medianIntervalMs",
        source="t",
        dtype="float64",
        description="Hardware sampling rate (60Hz for eye tracker vs 30Hz for webcam)",
        category="UNAVAILABLE",
    ),
    "meanIntervalMs": FeatureSpec(
        name="meanIntervalMs",
        formula="mean(dt * 1000.0)",
        source="t",
        dtype="float64",
        description="Hardware inter-frame interval (16.6ms for 60Hz vs 33.3ms for 30Hz)",
        category="UNAVAILABLE",
    ),
    "timeToPeak": FeatureSpec(
        name="timeToPeak",
        formula="t_peak - t_uncover (Cover Test phase specific)",
        source="phase, t, trackedEye",
        dtype="float64",
        description="Time to refixation peak; unavailable in Korean unphased dataset",
        category="UNAVAILABLE",
    ),
    "cycleConsistency": FeatureSpec(
        name="cycleConsistency",
        formula="Cross-cycle peak displacement stability",
        source="Cover Test repeated cycles",
        dtype="float64",
        description="Inter-cycle repeatability; unavailable in Korean continuous dataset",
        category="UNAVAILABLE",
    ),
}

# Explicit sets of feature names by contract category
REQUIRED_SHARED_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items() if spec.category == "REQUIRED"
]

OPTIONAL_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items() if spec.category == "OPTIONAL"
]

UNAVAILABLE_FEATURES: List[str] = [
    name for name, spec in FEATURE_SPECS.items() if spec.category == "UNAVAILABLE"
]


def extract_contract_features(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract standard contract features from any time-series of eye samples.
    
    Expects each sample in `samples` to have:
      - 't': float (in seconds or ms; automatically detected: if > 100, treated as ms)
      - 'leftX', 'leftY': float or None
      - 'leftValid': bool
      - 'rightX', 'rightY': float or None
      - 'rightValid': bool
    """
    total_samples = len(samples)
    if total_samples == 0:
        return {f: 0.0 for f in REQUIRED_SHARED_FEATURES}

    # Normalize timestamps to seconds for velocity calculation
    raw_ts = [s["t"] for s in samples]
    # Detect if timestamp is in ms (mean interval > 5ms) or seconds
    if len(raw_ts) > 1 and (raw_ts[-1] - raw_ts[0]) > 200:
        ts_sec = [t / 1000.0 for t in raw_ts]
    else:
        ts_sec = [float(t) for t in raw_ts]

    left_valid_samples = []
    right_valid_samples = []
    both_valid_samples = []

    for i, s in enumerate(samples):
        t_s = ts_sec[i]
        lx, ly = s.get("leftX"), s.get("leftY")
        rx, ry = s.get("rightX"), s.get("rightY")
        lv = bool(s.get("leftValid", False) and lx is not None and ly is not None and math.isfinite(lx) and math.isfinite(ly))
        rv = bool(s.get("rightValid", False) and rx is not None and ry is not None and math.isfinite(rx) and math.isfinite(ry))

        if lv:
            left_valid_samples.append((t_s, float(lx), float(ly)))
        if rv:
            right_valid_samples.append((t_s, float(rx), float(ry)))
        if lv and rv:
            both_valid_samples.append((t_s, float(lx), float(ly), float(rx), float(ry)))

    both_count = len(both_valid_samples)
    left_count = len(left_valid_samples)
    right_count = len(right_valid_samples)

    # Disparity features
    if both_count > 0:
        dxs = np.array([r[3] - r[1] for r in both_valid_samples], dtype=float)
        dys = np.array([r[4] - r[2] for r in both_valid_samples], dtype=float)

        mean_dx = float(np.mean(dxs))
        med_dx = float(np.median(dxs))
        std_dx = float(np.std(dxs))
        min_dx = float(np.min(dxs))
        max_dx = float(np.max(dxs))
        range_dx = float(max_dx - min_dx)
        mean_abs_dx = float(np.mean(np.abs(dxs)))

        mean_dy = float(np.mean(dys))
        med_dy = float(np.median(dys))
        std_dy = float(np.std(dys))
        min_dy = float(np.min(dys))
        max_dy = float(np.max(dys))
        range_dy = float(max_dy - min_dy)
        mean_abs_dy = float(np.mean(np.abs(dys)))
    else:
        mean_dx = med_dx = std_dx = min_dx = max_dx = range_dx = mean_abs_dx = 0.0
        mean_dy = med_dy = std_dy = min_dy = max_dy = range_dy = mean_abs_dy = 0.0

    # Single-eye coordinates
    if left_count > 0:
        l_xs = [r[1] for r in left_valid_samples]
        l_ys = [r[2] for r in left_valid_samples]
        mean_lx = float(np.mean(l_xs))
        std_lx = float(np.std(l_xs))
        mean_ly = float(np.mean(l_ys))
        std_ly = float(np.std(l_ys))
    else:
        mean_lx = std_lx = mean_ly = std_ly = 0.0

    if right_count > 0:
        r_xs = [r[1] for r in right_valid_samples]
        r_ys = [r[2] for r in right_valid_samples]
        mean_rx = float(np.mean(r_xs))
        std_rx = float(np.std(r_xs))
        mean_ry = float(np.mean(r_ys))
        std_ry = float(np.std(r_ys))
    else:
        mean_rx = std_rx = mean_ry = std_ry = 0.0

    # Velocities
    l_vels: List[float] = []
    for i in range(len(left_valid_samples) - 1):
        dt = left_valid_samples[i + 1][0] - left_valid_samples[i][0]
        if dt > 0.001:
            d = math.hypot(
                left_valid_samples[i + 1][1] - left_valid_samples[i][1],
                left_valid_samples[i + 1][2] - left_valid_samples[i][2],
            )
            l_vels.append(d / dt)

    r_vels: List[float] = []
    for i in range(len(right_valid_samples) - 1):
        dt = right_valid_samples[i + 1][0] - right_valid_samples[i][0]
        if dt > 0.001:
            d = math.hypot(
                right_valid_samples[i + 1][1] - right_valid_samples[i][1],
                right_valid_samples[i + 1][2] - right_valid_samples[i][2],
            )
            r_vels.append(d / dt)

    mean_l_vel = float(np.mean(l_vels)) if l_vels else 0.0
    peak_l_vel = float(np.max(l_vels)) if l_vels else 0.0
    mean_r_vel = float(np.mean(r_vels)) if r_vels else 0.0
    peak_r_vel = float(np.max(r_vels)) if r_vels else 0.0

    return {
        "meanDeltaX": round(mean_dx, 6),
        "medianDeltaX": round(med_dx, 6),
        "stdDeltaX": round(std_dx, 6),
        "minDeltaX": round(min_dx, 6),
        "maxDeltaX": round(max_dx, 6),
        "rangeDeltaX": round(range_dx, 6),
        "meanAbsDeltaX": round(mean_abs_dx, 6),
        "meanDeltaY": round(mean_dy, 6),
        "medianDeltaY": round(med_dy, 6),
        "stdDeltaY": round(std_dy, 6),
        "minDeltaY": round(min_dy, 6),
        "maxDeltaY": round(max_dy, 6),
        "rangeDeltaY": round(range_dy, 6),
        "meanAbsDeltaY": round(mean_abs_dy, 6),
        "meanLeftX": round(mean_lx, 6),
        "stdLeftX": round(std_lx, 6),
        "meanLeftY": round(mean_ly, 6),
        "stdLeftY": round(std_ly, 6),
        "meanRightX": round(mean_rx, 6),
        "stdRightX": round(std_rx, 6),
        "meanRightY": round(mean_ry, 6),
        "stdRightY": round(std_ry, 6),
        "meanLeftVelocity": round(mean_l_vel, 6),
        "peakLeftVelocity": round(peak_l_vel, 6),
        "meanRightVelocity": round(mean_r_vel, 6),
        "peakRightVelocity": round(peak_r_vel, 6),
        "velocityDisparity": round(abs(mean_r_vel - mean_l_vel), 6),
        "leftValidRatio": round(left_count / total_samples, 4),
        "rightValidRatio": round(right_count / total_samples, 4),
        "bothValidRatio": round(both_count / total_samples, 4),
    }


def validate_feature_vector(feature_dict: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate a feature dictionary against the REQUIRED_SHARED_FEATURES contract."""
    present_keys = set(feature_dict.keys())
    required_set = set(REQUIRED_SHARED_FEATURES)

    available = sorted(list(present_keys.intersection(required_set)))
    missing = sorted(list(required_set.difference(present_keys)))

    is_compatible = len(missing) == 0
    return is_compatible, available, missing
