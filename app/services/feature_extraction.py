"""Feature extraction service for RemiCare Strabismus AI.

Extracts objective technical refixation and movement dynamics features
from UNCOVER/TRACKING phases on a cycle-by-cycle basis:
- Horizontal displacement statistics (signed & absolute)
- Vertical displacement statistics (signed & absolute)
- Temporal dynamics (duration, time to peak, velocities)
- Quality and baseline stability metrics
- Inter-cycle consistency features

CRITICAL: Features are purely technical coordinates and temporal statistics.
They are NOT converted to prism diopters and NOT labeled as clinical angles.
"""

import math
from typing import Any, Dict, List
import numpy as np

from app.services.preprocessing import ProcessedCycle, ProcessedData


def extract_cycle_features(cycle: ProcessedCycle) -> Dict[str, Any]:
    """Extract technical movement and refixation features for an individual cycle.
    
    Each cycle is analyzed independently without cross-cycle mixing.
    """
    tracking_samples = [s for s in cycle.samples if s.phase in ("UNCOVER", "TRACKING")]
    valid_tracking = [s for s in tracking_samples if s.trackedEyeValid and s.signedDx is not None and s.signedDy is not None]

    total_tracking_count = len(tracking_samples)
    valid_tracking_count = len(valid_tracking)
    valid_ratio = valid_tracking_count / total_tracking_count if total_tracking_count > 0 else 0.0

    mean_quality = (
        float(np.mean([s.trackingQuality for s in tracking_samples]))
        if total_tracking_count > 0
        else 0.0
    )

    if valid_tracking_count == 0:
        return {
            "cycle": cycle.cycle,
            "coveredEye": cycle.coveredEye,
            "trackedEye": cycle.trackedEye,
            # Horizontal
            "meanDx": 0.0,
            "medianDx": 0.0,
            "stdDx": 0.0,
            "minDx": 0.0,
            "maxDx": 0.0,
            "peakAbsDx": 0.0,
            # Vertical
            "meanDy": 0.0,
            "medianDy": 0.0,
            "stdDy": 0.0,
            "minDy": 0.0,
            "maxDy": 0.0,
            "peakAbsDy": 0.0,
            # Temporal
            "movementDuration": 0.0,
            "timeToPeak": 0.0,
            "meanVelocity": 0.0,
            "peakVelocity": 0.0,
            # Quality
            "sampleCount": total_tracking_count,
            "validSampleCount": 0,
            "validRatio": 0.0,
            "meanTrackingQuality": mean_quality,
            # Baseline
            "baselineStdX": cycle.baseline.baselineStdX,
            "baselineStdY": cycle.baseline.baselineStdY,
            "baselineTrackedX": cycle.baseline.baselineTrackedX,
            "baselineTrackedY": cycle.baseline.baselineTrackedY,
        }

    dxs = np.array([s.signedDx for s in valid_tracking], dtype=float)
    dys = np.array([s.signedDy for s in valid_tracking], dtype=float)
    abs_dxs = np.abs(dxs)
    abs_dys = np.abs(dys)
    ts = np.array([s.t for s in valid_tracking], dtype=float)

    # Horizontal metrics
    mean_dx = float(np.mean(dxs))
    median_dx = float(np.median(dxs))
    std_dx = float(np.std(dxs, ddof=0)) if len(dxs) > 1 else 0.0
    min_dx = float(np.min(dxs))
    max_dx = float(np.max(dxs))
    peak_abs_dx = float(np.max(abs_dxs))

    # Vertical metrics
    mean_dy = float(np.mean(dys))
    median_dy = float(np.median(dys))
    std_dy = float(np.std(dys, ddof=0)) if len(dys) > 1 else 0.0
    min_dy = float(np.min(dys))
    max_dy = float(np.max(dys))
    peak_abs_dy = float(np.max(abs_dys))

    # Temporal metrics
    start_t = float(ts[0])
    end_t = float(ts[-1])
    movement_duration = max(0.0, end_t - start_t)

    # Time to peak 2D displacement
    displacements_2d = np.hypot(dxs, dys)
    peak_idx = int(np.argmax(displacements_2d))
    time_to_peak = max(0.0, float(ts[peak_idx] - start_t))

    # Velocity estimation between consecutive valid samples (coord / second)
    velocities: List[float] = []
    for i in range(len(valid_tracking) - 1):
        dt = (valid_tracking[i + 1].t - valid_tracking[i].t) / 1000.0  # convert ms to seconds
        if dt > 0.001:  # avoid divide-by-zero or microsecond artifacts
            d_dist = math.hypot(
                valid_tracking[i + 1].trackedEyeX - valid_tracking[i].trackedEyeX,
                valid_tracking[i + 1].trackedEyeY - valid_tracking[i].trackedEyeY,
            )
            velocities.append(d_dist / dt)

    mean_velocity = float(np.mean(velocities)) if velocities else 0.0
    peak_velocity = float(np.max(velocities)) if velocities else 0.0

    return {
        "cycle": cycle.cycle,
        "coveredEye": cycle.coveredEye,
        "trackedEye": cycle.trackedEye,
        # Horizontal
        "meanDx": round(mean_dx, 6),
        "medianDx": round(median_dx, 6),
        "stdDx": round(std_dx, 6),
        "minDx": round(min_dx, 6),
        "maxDx": round(max_dx, 6),
        "peakAbsDx": round(peak_abs_dx, 6),
        # Vertical
        "meanDy": round(mean_dy, 6),
        "medianDy": round(median_dy, 6),
        "stdDy": round(std_dy, 6),
        "minDy": round(min_dy, 6),
        "maxDy": round(max_dy, 6),
        "peakAbsDy": round(peak_abs_dy, 6),
        # Temporal
        "movementDuration": round(movement_duration, 2),
        "timeToPeak": round(time_to_peak, 2),
        "meanVelocity": round(mean_velocity, 6),
        "peakVelocity": round(peak_velocity, 6),
        # Quality
        "sampleCount": total_tracking_count,
        "validSampleCount": valid_tracking_count,
        "validRatio": round(valid_ratio, 4),
        "meanTrackingQuality": round(mean_quality, 4),
        # Baseline
        "baselineStdX": round(cycle.baseline.baselineStdX, 6),
        "baselineStdY": round(cycle.baseline.baselineStdY, 6),
        "baselineTrackedX": round(cycle.baseline.baselineTrackedX, 6),
        "baselineTrackedY": round(cycle.baseline.baselineTrackedY, 6),
    }


def extract_inter_cycle_consistency(cycle_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute consistency and stability metrics across repeated cover cycles."""
    if not cycle_features:
        return {"cycleConsistency": 0.0, "notes": "No cycles available"}

    consistency: Dict[str, Any] = {}

    # Explicit cycle-level peak displacement tracking
    for cf in cycle_features:
        c_num = cf["cycle"]
        consistency[f"cycle{c_num}_peakDx"] = cf.get("medianDx", 0.0)
        consistency[f"cycle{c_num}_peakAbsDx"] = cf.get("peakAbsDx", 0.0)
        consistency[f"cycle{c_num}_peakDy"] = cf.get("medianDy", 0.0)
        consistency[f"cycle{c_num}_peakAbsDy"] = cf.get("peakAbsDy", 0.0)

    peak_abs_dxs = [cf.get("peakAbsDx", 0.0) for cf in cycle_features]
    peak_abs_dys = [cf.get("peakAbsDy", 0.0) for cf in cycle_features]

    mean_peak_dx = float(np.mean(peak_abs_dxs))
    std_peak_dx = float(np.std(peak_abs_dxs, ddof=0)) if len(peak_abs_dxs) > 1 else 0.0
    mean_peak_dy = float(np.mean(peak_abs_dys))
    std_peak_dy = float(np.std(peak_abs_dys, ddof=0)) if len(peak_abs_dys) > 1 else 0.0

    # Repeatability / consistency index: 1.0 / (1.0 + normalized variation)
    # Higher value indicates high repeatability across cycles
    cv_dx = (std_peak_dx / mean_peak_dx) if mean_peak_dx > 1e-5 else 0.0
    cycle_consistency_score = round(1.0 / (1.0 + cv_dx), 4)

    consistency.update({
        "meanPeakAbsDx": round(mean_peak_dx, 6),
        "stdPeakAbsDx": round(std_peak_dx, 6),
        "meanPeakAbsDy": round(mean_peak_dy, 6),
        "stdPeakAbsDy": round(std_peak_dy, 6),
        "cycleConsistency": cycle_consistency_score,
        "cyclesCount": len(cycle_features),
    })

    return consistency


def extract_sample_aggregated_features(
    cycle_features: List[Dict[str, Any]],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate cycle-level features into a unified sample-level feature set."""
    if not cycle_features:
        return {}

    all_mean_dx = [cf["meanDx"] for cf in cycle_features]
    all_median_dx = [cf["medianDx"] for cf in cycle_features]
    all_peak_abs_dx = [cf["peakAbsDx"] for cf in cycle_features]

    all_mean_dy = [cf["meanDy"] for cf in cycle_features]
    all_median_dy = [cf["medianDy"] for cf in cycle_features]
    all_peak_abs_dy = [cf["peakAbsDy"] for cf in cycle_features]

    all_velocities = [cf["meanVelocity"] for cf in cycle_features]
    all_peak_velocities = [cf["peakVelocity"] for cf in cycle_features]
    all_durations = [cf["movementDuration"] for cf in cycle_features]
    all_time_to_peaks = [cf["timeToPeak"] for cf in cycle_features]

    all_baseline_std_x = [cf["baselineStdX"] for cf in cycle_features]
    all_baseline_std_y = [cf["baselineStdY"] for cf in cycle_features]

    return {
        "global_meanDx": round(float(np.mean(all_mean_dx)), 6),
        "global_medianDx": round(float(np.median(all_median_dx)), 6),
        "global_maxPeakAbsDx": round(float(np.max(all_peak_abs_dx)), 6),
        "global_meanPeakAbsDx": round(float(np.mean(all_peak_abs_dx)), 6),
        "global_meanDy": round(float(np.mean(all_mean_dy)), 6),
        "global_medianDy": round(float(np.median(all_median_dy)), 6),
        "global_maxPeakAbsDy": round(float(np.max(all_peak_abs_dy)), 6),
        "global_meanPeakAbsDy": round(float(np.mean(all_peak_abs_dy)), 6),
        "global_meanVelocity": round(float(np.mean(all_velocities)), 6),
        "global_peakVelocity": round(float(np.max(all_peak_velocities)), 6),
        "global_meanDuration": round(float(np.mean(all_durations)), 2),
        "global_meanTimeToPeak": round(float(np.mean(all_time_to_peaks)), 2),
        "global_meanBaselineStdX": round(float(np.mean(all_baseline_std_x)), 6),
        "global_meanBaselineStdY": round(float(np.mean(all_baseline_std_y)), 6),
        "cycleConsistency": consistency.get("cycleConsistency", 0.0),
    }


def extract_screening_features(processed_data: ProcessedData) -> Dict[str, Any]:
    """Execute complete feature extraction pipeline on preprocessed data.
    
    Returns structured dictionary containing cycle features, inter-cycle consistency,
    and sample-level aggregated features.
    """
    cycle_features = [extract_cycle_features(c) for c in processed_data.cycles]
    consistency = extract_inter_cycle_consistency(cycle_features)
    aggregated = extract_sample_aggregated_features(cycle_features, consistency)

    return {
        "cyclesAnalyzed": len(cycle_features),
        "cycleFeatures": cycle_features,
        "consistency": consistency,
        "aggregatedFeatures": aggregated,
    }
