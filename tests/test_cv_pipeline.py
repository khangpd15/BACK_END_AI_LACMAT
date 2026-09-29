"""Unit and integration tests for CV & Kinematics pipeline enhancements.

Covers:
1. One Euro Filter jitter attenuation and saccade peak preservation
2. Canthal and IPD geometry normalization
3. Advanced Quality Gate (Head pose, blur, blink, face distance)
4. Refixation Event Detector (Pathological saccade vs physiological jitter)
"""

import math
import numpy as np
import pytest

from app.services.cv.canthal_normalizer import CanthalGeometryNormalizer
from app.services.cv.one_euro_filter import (
    EyeTrajectoryOneEuroFilter,
    OneEuroFilter1D,
)
from app.services.cv.quality_gate import AdvancedQualityGate
from app.services.kinematics.refixation_detector import (
    RefixationEvent,
    RefixationEventDetector,
)


# =============================================================================
# 1. One Euro Filter Tests
# =============================================================================

def test_one_euro_filter_jitter_smoothing():
    """Verify that stationary jitter is smoothed out."""
    filt = OneEuroFilter1D(min_cutoff=1.0, beta=0.007)
    
    # Simulate a steady fixation at x=0.5 with high-frequency noise
    noisy_signal = [0.50 + 0.005 * ((-1) ** i) for i in range(30)]
    filtered = []
    for i, val in enumerate(noisy_signal):
        filtered.append(filt.filter(val, timestamp_sec=i * 0.0333))
    
    raw_std = float(np.std(noisy_signal[-15:]))
    filt_std = float(np.std(filtered[-15:]))
    assert filt_std < raw_std * 0.60, f"Filtered std {filt_std} should be much smaller than raw {raw_std}"


def test_one_euro_filter_saccade_peak_preservation():
    """Verify that during a fast saccadic movement, the peak is preserved without severe clipping."""
    filt = OneEuroFilter1D(min_cutoff=0.8, beta=0.007)
    
    # Simulate sudden eye jump from 0.40 to 0.46 (saccade) and stay at 0.46
    trajectory = [0.40] * 10 + [0.41, 0.43, 0.455, 0.460] + [0.460] * 10
    filtered = []
    for i, val in enumerate(trajectory):
        filtered.append(filt.filter(val, timestamp_sec=i * 0.0333))
    
    peak_raw = max(trajectory)
    peak_filt = max(filtered)
    # Preservation ratio should exceed 95%
    assert peak_filt >= peak_raw * 0.95, f"Peak {peak_filt} should preserve raw peak {peak_raw}"


def test_bilateral_trajectory_filter():
    """Verify bilateral filter handles None/NaN coordinates gracefully."""
    b_filt = EyeTrajectoryOneEuroFilter()
    
    # Normal frame
    l, r = b_filt.process_frame(0.033, (0.45, 0.60), (0.58, 0.60))
    assert l is not None and r is not None
    assert pytest.approx(l[0], abs=1e-3) == 0.45
    assert pytest.approx(r[0], abs=1e-3) == 0.58

    # Frame with left eye closed / missing
    l2, r2 = b_filt.process_frame(0.066, None, (0.582, 0.601))
    assert l2 is None
    assert r2 is not None


# =============================================================================
# 2. Canthal & IPD Normalizer Tests
# =============================================================================

def test_canthal_geometry_normalization():
    """Verify iris displacement normalized by canthal width is invariant to distance."""
    # Subject A: Close to camera (Eye width = 100 pixels in frame units: 0.10)
    res_close = CanthalGeometryNormalizer.normalize_iris_coordinates(
        iris_xy=(0.52, 0.50),
        inner_canthus=(0.55, 0.50),
        outer_canthus=(0.45, 0.50),
    )
    # Eye center is 0.50, displacement = 0.02, eye width = 0.10 -> normDx = 0.02 / 0.10 = 0.20
    assert pytest.approx(res_close["normDx"], abs=1e-3) == 0.20

    # Subject B: Far from camera (Eye width = 50 pixels in frame units: 0.05)
    # The physical eye turn is identical (20% of eye width), so displacement in frame is 0.01
    res_far = CanthalGeometryNormalizer.normalize_iris_coordinates(
        iris_xy=(0.51, 0.50),
        inner_canthus=(0.525, 0.50),
        outer_canthus=(0.475, 0.50),
    )
    assert pytest.approx(res_far["normDx"], abs=1e-3) == 0.20


# =============================================================================
# 3. Quality Gate Tests
# =============================================================================

def test_quality_gate_pass_nominal():
    """Verify a clean, centered, open-eyed frame passes the gate."""
    gate = AdvancedQualityGate()
    result = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),  # face width ratio = 0.40
        head_pose=(1.5, -2.0, 0.5),           # minimal yaw/pitch/roll
        left_eye_ear=0.28,
        right_eye_ear=0.29,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        mediapipe_confidence=0.98,
    )
    assert result.is_acceptable is True
    assert result.camera_quality_score > 0.80
    assert result.tracking_quality_score > 0.85
    assert len(result.failure_reasons) == 0


def test_quality_gate_head_yaw_rejection():
    """Verify excessive head turn triggers guidance and warning."""
    gate = AdvancedQualityGate(max_yaw_deg=10.0)
    result = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(18.5, 0.0, 0.0),           # Yaw = 18.5 deg > 10.0
        left_eye_ear=0.28,
        right_eye_ear=0.29,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        mediapipe_confidence=0.98,
    )
    assert result.is_acceptable is False
    assert any("HEAD_YAW_EXCEEDED" in r for r in result.failure_reasons)
    assert "nhìn thẳng" in result.user_guidance.lower()


def test_quality_gate_blink_detection():
    """Verify blink frames (low EAR) are flagged."""
    gate = AdvancedQualityGate(min_ear_blink=0.18)
    result = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(0.0, 0.0, 0.0),
        left_eye_ear=0.08,                    # Eye closed
        right_eye_ear=0.09,                   # Eye closed
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        mediapipe_confidence=0.98,
    )
    assert result.is_acceptable is False
    assert "EYE_BLINK_DETECTED" in result.failure_reasons


# =============================================================================
# 4. Refixation Event Detector Tests
# =============================================================================

def test_refixation_detector_pathological_saccade():
    """Verify a true refixation saccade is detected with high confidence."""
    detector = RefixationEventDetector(min_saccade_velocity=0.06, min_displacement_threshold=0.012)
    
    # 20 samples at 33.3ms intervals
    t_ms = np.linspace(0.0, 666.0, 21)
    # Eye suddenly refixates: 0 -> 0.035 in 100ms, then stabilizes
    displacements = np.array([
        0.001, 0.002, 0.010, 0.025, 0.035, 0.036, 0.035, 0.034, 0.035,
        0.035, 0.035, 0.034, 0.035, 0.035, 0.035, 0.035, 0.034, 0.035,
        0.035, 0.035, 0.035
    ])
    event = detector.analyze_uncover_trajectory(t_ms, displacements)

    assert event.has_refixation is True
    assert event.confidence > 0.70
    assert event.peak_displacement >= 0.034
    assert event.peak_velocity > 0.08
    assert event.direction_sign == 1


def test_refixation_detector_physiological_jitter_rejection():
    """Verify micro-jitter around baseline does NOT trigger a refixation saccade."""
    detector = RefixationEventDetector(min_saccade_velocity=0.06, min_displacement_threshold=0.012)
    
    t_ms = np.linspace(0.0, 666.0, 21)
    # Random micro-jitter < 0.003
    displacements = np.array([0.001 * ((-1) ** i) for i in range(21)])
    event = detector.analyze_uncover_trajectory(t_ms, displacements)

    assert event.has_refixation is False
    assert event.confidence == 0.0
