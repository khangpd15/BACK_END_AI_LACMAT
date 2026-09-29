"""Unit and integration tests for CV & Kinematics pipeline enhancements.

Covers:
1. One Euro Filter jitter attenuation and saccade peak preservation
2. Canthal and IPD geometry normalization
3. Advanced Quality Gate (Head pose, blur, blink, face distance, lighting, occlusion)
4. Refixation Event Detector (Pathological saccade vs physiological jitter)
5. MediaPipe Face & Iris Landmark Extractor (landmark mapping, distance estimation)
6. Eye ROI Alignment & Cropping (roll-invariance, horizontal canthal alignment)
7. Gaze & Fixation Tracking (vector projections, I-DT fixation detection, saccade classification)
8. Temporal Aggregation & Majority Voting (multi-frame consensus, soft probability voting)
"""

from dataclasses import dataclass
import math
import numpy as np
import pytest

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
    LEFT_TOP_EYELID,
    LEFT_BOT_EYELID,
    RIGHT_TOP_EYELID,
    RIGHT_BOT_EYELID,
    FACE_NOSE_TIP,
    FACE_CHIN,
)
from app.services.cv.one_euro_filter import (
    EyeTrajectoryOneEuroFilter,
    OneEuroFilter1D,
)
from app.services.cv.quality_gate import AdvancedQualityGate, QualityGateResult
from app.services.cv.temporal_aggregator import (
    FrameInferenceCandidate,
    TemporalAggregationResult,
    TemporalConsensusAggregator,
)
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
    
    trajectory = [0.40] * 10 + [0.41, 0.43, 0.455, 0.460] + [0.460] * 10
    filtered = []
    for i, val in enumerate(trajectory):
        filtered.append(filt.filter(val, timestamp_sec=i * 0.0333))
    
    peak_raw = max(trajectory)
    peak_filt = max(filtered)
    assert peak_filt >= peak_raw * 0.95, f"Peak {peak_filt} should preserve raw peak {peak_raw}"


def test_bilateral_trajectory_filter():
    """Verify bilateral filter handles None/NaN coordinates gracefully."""
    b_filt = EyeTrajectoryOneEuroFilter()
    
    l, r = b_filt.process_frame(0.033, (0.45, 0.60), (0.58, 0.60))
    assert l is not None and r is not None
    assert pytest.approx(l[0], abs=1e-3) == 0.45
    assert pytest.approx(r[0], abs=1e-3) == 0.58

    l2, r2 = b_filt.process_frame(0.066, None, (0.582, 0.601))
    assert l2 is None
    assert r2 is not None


# =============================================================================
# 2. Canthal & IPD Normalizer Tests
# =============================================================================

def test_canthal_geometry_normalization():
    """Verify iris displacement normalized by canthal width is invariant to distance."""
    res_close = CanthalGeometryNormalizer.normalize_iris_coordinates(
        iris_xy=(0.52, 0.50),
        inner_canthus=(0.55, 0.50),
        outer_canthus=(0.45, 0.50),
    )
    assert pytest.approx(res_close["normDx"], abs=1e-3) == 0.20

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
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(1.5, -2.0, 0.5),
        left_eye_ear=0.28,
        right_eye_ear=0.29,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        mediapipe_confidence=0.98,
        brightness=120.0,
        estimated_distance_cm=52.0,
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
        head_pose=(18.5, 0.0, 0.0),
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
        left_eye_ear=0.08,
        right_eye_ear=0.09,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        mediapipe_confidence=0.98,
    )
    assert result.is_acceptable is False
    assert "EYE_BLINK_DETECTED" in result.failure_reasons


def test_quality_gate_lighting_rejection_dark_and_glare():
    """Verify dark lighting and intense glare are flagged."""
    gate = AdvancedQualityGate(min_brightness=40.0, max_brightness=235.0)
    
    # 1. Dark room
    res_dark = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(0.0, 0.0, 0.0),
        left_eye_ear=0.28,
        right_eye_ear=0.28,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        brightness=25.0,  # Below 40.0
    )
    assert res_dark.is_acceptable is False
    assert any("LIGHTING_TOO_DARK" in r for r in res_dark.failure_reasons)
    assert "tối" in res_dark.user_guidance.lower()

    # 2. Overexposed glare
    res_glare = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(0.0, 0.0, 0.0),
        left_eye_ear=0.28,
        right_eye_ear=0.28,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        brightness=245.0,  # Above 235.0
    )
    assert res_glare.is_acceptable is False
    assert any("LIGHTING_TOO_BRIGHT" in r for r in res_glare.failure_reasons)


def test_quality_gate_eye_occlusion():
    """Verify occluded eye (e.g. hand or eye patch) is detected."""
    gate = AdvancedQualityGate()
    result = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.30, 0.20, 0.70, 0.80),
        head_pose=(0.0, 0.0, 0.0),
        left_eye_ear=0.28,
        right_eye_ear=0.28,
        left_eye_width_px=65.0,
        right_eye_width_px=64.0,
        blur_variance=250.0,
        left_eye_occluded=True,  # Left eye covered
    )
    assert result.is_acceptable is False
    assert "EYE_OCCLUDED" in result.failure_reasons


def test_quality_gate_physical_distance_bounds():
    """Verify physical distance D_cm bounds."""
    gate = AdvancedQualityGate(min_distance_cm=35.0, max_distance_cm=85.0)
    
    # Too far away (95 cm)
    res_far = gate.evaluate_frame(
        frame_shape=(720, 1280),
        face_bbox=(0.35, 0.35, 0.65, 0.65),
        head_pose=(0.0, 0.0, 0.0),
        left_eye_ear=0.28,
        right_eye_ear=0.28,
        left_eye_width_px=45.0,
        right_eye_width_px=45.0,
        blur_variance=250.0,
        estimated_distance_cm=95.0,
    )
    assert res_far.is_acceptable is False
    assert any("DISTANCE_TOO_FAR" in r for r in res_far.failure_reasons)


# =============================================================================
# 4. Refixation Event Detector Tests
# =============================================================================

def test_refixation_detector_pathological_saccade():
    """Verify a true refixation saccade is detected with high confidence."""
    detector = RefixationEventDetector(min_saccade_velocity=0.06, min_displacement_threshold=0.012)
    
    t_ms = np.linspace(0.0, 666.0, 21)
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
    displacements = np.array([0.001 * ((-1) ** i) for i in range(21)])
    event = detector.analyze_uncover_trajectory(t_ms, displacements)

    assert event.has_refixation is False
    assert event.confidence == 0.0


# =============================================================================
# 5. MediaPipe Face & Iris Landmark Extractor Tests
# =============================================================================

@dataclass
class MockLandmark:
    x: float
    y: float
    z: float = 0.0


def _build_synthetic_landmarks_478() -> list:
    """Builds a synthetic 478-point landmark array for deterministic testing."""
    landmarks = [MockLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    
    # Head & face
    landmarks[FACE_NOSE_TIP] = MockLandmark(0.50, 0.50, -0.05)
    landmarks[FACE_CHIN] = MockLandmark(0.50, 0.85, 0.0)
    
    # Left Eye (User's left eye, rendered on right side of image x ~ 0.58)
    landmarks[LEFT_INNER_CANTHUS] = MockLandmark(0.53, 0.40, 0.0)
    landmarks[LEFT_OUTER_CANTHUS] = MockLandmark(0.63, 0.40, 0.0)
    landmarks[LEFT_TOP_EYELID] = MockLandmark(0.58, 0.38, 0.0)
    landmarks[LEFT_BOT_EYELID] = MockLandmark(0.58, 0.42, 0.0)
    landmarks[385] = MockLandmark(0.56, 0.38, 0.0)
    landmarks[380] = MockLandmark(0.56, 0.42, 0.0)
    landmarks[LEFT_IRIS_CENTER] = MockLandmark(0.58, 0.40, 0.0)

    # Right Eye (User's right eye, rendered on left side of image x ~ 0.42)
    landmarks[RIGHT_OUTER_CANTHUS] = MockLandmark(0.37, 0.40, 0.0)
    landmarks[RIGHT_INNER_CANTHUS] = MockLandmark(0.47, 0.40, 0.0)
    landmarks[RIGHT_TOP_EYELID] = MockLandmark(0.42, 0.38, 0.0)
    landmarks[RIGHT_BOT_EYELID] = MockLandmark(0.42, 0.42, 0.0)
    landmarks[158] = MockLandmark(0.44, 0.38, 0.0)
    landmarks[153] = MockLandmark(0.44, 0.42, 0.0)
    landmarks[RIGHT_IRIS_CENTER] = MockLandmark(0.42, 0.40, 0.0)

    return landmarks


def test_mediapipe_extractor_landmark_mapping():
    """Verify that landmark extraction maps coordinates, EAR, and distance correctly."""
    extractor = MediaPipeEyeExtractor(ref_distance_constant=4095.0)
    synthetic_landmarks = _build_synthetic_landmarks_478()
    
    data = extractor.extract_from_landmarks(synthetic_landmarks, frame_shape=(720, 1280))
    
    assert data.left_iris_norm == (0.58, 0.40)
    assert data.right_iris_norm == (0.42, 0.40)
    # IPD in normalized units = 0.58 - 0.42 = 0.16
    assert pytest.approx(data.ipd_norm, abs=1e-3) == 0.16
    # In 1280px width, IPD = 0.16 * 1280 = 204.8 px
    assert pytest.approx(data.ipd_px, abs=1.0) == 204.8
    # Distance = 4095 / 204.8 ~ 20.0 cm
    assert pytest.approx(data.estimated_distance_cm, abs=1.0) == 20.0
    # EAR is healthy (> 0.20)
    assert data.left_ear > 0.20
    assert data.right_ear > 0.20
    assert data.left_eye_valid is True
    assert data.right_eye_valid is True


# =============================================================================
# 6. Eye ROI Alignment & Cropping Tests
# =============================================================================

def test_eye_roi_aligner_crop():
    """Verify roll-aligned ocular ROI cropping with standard target shape."""
    aligner = EyeRoiAligner(target_size=(128, 64), expansion_x=1.8, expansion_y=2.0)
    
    # Synthetic frame 480x640x3
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    
    crop = aligner.crop_aligned_eye(
        frame=frame,
        inner_canthus_px=(320.0, 200.0),
        outer_canthus_px=(400.0, 200.0),
        iris_center_px=(360.0, 200.0),
        eye_side="LEFT",
    )
    
    assert isinstance(crop, EyeRoiCrop)
    assert crop.image.shape == (64, 128, 3)
    assert crop.eye_side == "LEFT"
    assert crop.rotation_deg == 0.0  # Horizontal axis has 0 degree rotation
    assert pytest.approx(crop.iris_roi_norm[0], abs=0.1) == 0.50
    assert pytest.approx(crop.iris_roi_norm[1], abs=0.1) == 0.50


def test_eye_roi_aligner_tilted_head():
    """Verify that head tilt (canthi at an angle) produces non-zero rotation correction."""
    aligner = EyeRoiAligner(target_size=(128, 64))
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    
    # 30-degree tilt
    crop = aligner.crop_aligned_eye(
        frame=frame,
        inner_canthus_px=(300.0, 200.0),
        outer_canthus_px=(386.6, 250.0),  # dx ~ 86.6, dy = 50 -> angle = 30 deg
        iris_center_px=(343.3, 225.0),
        eye_side="LEFT",
    )
    assert pytest.approx(crop.rotation_deg, abs=1.0) == 30.0
    assert crop.image.shape == (64, 128, 3)


# =============================================================================
# 7. Gaze & Fixation Tracking Tests
# =============================================================================

def test_gaze_fixation_tracker_vector_projection():
    """Verify vector projection of iris onto canthi and eyelid segments."""
    tracker = GazeFixationTracker()
    
    # Centered eye
    gaze_x, gaze_y = tracker.compute_monocular_gaze_ratio(
        iris_xy=(0.50, 0.50),
        inner_canthus=(0.40, 0.50),
        outer_canthus=(0.60, 0.50),
        eyelid_top=(0.50, 0.40),
        eyelid_bot=(0.50, 0.60),
    )
    assert pytest.approx(gaze_x, abs=1e-3) == 0.50
    assert pytest.approx(gaze_y, abs=1e-3) == 0.50

    # Looking to the lateral side (outer canthus)
    gaze_lateral, _ = tracker.compute_monocular_gaze_ratio(
        iris_xy=(0.58, 0.50),
        inner_canthus=(0.40, 0.50),
        outer_canthus=(0.60, 0.50),
        eyelid_top=(0.50, 0.40),
        eyelid_bot=(0.50, 0.60),
    )
    assert pytest.approx(gaze_lateral, abs=1e-3) == 0.90


def test_gaze_fixation_tracker_stability_and_saccade():
    """Verify I-DT fixation detection during rest and saccade trigger during rapid shift."""
    tracker = GazeFixationTracker(dispersion_threshold=0.08, saccade_velocity_threshold=2.0)
    
    # 1. Steady fixation: feed 12 frames at x=0.50, y=0.50 with minimal jitter
    for i in range(12):
        metrics = tracker.update(
            t_sec=i * 0.033,
            left_iris_xy=(0.50 + 0.002 * ((-1) ** i), 0.50),
            right_iris_xy=(0.50 + 0.002 * ((-1) ** i), 0.50),
        )
    assert metrics.is_fixating is True
    assert metrics.is_saccade is False
    assert metrics.stability_score > 0.80

    # 2. Sudden saccade (jump from 0.50 to 0.70 in 33ms -> velocity ~ 0.20 / 0.033 = 6.0 > 2.0)
    saccade_metrics = tracker.update(
        t_sec=12 * 0.033 + 0.033,
        left_iris_xy=(0.70, 0.50),
        right_iris_xy=(0.70, 0.50),
    )
    assert saccade_metrics.gaze_velocity > 2.0
    assert saccade_metrics.is_saccade is True


# =============================================================================
# 8. Temporal Aggregation & Majority Voting Tests
# =============================================================================

def test_temporal_consensus_aggregator_majority_vote():
    """Verify that majority voting and soft probability weighting produce correct consensus."""
    aggregator = TemporalConsensusAggregator(min_valid_frames=3, min_consensus_ratio=0.60)
    
    # 5 frames: 4 NORMAL, 1 momentary STRABISMUS noise
    candidates = [
        FrameInferenceCandidate(0, 0.0, "NORMAL", {"NORMAL": 0.90, "STRABISMUS": 0.10}, is_fixating=True),
        FrameInferenceCandidate(1, 0.033, "NORMAL", {"NORMAL": 0.88, "STRABISMUS": 0.12}, is_fixating=True),
        FrameInferenceCandidate(2, 0.066, "NORMAL", {"NORMAL": 0.92, "STRABISMUS": 0.08}, is_fixating=True),
        FrameInferenceCandidate(3, 0.099, "STRABISMUS", {"NORMAL": 0.35, "STRABISMUS": 0.65}, is_fixating=False),  # Jitter outlier
        FrameInferenceCandidate(4, 0.132, "NORMAL", {"NORMAL": 0.89, "STRABISMUS": 0.11}, is_fixating=True),
    ]
    
    result = aggregator.aggregate_candidates(candidates)
    
    assert result.consensus_status == "NORMAL"
    assert result.agreement_ratio == 0.80  # 4 / 5 = 80%
    assert result.is_reliable is True
    assert result.majority_vote_counts == {"NORMAL": 4, "STRABISMUS": 1}
    assert result.class_probabilities["NORMAL"] > 0.75


def test_temporal_consensus_aggregator_rejection_and_inconclusive():
    """Verify that low-quality frames (e.g. blinks) are filtered out, triggering inconclusive if too few."""
    aggregator = TemporalConsensusAggregator(min_valid_frames=3)
    
    gate = AdvancedQualityGate()
    # Create 3 candidates, but 2 are blinking
    cand_blink1 = FrameInferenceCandidate(
        0, 0.0, "NORMAL", {"NORMAL": 0.8, "STRABISMUS": 0.2},
        quality_result=gate.evaluate_frame((720, 1280), (0.3, 0.2, 0.7, 0.8), (0, 0, 0), 0.05, 0.05, 60, 60, 200) # blink
    )
    cand_blink2 = FrameInferenceCandidate(
        1, 0.033, "NORMAL", {"NORMAL": 0.8, "STRABISMUS": 0.2},
        quality_result=gate.evaluate_frame((720, 1280), (0.3, 0.2, 0.7, 0.8), (0, 0, 0), 0.06, 0.05, 60, 60, 200) # blink
    )
    cand_good = FrameInferenceCandidate(
        2, 0.066, "NORMAL", {"NORMAL": 0.8, "STRABISMUS": 0.2},
        quality_result=gate.evaluate_frame((720, 1280), (0.3, 0.2, 0.7, 0.8), (0, 0, 0), 0.28, 0.28, 60, 60, 200) # good
    )

    result = aggregator.aggregate_candidates([cand_blink1, cand_blink2, cand_good])
    
    # Only 1 valid frame remaining < 3 required -> INCONCLUSIVE
    assert result.consensus_status == "SCREENING_INCONCLUSIVE"
    assert result.is_reliable is False
    assert result.valid_frames_count == 1
    assert result.rejected_frames_count == 2
