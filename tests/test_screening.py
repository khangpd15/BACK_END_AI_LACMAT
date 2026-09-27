"""Comprehensive unit tests for RemiCare Strabismus AI Backend.

Verifies:
1. Valid payload processing end-to-end
2. Empty cycles rejection
3. Negative / invalid timestamp rejection
4. Non-monotonic timestamp rejection
5. NaN coordinate rejection
6. Missing baseline phase rejection
7. Missing tracking phase rejection
8. Insufficient valid samples rejection
9. Baseline median calculation correctness & outlier resistance
10. Relative coordinate computation (right - left)
11. Signed movement (signedDx / signedDy) preservation
12. Inter-cycle displacement consistency metrics
13. Cycle independent analysis & aggregation
14. Health check endpoint
15. HTTP API integration test
"""

import math
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import CoverCycle, EyeSample, ScreeningRequest
from app.services.feature_extraction import (
    extract_cycle_features,
    extract_inter_cycle_consistency,
    extract_screening_features,
)
from app.services.preprocessing import (
    compute_cycle_baseline,
    preprocess_cycle,
    preprocess_screening_request,
)
from app.services.screening import run_screening_pipeline
from app.services.validation import validate_screening_request


def create_sample(
    index: int,
    t: float,
    phase: str,
    left_x: float = 0.44,
    left_y: float = 0.60,
    left_valid: bool = True,
    right_x: float = 0.58,
    right_y: float = 0.60,
    right_valid: bool = True,
    quality: float = 0.95,
) -> EyeSample:
    """Helper to construct a valid EyeSample."""
    return EyeSample(
        index=index,
        t=t,
        phase=phase,
        leftX=left_x,
        leftY=left_y,
        leftValid=left_valid,
        rightX=right_x,
        rightY=right_y,
        rightValid=right_valid,
        trackingQuality=quality,
    )


def create_standard_cycle(
    cycle_num: int = 1,
    covered: str = "LEFT",
    tracked: str = "RIGHT",
    refixation_shift_x: float = 0.02,
) -> CoverCycle:
    """Helper to build a realistic cover cycle with baseline, cover, and uncover/tracking phases."""
    samples = []
    t = 0.0

    # 1. BASELINE phase (5 samples)
    for i in range(5):
        t += 33.3
        samples.append(
            create_sample(
                index=len(samples),
                t=t,
                phase="BASELINE",
                left_x=0.440 + (0.001 * (i % 2)),
                left_y=0.600,
                right_x=0.580 + (0.001 * (i % 2)),
                right_y=0.600,
            )
        )

    # 2. COVER phase (3 samples)
    for i in range(3):
        t += 33.3
        samples.append(
            create_sample(
                index=len(samples),
                t=t,
                phase="COVER",
                left_x=None if covered == "LEFT" else 0.440,
                left_y=None if covered == "LEFT" else 0.600,
                left_valid=(covered != "LEFT"),
                right_x=None if covered == "RIGHT" else 0.580,
                right_y=None if covered == "RIGHT" else 0.600,
                right_valid=(covered != "RIGHT"),
            )
        )

    # 3. UNCOVER / TRACKING phase (8 samples) with slight refixation movement
    for i in range(8):
        t += 33.3
        shift = refixation_shift_x if i < 4 else (refixation_shift_x * 0.5)
        l_x = 0.440 + (shift if tracked == "LEFT" else 0.0)
        l_y = 0.600 + ((shift * 0.2) if tracked == "LEFT" else 0.0)
        r_x = 0.580 + (shift if tracked == "RIGHT" else 0.0)
        r_y = 0.600 + ((shift * 0.2) if tracked == "RIGHT" else 0.0)

        samples.append(
            create_sample(
                index=len(samples),
                t=t,
                phase="TRACKING",
                left_x=l_x,
                left_y=l_y,
                left_valid=True,
                right_x=r_x,
                right_y=r_y,
                right_valid=True,
            )
        )

    return CoverCycle(
        cycle=cycle_num,
        coveredEye=covered,
        trackedEye=tracked,
        samples=samples,
    )


# =====================================================================
# Test 1: Valid Payload
# =====================================================================
def test_valid_payload():
    """Verify that a well-formed 3-cycle Cover Test payload passes gate and executes."""
    req = ScreeningRequest(
        sampleId="valid-test-001",
        test="COVER_TEST",
        cycles=[
            create_standard_cycle(1, "LEFT", "RIGHT", refixation_shift_x=0.01),
            create_standard_cycle(2, "RIGHT", "LEFT", refixation_shift_x=-0.01),
            create_standard_cycle(3, "LEFT", "RIGHT", refixation_shift_x=0.01),
        ],
    )

    res = run_screening_pipeline(req)

    assert res.sampleId == "valid-test-001"
    assert res.quality.status == "PASS"
    assert res.analysis.cyclesAnalyzed == 3
    # Model not yet trained without clinical ground truth -> safe inconclusive
    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.modelVersion is None
    assert "Screening result only — not a diagnosis." in res.notice


# =====================================================================
# Test 2: Empty Cycles
# =====================================================================
def test_empty_cycles():
    """Verify rejection when cycles list is empty."""
    req = ScreeningRequest(sampleId="empty-001", test="COVER_TEST", cycles=[])
    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "EMPTY_CYCLES"
    assert res.analysis.cyclesAnalyzed == 0


# =====================================================================
# Test 3: Invalid Negative Timestamp
# =====================================================================
def test_invalid_negative_timestamp():
    """Verify rejection when timestamp is negative."""
    cycle = create_standard_cycle(1)
    # Mutate one sample to have negative timestamp
    cycle.samples[1] = create_sample(1, -50.0, "BASELINE")

    req = ScreeningRequest(sampleId="neg-t-001", test="COVER_TEST", cycles=[cycle])
    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "NEGATIVE_TIMESTAMP"


# =====================================================================
# Test 4: Non-Monotonic Timestamp
# =====================================================================
def test_non_monotonic_timestamp():
    """Verify rejection when timestamps do not strictly increase."""
    cycle = create_standard_cycle(1)
    # Set sample 3 timestamp equal to or less than sample 2
    cycle.samples[3] = create_sample(3, cycle.samples[2].t - 10.0, "BASELINE")

    req = ScreeningRequest(sampleId="non-mono-001", test="COVER_TEST", cycles=[cycle])
    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "NON_MONOTONIC_TIMESTAMPS"


# =====================================================================
# Test 5: NaN Coordinate
# =====================================================================
def test_nan_coordinate():
    """Verify rejection when coordinate is NaN."""
    cycle = create_standard_cycle(1)
    cycle.samples[2] = create_sample(2, cycle.samples[2].t, "BASELINE", right_x=float("nan"))

    req = ScreeningRequest(sampleId="nan-coord-001", test="COVER_TEST", cycles=[cycle])
    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "COORDINATES_NAN_OR_INFINITE"


# =====================================================================
# Test 6: Missing Baseline Phase
# =====================================================================
def test_missing_baseline():
    """Verify rejection when cycle contains no BASELINE phase samples."""
    samples = [
        create_sample(0, 33.3, "TRACKING"),
        create_sample(1, 66.6, "TRACKING"),
        create_sample(2, 99.9, "TRACKING"),
        create_sample(3, 133.2, "TRACKING"),
    ]
    cycle = CoverCycle(cycle=1, coveredEye="LEFT", trackedEye="RIGHT", samples=samples)
    req = ScreeningRequest(sampleId="no-baseline-001", test="COVER_TEST", cycles=[cycle])

    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "MISSING_BASELINE_PHASE"


# =====================================================================
# Test 7: Missing Tracking Phase
# =====================================================================
def test_missing_tracking_data():
    """Verify rejection when cycle contains no UNCOVER/TRACKING phase samples."""
    samples = [
        create_sample(0, 33.3, "BASELINE"),
        create_sample(1, 66.6, "BASELINE"),
        create_sample(2, 99.9, "BASELINE"),
        create_sample(3, 133.2, "BASELINE"),
    ]
    cycle = CoverCycle(cycle=1, coveredEye="LEFT", trackedEye="RIGHT", samples=samples)
    req = ScreeningRequest(sampleId="no-tracking-001", test="COVER_TEST", cycles=[cycle])

    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "MISSING_TRACKING_PHASE"


# =====================================================================
# Test 8: Insufficient Valid Samples
# =====================================================================
def test_insufficient_valid_samples():
    """Verify rejection when valid samples count in baseline or tracking is too low."""
    # Only 1 valid baseline sample (< MIN_BASELINE_VALID_SAMPLES=3)
    samples = [
        create_sample(0, 33.3, "BASELINE", right_valid=True),
        create_sample(1, 66.6, "BASELINE", right_valid=False, right_x=None),
        create_sample(2, 99.9, "TRACKING", right_valid=True),
        create_sample(3, 133.2, "TRACKING", right_valid=True),
        create_sample(4, 166.5, "TRACKING", right_valid=True),
    ]
    cycle = CoverCycle(cycle=1, coveredEye="LEFT", trackedEye="RIGHT", samples=samples)
    req = ScreeningRequest(sampleId="low-valid-001", test="COVER_TEST", cycles=[cycle])

    res = run_screening_pipeline(req)

    assert res.status == "SCREENING_INCONCLUSIVE"
    assert res.quality.status == "FAIL"
    assert res.reason == "INSUFFICIENT_VALID_BASELINE_SAMPLES"


# =====================================================================
# Test 9: Baseline Median & Outlier Robustness
# =====================================================================
def test_baseline_median():
    """Verify baseline uses median, not single frame or contaminated mean."""
    samples = [
        create_sample(0, 33.3, "BASELINE", right_x=0.500),
        create_sample(1, 66.6, "BASELINE", right_x=0.502),
        create_sample(2, 99.9, "BASELINE", right_x=0.501),
        create_sample(3, 133.2, "BASELINE", right_x=0.999),  # massive single-frame outlier
        create_sample(4, 166.5, "BASELINE", right_x=0.503),
    ]
    cycle = CoverCycle(cycle=1, coveredEye="LEFT", trackedEye="RIGHT", samples=samples)

    baseline = compute_cycle_baseline(cycle)

    # Median of [0.500, 0.501, 0.502, 0.503, 0.999] is 0.502
    assert baseline.baselineTrackedX == pytest.approx(0.502, abs=1e-4)
    # Mean would have been ~0.601, proving median rejected the outlier
    assert baseline.baselineTrackedX < 0.55


# =====================================================================
# Test 10: Relative Coordinates (Right - Left)
# =====================================================================
def test_relative_coordinates():
    """Verify relativeX = rightX - leftX and relativeY = rightY - leftY."""
    cycle = CoverCycle(
        cycle=1,
        coveredEye="LEFT",
        trackedEye="RIGHT",
        samples=[
            create_sample(0, 33.3, "BASELINE", left_x=0.44, right_x=0.58, left_y=0.61, right_y=0.60),
            create_sample(1, 66.6, "BASELINE", left_x=0.44, right_x=0.58, left_y=0.61, right_y=0.60),
            create_sample(2, 99.9, "BASELINE", left_x=0.44, right_x=0.58, left_y=0.61, right_y=0.60),
            create_sample(3, 133.2, "TRACKING", left_x=0.44, right_x=0.59, left_y=0.60, right_y=0.62),
        ],
    )

    proc = preprocess_cycle(cycle)

    sample_0 = proc.samples[0]
    assert sample_0.relativeX == pytest.approx(0.58 - 0.44, abs=1e-4)
    assert sample_0.relativeY == pytest.approx(0.60 - 0.61, abs=1e-4)

    sample_3 = proc.samples[3]
    assert sample_3.relativeX == pytest.approx(0.59 - 0.44, abs=1e-4)
    assert sample_3.relativeY == pytest.approx(0.62 - 0.60, abs=1e-4)


# =====================================================================
# Test 11: Signed Movement Preservation (signedDx / signedDy)
# =====================================================================
def test_signed_movement_preservation():
    """Verify signed direction is retained alongside absDx/absDy."""
    cycle_leftward = CoverCycle(
        cycle=1,
        coveredEye="LEFT",
        trackedEye="RIGHT",
        samples=[
            create_sample(0, 33.3, "BASELINE", right_x=0.580, right_y=0.600),
            create_sample(1, 66.6, "BASELINE", right_x=0.580, right_y=0.600),
            create_sample(2, 99.9, "BASELINE", right_x=0.580, right_y=0.600),
            # Move leftward (-dx)
            create_sample(3, 133.2, "TRACKING", right_x=0.550, right_y=0.580),
        ],
    )

    proc_left = preprocess_cycle(cycle_leftward)
    tracking_sample = proc_left.samples[3]

    assert tracking_sample.signedDx == pytest.approx(-0.030, abs=1e-4)
    assert tracking_sample.signedDy == pytest.approx(-0.020, abs=1e-4)
    assert tracking_sample.absDx == pytest.approx(0.030, abs=1e-4)
    assert tracking_sample.absDy == pytest.approx(0.020, abs=1e-4)


# =====================================================================
# Test 12: Displacement Consistency Between Cycles
# =====================================================================
def test_displacement_consistency():
    """Verify inter-cycle consistency metrics are extracted across multiple cycles."""
    c1 = create_standard_cycle(1, "LEFT", "RIGHT", refixation_shift_x=0.015)
    c2 = create_standard_cycle(2, "RIGHT", "LEFT", refixation_shift_x=0.016)
    c3 = create_standard_cycle(3, "LEFT", "RIGHT", refixation_shift_x=0.014)

    req = ScreeningRequest(sampleId="consist-001", test="COVER_TEST", cycles=[c1, c2, c3])
    proc_data = preprocess_screening_request(req)
    features = extract_screening_features(proc_data)

    consistency = features["consistency"]
    assert "cycle1_peakAbsDx" in consistency
    assert "cycle2_peakAbsDx" in consistency
    assert "cycle3_peakAbsDx" in consistency
    assert "cycleConsistency" in consistency
    assert consistency["cyclesCount"] == 3
    assert consistency["cycleConsistency"] > 0.0


# =====================================================================
# Test 13: Cycle Independent Analysis & Aggregation
# =====================================================================
def test_cycle_aggregation():
    """Verify that cycles are computed independently and then aggregated."""
    c1 = create_standard_cycle(1, "LEFT", "RIGHT", refixation_shift_x=0.010)
    c2 = create_standard_cycle(2, "RIGHT", "LEFT", refixation_shift_x=0.020)

    req = ScreeningRequest(sampleId="indep-001", test="COVER_TEST", cycles=[c1, c2])
    proc_data = preprocess_screening_request(req)
    features = extract_screening_features(proc_data)

    cycle_features = features["cycleFeatures"]
    assert len(cycle_features) == 2
    assert cycle_features[0]["cycle"] == 1
    assert cycle_features[1]["cycle"] == 2

    # Cycle 1 has shift 0.010, cycle 2 has shift 0.020 -> distinct metrics
    assert cycle_features[0]["peakAbsDx"] < cycle_features[1]["peakAbsDx"]

    # Global aggregation
    aggregated = features["aggregatedFeatures"]
    assert "global_meanDx" in aggregated
    assert "global_maxPeakAbsDx" in aggregated
    assert aggregated["global_maxPeakAbsDx"] == pytest.approx(cycle_features[1]["peakAbsDx"], abs=1e-4)


# =====================================================================
# Test 14: Health Check Endpoint
# =====================================================================
def test_health_check_endpoint():
    """Verify GET /health responds with 200 OK and expected structure."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "remicare-strabismus-ai"
    assert data["version"] == "0.1.0"


# =====================================================================
# Test 15: HTTP API Integration Test
# =====================================================================
def test_analyze_endpoint_http():
    """Verify POST /api/v1/screening/analyze end-to-end via FastAPI client."""
    client = TestClient(app)
    req = ScreeningRequest(
        sampleId="http-test-001",
        test="COVER_TEST",
        cycles=[
            create_standard_cycle(1, "LEFT", "RIGHT"),
            create_standard_cycle(2, "RIGHT", "LEFT"),
        ],
    )

    response = client.post("/api/v1/screening/analyze", json=req.model_dump())
    assert response.status_code == 200

    data = response.json()
    assert data["sampleId"] == "http-test-001"
    assert data["quality"]["status"] == "PASS"
    assert data["analysis"]["cyclesAnalyzed"] == 2
    assert data["status"] == "SCREENING_INCONCLUSIVE"
    assert "Screening result only — not a diagnosis." in data["notice"]
