import base64
import io
import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.schemas.research_measurement import ResearchMeasurementRequest
from app.services.research_measurement_service import (
    FEATURE_VERSION,
    ResearchMeasurementError,
    detect_pupil_in_roi,
    detect_reflexes_in_roi,
    measure_research_request,
)


def _eligibility():
    return {"consent": True, "ageYears": 9, "redFlag": False}


def _cover_sample(t, phase, u, quality=0.95):
    temporal = 0.40
    nasal = 0.60
    return {
        "realTimestampMs": t,
        "t": t,
        "phase": phase,
        "iris_x": temporal + u * (nasal - temporal),
        "eye_corner": {
            "temporal": {"x": temporal, "y": 0.50},
            "nasal": {"x": nasal, "y": 0.50},
        },
        "visibility": {"trackedStatus": "VISIBLE", "trackedEye": "RIGHT"},
        "trackingQuality": quality,
        "frameValid": True,
    }


def test_cover_measurement_returns_experimental_review_required_for_synthetic_movement():
    samples = [
        _cover_sample(0, "BASELINE", 0.50),
        _cover_sample(100, "BASELINE", 0.50),
        _cover_sample(200, "BASELINE", 0.51),
        _cover_sample(300, "TRACKING", 0.54),
        _cover_sample(400, "TRACKING", 0.63),
        _cover_sample(500, "TRACKING", 0.62),
    ]
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="COVER",
        sessionId="11111111-1111-4111-8111-111111111111",
        requestId="cover-synthetic",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        samples=samples,
    )

    result = measure_research_request(req)

    assert result["experimental"] is True
    assert result["status"] == "COMPLETED"
    assert result["result"] == "REVIEW_REQUIRED"
    assert result["measurements"]["peakAbsoluteAmplitudeU"] >= 0.08
    assert "COVER_EXPERIMENTAL_ONLY" in result["reasonCodes"]


def test_cover_measurement_rejects_duplicate_timestamp():
    samples = [
        _cover_sample(0, "BASELINE", 0.50),
        _cover_sample(100, "BASELINE", 0.50),
        _cover_sample(100, "TRACKING", 0.60),
        _cover_sample(200, "TRACKING", 0.61),
        _cover_sample(300, "TRACKING", 0.62),
        _cover_sample(400, "TRACKING", 0.62),
    ]
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="COVER",
        sessionId="11111111-1111-4111-8111-111111111111",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        samples=samples,
    )

    with pytest.raises(ResearchMeasurementError) as exc:
        measure_research_request(req)

    assert exc.value.code == "TIMESERIES_INVALID"


def _synthetic_hirschberg_payload(with_pupil=False, num_reflexes=1):
    image = Image.new("RGB", (120, 80), "black")
    draw = ImageDraw.Draw(image)
    right_center = (42, 40)
    left_center = (78, 40)
    iris_r = 14
    pupil_r = 6

    # Irises
    draw.ellipse((right_center[0] - iris_r, right_center[1] - iris_r, right_center[0] + iris_r, right_center[1] + iris_r), fill=(100, 80, 60))
    draw.ellipse((left_center[0] - iris_r, left_center[1] - iris_r, left_center[0] + iris_r, left_center[1] + iris_r), fill=(100, 80, 60))

    if with_pupil:
        # Pupils (dark circular apertures)
        draw.ellipse((right_center[0] - pupil_r, right_center[1] - pupil_r, right_center[0] + pupil_r, right_center[1] + pupil_r), fill=(15, 15, 15))
        draw.ellipse((left_center[0] - pupil_r, left_center[1] - pupil_r, left_center[0] + pupil_r, left_center[1] + pupil_r), fill=(15, 15, 15))

    if num_reflexes == 1:
        # Exactly one bright reflex per eye
        draw.rectangle((right_center[0] - 2, right_center[1] - 2, right_center[0], right_center[1]), fill=(255, 255, 255))
        draw.rectangle((left_center[0], left_center[1] - 2, left_center[0] + 2, left_center[1]), fill=(255, 255, 255))
    elif num_reflexes > 1:
        # Multiple bright reflexes per eye (e.g. two light spots)
        draw.rectangle((right_center[0] - 4, right_center[1] - 2, right_center[0] - 2, right_center[1]), fill=(255, 255, 255))
        draw.rectangle((right_center[0] + 2, right_center[1] - 2, right_center[0] + 4, right_center[1]), fill=(255, 255, 255))
        draw.rectangle((left_center[0] - 4, left_center[1] - 2, left_center[0] - 2, left_center[1]), fill=(255, 255, 255))
        draw.rectangle((left_center[0] + 2, left_center[1] - 2, left_center[0] + 4, left_center[1]), fill=(255, 255, 255))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    landmarks = [{"x": 0.5, "y": 0.5} for _ in range(478)]
    for idx, x, y in [
        (473, right_center[0] / 120, right_center[1] / 80),
        (474, (right_center[0] - iris_r) / 120, right_center[1] / 80),
        (475, (right_center[0] + iris_r) / 120, right_center[1] / 80),
        (476, right_center[0] / 120, (right_center[1] - iris_r) / 80),
        (477, right_center[0] / 120, (right_center[1] + iris_r) / 80),
        (468, left_center[0] / 120, left_center[1] / 80),
        (469, (left_center[0] - iris_r) / 120, left_center[1] / 80),
        (470, (left_center[0] + iris_r) / 120, left_center[1] / 80),
        (471, left_center[0] / 120, (left_center[1] - iris_r) / 80),
        (472, left_center[0] / 120, (left_center[1] + iris_r) / 80),
    ]:
        landmarks[idx] = {"x": x, "y": y}
    return image_data_url, landmarks


def test_hirschberg_measures_reflexes_but_remains_measurement_only():
    image_data_url, landmarks = _synthetic_hirschberg_payload(with_pupil=False, num_reflexes=1)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="11111111-1111-4111-8111-111111111111",
        requestId="hirschberg-synthetic",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": landmarks},
    )

    result = measure_research_request(req)

    assert result["status"] == "INCONCLUSIVE"
    assert result["result"] == "MEASUREMENT_ONLY"
    assert result["measurements"]["eyes"]["OD"]["reflex_count"] == 1
    assert result["measurements"]["eyes"]["OS"]["reflex_count"] == 1
    assert result["versions"]["modelVersion"] is None


# ==============================================================================
# Phase 6A Specific Unit Tests
# ==============================================================================


def test_detect_reflex_single_spot():
    """Unit test: Detect reflex when a clear single bright spot is present in eye ROI."""
    im = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(im)
    cx, cy, iris_diameter = 50.0, 50.0, 30.0
    # Dark iris
    draw.ellipse((35, 35, 65, 65), fill=(80, 60, 40))
    # Single bright spot (3x3 white pixels)
    draw.rectangle((48, 48, 51, 51), fill=(255, 255, 255))
    arr = np.array(im)

    candidates, status, tier = detect_reflexes_in_roi(arr, cx, cy, iris_diameter)

    assert status == "DETECTED"
    assert len(candidates) == 1
    assert tier == "compact_specular"
    assert abs(candidates[0]["x"] - 49.5) < 1.0
    assert abs(candidates[0]["y"] - 49.5) < 1.0


def test_detect_reflex_not_found():
    """Unit test: LOW_PEAK_BRIGHTNESS when no bright spot exists in eye ROI."""
    im = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(im)
    cx, cy, iris_diameter = 50.0, 50.0, 30.0
    # Dim iris with no bright specular highlight
    draw.ellipse((35, 35, 65, 65), fill=(70, 50, 30))
    arr = np.array(im)

    candidates, status, tier = detect_reflexes_in_roi(arr, cx, cy, iris_diameter)

    assert status == "LOW_PEAK_BRIGHTNESS"
    assert len(candidates) == 0
    assert tier is None


def test_detect_reflex_clustered_spots():
    """Unit test: clustered bright glints are rejected as ambiguous."""
    im = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(im)
    cx, cy, iris_diameter = 50.0, 50.0, 30.0
    draw.ellipse((35, 35, 65, 65), fill=(80, 60, 40))
    # Two close, similarly strong spots.
    draw.rectangle((45, 48, 47, 50), fill=(255, 255, 255))
    draw.rectangle((51, 48, 53, 50), fill=(255, 255, 255))
    arr = np.array(im)

    candidates, status, tier = detect_reflexes_in_roi(arr, cx, cy, iris_diameter)

    assert status == "CLUSTERED_REFLEX_CANDIDATES"
    assert candidates == []
    assert tier is None


def test_detect_reflex_rejects_large_or_elongated_glare():
    """Unit test: long bright glare streak is rejected."""
    im = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(im)
    cx, cy, iris_diameter = 50.0, 50.0, 30.0
    draw.ellipse((35, 35, 65, 65), fill=(80, 60, 40))
    draw.rectangle((35, 48, 70, 50), fill=(255, 255, 255))
    arr = np.array(im)

    candidates, status, tier = detect_reflexes_in_roi(arr, cx, cy, iris_diameter)

    assert status == "LARGE_OR_ELONGATED_GLARE"
    assert candidates == []
    assert tier is None


def test_detect_reflex_rejects_candidate_too_far_from_iris():
    """Unit test: bright candidate inside broad search but outside strict iris region is rejected."""
    im = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(im)
    cx, cy, iris_diameter = 50.0, 50.0, 30.0
    draw.ellipse((35, 35, 65, 65), fill=(80, 60, 40))
    draw.rectangle((71, 48, 73, 50), fill=(255, 255, 255))
    arr = np.array(im)

    candidates, status, tier = detect_reflexes_in_roi(arr, cx, cy, iris_diameter)

    assert status == "REFLEX_CANDIDATE_TOO_FAR"
    assert candidates == []
    assert tier is None


def test_detect_pupil_not_found_when_insufficient_data():
    """Unit test: pupil detector returns PUPIL_NOT_FOUND when ROI lacks data / low contrast."""
    # Case A: Uniform gray image (no contrast difference between pupil and iris)
    im_flat = Image.new("RGB", (80, 80), (100, 100, 100))
    arr_flat = np.array(im_flat)
    pupil_c, pupil_d, status_flat = detect_pupil_in_roi(arr_flat, 40.0, 40.0, 24.0)

    assert status_flat == "PUPIL_NOT_FOUND"
    assert pupil_c is None
    assert pupil_d is None

    # Case B: Tiny out-of-bounds / insufficient pixels ROI
    im_tiny = Image.new("RGB", (10, 10), "black")
    arr_tiny = np.array(im_tiny)
    pupil_c_tiny, pupil_d_tiny, status_tiny = detect_pupil_in_roi(arr_tiny, 1.0, 1.0, 4.0)

    assert status_tiny in {"PUPIL_NOT_FOUND", "LOW_QUALITY_INPUT"}
    assert pupil_c_tiny is None


def test_hirschberg_parallel_iris_and_pupil_measurements_and_schema():
    """Unit test: Full Hirschberg request with both pupils and reflexes returns parallel fields and backward-compatible schema."""
    image_data_url, landmarks = _synthetic_hirschberg_payload(with_pupil=True, num_reflexes=1)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="22222222-2222-4222-8222-222222222222",
        requestId="hirschberg-full-parallel",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": landmarks},
    )

    result = measure_research_request(req)

    # 1. Measurement only / inconclusive, strictly not normal clearance
    assert result["status"] == "INCONCLUSIVE"
    assert result["result"] == "MEASUREMENT_ONLY"
    assert result["experimental"] is True

    # 2. Both eyes have parallel iris & pupil measurements
    od = result["measurements"]["eyes"]["OD"]
    os = result["measurements"]["eyes"]["OS"]

    assert od["pupil_status"] == "DETECTED"
    assert od["reflex_status"] == "DETECTED"
    assert od["status"] == "MEASURED"
    assert od["iris_center"] is not None
    assert od["pupil_center"] is not None
    assert od["reflex_center"] is not None
    assert od["iris_diameter"] > 0
    assert od["h_iris"] is not None
    assert od["h_pupil"] is not None

    assert os["pupil_status"] == "DETECTED"
    assert os["reflex_status"] == "DETECTED"
    assert os["status"] == "MEASURED"
    assert os["iris_center"] is not None
    assert os["pupil_center"] is not None
    assert os["reflex_center"] is not None
    assert os["iris_diameter"] > 0
    assert os["h_iris"] is not None
    assert os["h_pupil"] is not None

    # 3. Delta measurements
    assert result["measurements"]["delta_h"] is not None
    assert result["measurements"]["delta_h_iris"] is not None
    assert result["measurements"]["delta_h_pupil"] is not None
    assert result["measurements"]["delta_h"] == result["measurements"]["delta_h_iris"]

    # 4. Separate detection success tracking for face, eyes, iris, pupil, reflex
    detection_stats = result["measurements"]["detectionStats"]
    assert detection_stats["faceDetected"] is True
    assert detection_stats["eyesDetected"] is True
    assert detection_stats["irisDetected"] is True
    assert detection_stats["pupilDetected"] is True
    assert detection_stats["reflexDetected"] is True

    quality_success = result["quality"]["detectorSuccess"]
    assert quality_success["face"] is True
    assert quality_success["eyes"] is True
    assert quality_success["iris"] is True
    assert quality_success["pupil"] is True
    assert quality_success["reflex"] is True

    # 5. Schema backwards compatibility
    assert "iris_center_x_px" in od
    assert "iris_center_y_px" in od
    assert "iris_diameter_px" in od
    assert "reflex_count" in od
    assert "h" in od
    assert od["h"] == od["h_iris"]


def test_hirschberg_missing_reflex_reason_codes():
    """Unit test: Missing corneal reflex produces REFLEX_NOT_FOUND reason and MEASUREMENT_ONLY."""
    image_data_url, landmarks = _synthetic_hirschberg_payload(with_pupil=True, num_reflexes=0)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="33333333-3333-4333-8333-333333333333",
        requestId="hirschberg-no-reflex",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": landmarks},
    )

    result = measure_research_request(req)

    assert result["status"] == "INCONCLUSIVE"
    assert result["result"] == "MEASUREMENT_ONLY"
    assert "REFLEX_NOT_FOUND" in result["reasonCodes"]
    assert "LOW_PEAK_BRIGHTNESS" in result["reasonCodes"]
    assert result["quality"]["detectorSuccess"]["reflex"] is False


def test_hirschberg_clustered_reflex_reason_codes():
    """Unit test: clustered corneal reflexes produce detailed reason and MEASUREMENT_ONLY."""
    image_data_url, landmarks = _synthetic_hirschberg_payload(with_pupil=True, num_reflexes=2)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="44444444-4444-4444-8444-444444444444",
        requestId="hirschberg-multi-reflex",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": landmarks},
    )

    result = measure_research_request(req)

    assert result["status"] == "INCONCLUSIVE"
    assert result["result"] == "MEASUREMENT_ONLY"
    assert "REFLEX_NOT_FOUND" in result["reasonCodes"]
    assert "CLUSTERED_REFLEX_CANDIDATES" in result["reasonCodes"]
    assert result["quality"]["detectorSuccess"]["reflex"] is False


def test_hirschberg_includes_ai_prediction():
    """Unit test: Hirschberg request runs trained candidate AI model and returns predicted class and probabilities."""
    image_data_url, landmarks = _synthetic_hirschberg_payload(with_pupil=True, num_reflexes=1)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="55555555-5555-5555-5555-555555555555",
        requestId="hirschberg-with-ai",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": landmarks},
    )

    result = measure_research_request(req)

    assert "aiPrediction" in result
    assert "aiPrediction" in result["measurements"]
    ai = result["aiPrediction"]
    assert ai["status"] == "PREDICTED"
    assert ai["predictedClass"] in {"NORMAL", "ESOTROPIA", "EXOTROPIA"}
    assert 0.0 <= ai["confidence"] <= 1.0
    assert "probabilities" in ai
    assert "modelId" in ai
    assert ai["modelId"] == "hirschberg-candidate-v0.1"


def test_hirschberg_without_landmarks_runs_ai_fallback():
    """Unit test: Request without 478 landmarks does not raise error, but runs AI inference gracefully."""
    image_data_url, _ = _synthetic_hirschberg_payload(with_pupil=True, num_reflexes=1)
    req = ResearchMeasurementRequest(
        schemaVersion="remicare-research-quality-v0.1",
        featureVersion=FEATURE_VERSION,
        testType="HIRSCHBERG",
        sessionId="66666666-6666-6666-6666-666666666666",
        requestId="hirschberg-no-landmarks",
        distance_bucket="TARGET_20_25_CM",
        eligibility=_eligibility(),
        imageDataUrl=image_data_url,
        metadata={"landmarks": []},  # empty landmarks
    )

    result = measure_research_request(req)

    assert result["status"] == "INCONCLUSIVE"
    assert result["result"] == "MEASUREMENT_ONLY"
    assert "CLIENT_LANDMARKS_NOT_PROVIDED_AI_INFERRED" in result["reasonCodes"]
    assert "aiPrediction" in result
    assert result["aiPrediction"]["status"] == "PREDICTED"
    assert result["aiPrediction"]["predictedClass"] in {"NORMAL", "ESOTROPIA", "EXOTROPIA"}
