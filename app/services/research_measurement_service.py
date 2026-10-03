"""Research-only Hirschberg and Cover geometry measurements.

This module deliberately does not call or modify the production ONNX/image model.
All thresholds are TODO_PILOT and outputs remain measurement/experimental only.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
import uuid
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.schemas.research_measurement import ResearchMeasurementRequest

logger = logging.getLogger("remicare.research_measurement")

SUPPORTED_SCHEMA_VERSION = "remicare-research-quality-v0.1"
FEATURE_VERSION = "research-geometry-v0.1"
PREPROCESSING_VERSION = "research-geometry-preprocess-v0.1"
CONFIG_VERSION = "TODO_PILOT"

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 2560 * 1440
MAX_TIMESTAMP_GAP_MS = 600.0
MIN_AGE_YEARS = 7
MIN_VALID_FRAME_RATIO = 0.65
MIN_TRACKING_CONFIDENCE = 0.65
MIN_COVER_SAMPLES = 6
MIN_BASELINE_SAMPLES = 3
COVER_AMPLITUDE_TODO_PILOT = 0.08

LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
LEFT_CORNERS = {"nasal": 362, "temporal": 263}
RIGHT_CORNERS = {"nasal": 133, "temporal": 33}

# ==============================================================================
# Phase 6A Research Thresholds (All thresholds are TODO_PILOT pending clinical pilot)
# ==============================================================================

# Multi-threshold tiers for corneal light reflex detection (evaluated from strict to adaptive)
REFLEX_MULTI_THRESHOLDS_TODO_PILOT: List[Dict[str, Any]] = [
    {
        "tier": "strict_specular",
        "min_rgb": 235,
        "min_luma": 225.0,
        "threshold_source": "TODO_PILOT",
    },
    {
        "tier": "medium_specular",
        "min_rgb": 215,
        "min_luma": 205.0,
        "threshold_source": "TODO_PILOT",
    },
    {
        "tier": "adaptive_specular",
        "min_rgb": 195,
        "min_luma": 185.0,
        "threshold_source": "TODO_PILOT",
    },
]

REFLEX_MIN_PIXELS_TODO_PILOT = 2
REFLEX_MAX_PIXELS_TODO_PILOT = 120
REFLEX_SEARCH_RADIUS_FACTOR_TODO_PILOT = 1.35  # Relative to iris radius
REFLEX_MAX_ASPECT_RATIO_TODO_PILOT = 3.5       # Exclude elongated glare streaks

# Pupil Center Detection constants (all exploratory & TODO_PILOT)
PUPIL_SEARCH_RADIUS_FACTOR_TODO_PILOT = 0.85   # Search inside 85% of iris radius
PUPIL_MIN_RADIUS_RATIO_TODO_PILOT = 0.15       # Minimum pupil radius as fraction of iris diameter
PUPIL_MAX_RADIUS_RATIO_TODO_PILOT = 0.70       # Maximum pupil radius as fraction of iris diameter
PUPIL_MIN_AREA_PX_TODO_PILOT = 8               # Minimum pixel area for pupil candidate
PUPIL_MAX_OFFSET_FROM_IRIS_TODO_PILOT = 0.25   # Pupil center offset <= 25% iris diameter from iris center
PUPIL_MIN_CIRCULARITY_TODO_PILOT = 0.35        # 4*pi*area / perimeter^2
PUPIL_MIN_CONTRAST_DIFF_TODO_PILOT = 12.0      # Minimum contrast difference between pupil and surrounding iris
PUPIL_THRESHOLD_PERCENTILES_TODO_PILOT = [18.0, 28.0, 38.0]  # Intensity percentiles inside iris ROI
PUPIL_MAX_DARK_LUMA_TODO_PILOT = 95.0          # Absolute upper bound for pupil pixel luma


class ResearchMeasurementError(Exception):
    """Safe user-facing research measurement error."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class EyeMeasurement:
    eye: str
    iris_center_x_px: float
    iris_center_y_px: float
    iris_diameter_px: float
    reflex_count: int
    reflex_x_px: Optional[float]
    reflex_y_px: Optional[float]
    h: Optional[float]
    status: str
    # Phase 6A parallel measurement & detector statuses:
    pupil_center_x_px: Optional[float] = None
    pupil_center_y_px: Optional[float] = None
    pupil_diameter_px: Optional[float] = None
    pupil_status: str = "PUPIL_NOT_FOUND"
    reflex_status: str = "REFLEX_NOT_FOUND"
    h_iris: Optional[float] = None
    h_pupil: Optional[float] = None
    displacement_iris_raw: Optional[float] = None
    displacement_pupil_raw: Optional[float] = None
    iris_center: Optional[Dict[str, float]] = None
    pupil_center: Optional[Dict[str, float]] = None
    reflex_center: Optional[Dict[str, float]] = None
    iris_diameter: Optional[float] = None
    reflex_tier: Optional[str] = None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_request_id(request_id: Optional[str]) -> str:
    return request_id or str(uuid.uuid4())


def _validate_common(req: ResearchMeasurementRequest) -> List[str]:
    reasons: List[str] = []
    if req.schemaVersion != SUPPORTED_SCHEMA_VERSION:
        raise ResearchMeasurementError(
            "FEATURE_CONTRACT_MISMATCH",
            f"Unsupported schemaVersion '{req.schemaVersion}'.",
        )
    if not req.sessionId:
        raise ResearchMeasurementError("INVALID_REQUEST", "sessionId is required.")
    if req.featureVersion not in {FEATURE_VERSION, "research-quality-v0.1", SUPPORTED_SCHEMA_VERSION}:
        reasons.append("FEATURE_VERSION_UNRECOGNIZED_RECORDED_ONLY")

    eligibility = req.eligibility
    if not eligibility.consent:
        raise ResearchMeasurementError("INVALID_REQUEST", "Research consent is required.")
    if eligibility.ageYears is None or eligibility.ageYears < MIN_AGE_YEARS:
        raise ResearchMeasurementError("INVALID_REQUEST", "Age is outside the supported screening range.")
    if eligibility.redFlag:
        raise ResearchMeasurementError("INVALID_REQUEST", "Red flag present; measurement must not run.")
    return reasons


def _extract_image_bytes(req: ResearchMeasurementRequest) -> bytes:
    raw = req.imageDataUrl or req.imageBase64
    if not raw:
        raise ResearchMeasurementError("INVALID_REQUEST", "HIRSCHBERG request requires imageDataUrl or imageBase64.")

    if "," in raw and raw.strip().lower().startswith("data:"):
        header, raw = raw.split(",", 1)
        if "image/" not in header:
            raise ResearchMeasurementError("INVALID_REQUEST", "imageDataUrl must be an image/* data URL.")

    try:
        payload = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ResearchMeasurementError("INVALID_REQUEST", "Image payload is not valid base64.") from exc

    if len(payload) > MAX_IMAGE_BYTES:
        raise ResearchMeasurementError("INVALID_REQUEST", "Image payload exceeds research limit.")
    return payload


def _decode_image(req: ResearchMeasurementRequest) -> np.ndarray:
    payload = _extract_image_bytes(req)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            if rgb.width * rgb.height > MAX_IMAGE_PIXELS:
                raise ResearchMeasurementError("INVALID_REQUEST", "Image pixel count exceeds research limit.")
            return np.asarray(rgb)
    except ResearchMeasurementError:
        raise
    except Exception as exc:
        raise ResearchMeasurementError("INVALID_REQUEST", "Image could not be decoded.") from exc


def _landmark_point(landmarks: List[Dict[str, Any]], index: int, width: int, height: int) -> Optional[Tuple[float, float]]:
    if index >= len(landmarks):
        return None
    point = landmarks[index]
    x = point.get("x")
    y = point.get("y")
    if not _is_finite_number(x) or not _is_finite_number(y):
        return None
    if float(x) < 0.0 or float(x) > 1.0 or float(y) < 0.0 or float(y) > 1.0:
        return None
    return float(x) * width, float(y) * height


def _iris_geometry(
    landmarks: List[Dict[str, Any]],
    indices: Iterable[int],
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float]]:
    points = [_landmark_point(landmarks, idx, width, height) for idx in indices]
    if any(point is None for point in points):
        return None
    xs = [p[0] for p in points if p is not None]
    ys = [p[1] for p in points if p is not None]
    cx = float(sum(xs) / len(xs))
    cy = float(sum(ys) / len(ys))
    diameter = max(max(xs) - min(xs), max(ys) - min(ys))
    if diameter <= 1.0:
        return None
    return cx, cy, diameter


def detect_reflexes_in_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    iris_diameter: float,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Detect corneal light reflexes (Purkinje images) within eye ROI across multi-threshold tiers.

    Returns:
        (candidates, reflex_status, reflex_tier)
        reflex_status: "DETECTED", "REFLEX_NOT_FOUND", "MULTIPLE_REFLEX", "LOW_QUALITY_INPUT"
    """
    height, width, _ = image.shape
    radius = max(8.0, (iris_diameter / 2.0) * REFLEX_SEARCH_RADIUS_FACTOR_TODO_PILOT)
    x0 = max(0, int(round(cx - radius)))
    x1 = min(width, int(round(cx + radius)))
    y0 = max(0, int(round(cy - radius)))
    y1 = min(height, int(round(cy + radius)))
    if x1 <= x0 or y1 <= y0:
        return [], "LOW_QUALITY_INPUT", None

    crop = image[y0:y1, x0:x1, :]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circular = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
    luma = (0.299 * crop[:, :, 0]) + (0.587 * crop[:, :, 1]) + (0.114 * crop[:, :, 2])

    for tier_cfg in REFLEX_MULTI_THRESHOLDS_TODO_PILOT:
        min_rgb = tier_cfg["min_rgb"]
        min_luma = tier_cfg["min_luma"]
        tier_name = tier_cfg["tier"]

        bright = (
            circular
            & (crop[:, :, 0] >= min_rgb)
            & (crop[:, :, 1] >= min_rgb)
            & (crop[:, :, 2] >= min_rgb)
            & (luma >= min_luma)
        )
        if not np.any(bright):
            continue

        visited = np.zeros(bright.shape, dtype=bool)
        candidates: List[Dict[str, Any]] = []
        h, w = bright.shape
        for y in range(h):
            for x in range(w):
                if visited[y, x] or not bright[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                pixels: List[Tuple[int, int]] = []
                while stack:
                    px, py = stack.pop()
                    pixels.append((px, py))
                    for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and bright[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

                if REFLEX_MIN_PIXELS_TODO_PILOT <= len(pixels) <= REFLEX_MAX_PIXELS_TODO_PILOT:
                    xs = [p[0] for p in pixels]
                    ys = [p[1] for p in pixels]
                    w_box = max(xs) - min(xs) + 1
                    h_box = max(ys) - min(ys) + 1
                    aspect_ratio = max(w_box, h_box) / max(1, min(w_box, h_box))
                    if aspect_ratio <= REFLEX_MAX_ASPECT_RATIO_TODO_PILOT:
                        mean_x = x0 + sum(xs) / len(pixels)
                        mean_y = y0 + sum(ys) / len(pixels)
                        candidates.append({
                            "x": float(mean_x),
                            "y": float(mean_y),
                            "pixel_count": len(pixels),
                            "tier": tier_name,
                            "threshold_source": "TODO_PILOT",
                        })

        if len(candidates) == 1:
            return candidates, "DETECTED", tier_name
        elif len(candidates) > 1:
            return candidates, "MULTIPLE_REFLEX", tier_name

    return [], "REFLEX_NOT_FOUND", None


# Backwards compatibility alias
def _detect_reflexes(image: np.ndarray, cx: float, cy: float, radius: float) -> List[Tuple[float, float, int]]:
    candidates, _status, _tier = detect_reflexes_in_roi(image, cx, cy, radius * 2.0 / REFLEX_SEARCH_RADIUS_FACTOR_TODO_PILOT)
    return [(c["x"], c["y"], c["pixel_count"]) for c in candidates]


def detect_pupil_in_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    iris_diameter: float,
) -> Tuple[Optional[Dict[str, float]], Optional[float], str]:
    """Detect the pupil center within the iris ROI.

    Returns:
        (pupil_center_dict, pupil_diameter_px, pupil_status)
        pupil_status: "DETECTED", "PUPIL_NOT_FOUND", "LOW_QUALITY_INPUT"
    """
    height, width, _ = image.shape
    search_radius = max(6.0, (iris_diameter / 2.0) * PUPIL_SEARCH_RADIUS_FACTOR_TODO_PILOT)
    x0 = max(0, int(round(cx - search_radius)))
    x1 = min(width, int(round(cx + search_radius)))
    y0 = max(0, int(round(cy - search_radius)))
    y1 = min(height, int(round(cy + search_radius)))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, None, "LOW_QUALITY_INPUT"

    crop = image[y0:y1, x0:x1, :]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circular_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= search_radius**2
    luma = (0.299 * crop[:, :, 0]) + (0.587 * crop[:, :, 1]) + (0.114 * crop[:, :, 2])

    # Specular mask: exclude bright corneal reflexes from pupil intensity calculations
    bright_mask = (crop[:, :, 0] >= 180) & (crop[:, :, 1] >= 180) & (crop[:, :, 2] >= 180) & (luma >= 180)
    valid_iris_pixels = circular_mask & (~bright_mask)
    if np.count_nonzero(valid_iris_pixels) < 12:
        return None, None, "PUPIL_NOT_FOUND"

    iris_luma = luma[valid_iris_pixels]
    p15 = float(np.percentile(iris_luma, 15))
    p85 = float(np.percentile(iris_luma, 85))
    contrast_diff = p85 - p15
    if contrast_diff < PUPIL_MIN_CONTRAST_DIFF_TODO_PILOT:
        # Uniform or low-contrast ROI (e.g. synthetic flat circle or washed out)
        return None, None, "PUPIL_NOT_FOUND"

    iris_area = math.pi * (iris_diameter / 2.0) ** 2
    min_pupil_area = max(PUPIL_MIN_AREA_PX_TODO_PILOT, iris_area * (PUPIL_MIN_RADIUS_RATIO_TODO_PILOT ** 2))
    max_pupil_area = iris_area * (PUPIL_MAX_RADIUS_RATIO_TODO_PILOT ** 2)

    best_candidate: Optional[Dict[str, Any]] = None
    h, w = crop.shape[:2]

    for p_val in PUPIL_THRESHOLD_PERCENTILES_TODO_PILOT:
        thresh = min(float(np.percentile(iris_luma, p_val)), PUPIL_MAX_DARK_LUMA_TODO_PILOT)
        dark_mask = valid_iris_pixels & (luma <= thresh)
        if not np.any(dark_mask):
            continue

        visited = np.zeros(dark_mask.shape, dtype=bool)
        for y in range(h):
            for x in range(w):
                if visited[y, x] or not dark_mask[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                comp_pixels: List[Tuple[int, int]] = []
                while stack:
                    px, py = stack.pop()
                    comp_pixels.append((px, py))
                    for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and dark_mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

                area = len(comp_pixels)
                if not (min_pupil_area <= area <= max_pupil_area):
                    continue

                xs = [p[0] for p in comp_pixels]
                ys = [p[1] for p in comp_pixels]
                mean_x = x0 + sum(xs) / area
                mean_y = y0 + sum(ys) / area

                # Offset from iris center
                offset = math.hypot(mean_x - cx, mean_y - cy)
                if offset > PUPIL_MAX_OFFSET_FROM_IRIS_TODO_PILOT * iris_diameter:
                    continue

                # Estimate perimeter via boundary pixels
                pixel_set = set(comp_pixels)
                boundary_count = 0
                for px, py in comp_pixels:
                    if (
                        (px + 1, py) not in pixel_set
                        or (px - 1, py) not in pixel_set
                        or (px, py + 1) not in pixel_set
                        or (px, py - 1) not in pixel_set
                    ):
                        boundary_count += 1
                perimeter = max(1.0, float(boundary_count))
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                if circularity < PUPIL_MIN_CIRCULARITY_TODO_PILOT:
                    continue

                equiv_diameter = 2.0 * math.sqrt(area / math.pi)
                # Score balances circularity and proximity to expected center
                score = circularity - (offset / iris_diameter)

                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {
                        "x": float(mean_x),
                        "y": float(mean_y),
                        "diameter": float(equiv_diameter),
                        "score": score,
                    }

        if best_candidate is not None:
            break

    if best_candidate is None:
        return None, None, "PUPIL_NOT_FOUND"

    return (
        {"x": round(best_candidate["x"], 3), "y": round(best_candidate["y"], 3)},
        round(best_candidate["diameter"], 3),
        "DETECTED",
    )


def _measure_eye(image: np.ndarray, landmarks: List[Dict[str, Any]], eye: str) -> EyeMeasurement:
    height, width, _ = image.shape
    indices = LEFT_IRIS if eye == "left" else RIGHT_IRIS
    geom = _iris_geometry(landmarks, indices, width, height)
    if not geom:
        return EyeMeasurement(
            eye=eye,
            iris_center_x_px=0.0,
            iris_center_y_px=0.0,
            iris_diameter_px=0.0,
            reflex_count=0,
            reflex_x_px=None,
            reflex_y_px=None,
            h=None,
            status="INVALID_IRIS",
            pupil_status="LOW_QUALITY_INPUT",
            reflex_status="LOW_QUALITY_INPUT",
        )

    cx, cy, diameter = geom
    reflex_candidates, reflex_status, reflex_tier = detect_reflexes_in_roi(image, cx, cy, diameter)
    reflex_count = len(reflex_candidates)
    pupil_c, pupil_d, pupil_status = detect_pupil_in_roi(image, cx, cy, diameter)

    # + means nasal. OD/right nasal is image-left in ordinary frontal images; OS/left nasal is image-right.
    nasal_sign = -1.0 if eye == "right" else 1.0

    reflex_x: Optional[float] = None
    reflex_y: Optional[float] = None
    reflex_center: Optional[Dict[str, float]] = None
    h_iris: Optional[float] = None
    h_pupil: Optional[float] = None
    disp_iris_raw: Optional[float] = None
    disp_pupil_raw: Optional[float] = None

    if reflex_status == "DETECTED" and reflex_count == 1:
        reflex_x = reflex_candidates[0]["x"]
        reflex_y = reflex_candidates[0]["y"]
        reflex_center = {"x": round(reflex_x, 3), "y": round(reflex_y, 3)}

        # Parallel calculations:
        # h_iris = nasal_sign * (reflex_x - iris_center_x) / iris_diameter
        h_iris = round(nasal_sign * (reflex_x - cx) / diameter, 5)
        disp_iris_raw = round((reflex_x - cx) / diameter, 5)

        # h_pupil = nasal_sign * (reflex_x - pupil_center_x) / iris_diameter
        if pupil_status == "DETECTED" and pupil_c is not None:
            h_pupil = round(nasal_sign * (reflex_x - pupil_c["x"]) / diameter, 5)
            disp_pupil_raw = round((reflex_x - pupil_c["x"]) / diameter, 5)

    # Eye overall status
    if reflex_status == "DETECTED" and pupil_status == "DETECTED":
        status = "MEASURED"
    elif reflex_status == "MULTIPLE_REFLEX":
        status = "MULTIPLE_REFLEX"
    elif reflex_status == "REFLEX_NOT_FOUND":
        status = "REFLEX_NOT_FOUND"
    elif pupil_status == "PUPIL_NOT_FOUND":
        status = "PUPIL_NOT_FOUND"
    elif reflex_status == "LOW_QUALITY_INPUT" or pupil_status == "LOW_QUALITY_INPUT":
        status = "LOW_QUALITY_INPUT"
    else:
        status = "INCONCLUSIVE_REFLEX"

    return EyeMeasurement(
        eye=eye,
        iris_center_x_px=round(cx, 3),
        iris_center_y_px=round(cy, 3),
        iris_diameter_px=round(diameter, 3),
        reflex_count=reflex_count,
        reflex_x_px=round(reflex_x, 3) if reflex_x is not None else None,
        reflex_y_px=round(reflex_y, 3) if reflex_y is not None else None,
        h=h_iris,  # backwards-compatible alias
        status=status,
        pupil_center_x_px=pupil_c["x"] if pupil_c else None,
        pupil_center_y_px=pupil_c["y"] if pupil_c else None,
        pupil_diameter_px=pupil_d,
        pupil_status=pupil_status,
        reflex_status=reflex_status,
        h_iris=h_iris,
        h_pupil=h_pupil,
        displacement_iris_raw=disp_iris_raw,
        displacement_pupil_raw=disp_pupil_raw,
        iris_center={"x": round(cx, 3), "y": round(cy, 3)},
        pupil_center=pupil_c,
        reflex_center=reflex_center,
        iris_diameter=round(diameter, 3),
        reflex_tier=reflex_tier,
    )


def measure_hirschberg(req: ResearchMeasurementRequest) -> Dict[str, Any]:
    image = _decode_image(req)
    landmarks = (req.metadata or {}).get("landmarks") or (req.metadata or {}).get("faceLandmarks")
    has_valid_landmarks = isinstance(landmarks, list) and len(landmarks) >= 478

    if has_valid_landmarks:
        right = _measure_eye(image, landmarks, "right")
        left = _measure_eye(image, landmarks, "left")
        eyes = {
            "OD": right.__dict__,
            "OS": left.__dict__,
        }
        face_detected = True
        eyes_detected = right.status != "INVALID_IRIS" and left.status != "INVALID_IRIS"
        iris_detected = right.iris_diameter_px > 1.0 and left.iris_diameter_px > 1.0
        pupil_detected = right.pupil_status == "DETECTED" and left.pupil_status == "DETECTED"
        reflex_detected = right.reflex_count == 1 and left.reflex_count == 1
        measured = right.h_iris is not None and left.h_iris is not None

        reasons: List[str] = []
        if not reflex_detected:
            if right.reflex_status == "MULTIPLE_REFLEX" or left.reflex_status == "MULTIPLE_REFLEX":
                reasons.append("MULTIPLE_REFLEX")
            if right.reflex_status == "REFLEX_NOT_FOUND" or left.reflex_status == "REFLEX_NOT_FOUND":
                reasons.append("REFLEX_NOT_FOUND")
            if "MULTIPLE_REFLEX" not in reasons and "REFLEX_NOT_FOUND" not in reasons:
                reasons.append("REFLEX_COUNT_NOT_EXACTLY_ONE_PER_EYE")

        if not pupil_detected:
            reasons.append("PUPIL_NOT_FOUND")

        if req.distance_bucket == "UNKNOWN":
            reasons.append("DISTANCE_BUCKET_UNKNOWN")

        delta_h_iris = round(float(right.h_iris - left.h_iris), 5) if (right.h_iris is not None and left.h_iris is not None) else None
        delta_h_pupil = (
            round(float(right.h_pupil - left.h_pupil), 5)
            if (right.h_pupil is not None and left.h_pupil is not None)
            else None
        )
        quality_reflex_count = {"OD": right.reflex_count, "OS": left.reflex_count}
        quality_pupil_status = {"OD": right.pupil_status, "OS": left.pupil_status}
        quality_reflex_status = {"OD": right.reflex_status, "OS": left.reflex_status}
    else:
        # Landmarks not provided by client (e.g. uploaded photo or crop)
        eyes = {}
        face_detected = False
        eyes_detected = True
        iris_detected = True
        pupil_detected = False
        reflex_detected = False
        measured = False
        reasons = ["CLIENT_LANDMARKS_NOT_PROVIDED_AI_INFERRED"]
        delta_h_iris = None
        delta_h_pupil = None
        quality_reflex_count = {"OD": 0, "OS": 0}
        quality_pupil_status = {"OD": "NOT_PROVIDED", "OS": "NOT_PROVIDED"}
        quality_reflex_status = {"OD": "NOT_PROVIDED", "OS": "NOT_PROVIDED"}

    # Run AI inference with trained Hirschberg model
    from app.services.hirschberg_ai_service import predict_hirschberg
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ai_prediction = predict_hirschberg(image_bgr, landmarks if has_valid_landmarks else None)

    return {
        "status": "INCONCLUSIVE",
        "result": "MEASUREMENT_ONLY",
        "reasonCodes": reasons or ["THRESHOLDS_TODO_PILOT"],
        "aiPrediction": ai_prediction,
        "measurements": {
            "distanceBucket": req.distance_bucket,
            "eyes": eyes,
            "delta_h": delta_h_iris,
            "delta_h_iris": delta_h_iris,
            "delta_h_pupil": delta_h_pupil,
            "aiPrediction": ai_prediction,
            "detectionStats": {
                "faceDetected": face_detected,
                "eyesDetected": eyes_detected,
                "irisDetected": iris_detected,
                "pupilDetected": pupil_detected,
                "reflexDetected": reflex_detected,
            },
            "signConvention": "+ nasal, - temporal; delta_h = h_OD - h_OS",
            "thresholds": "TODO_PILOT_BY_DISTANCE_BUCKET",
        },
        "quality": {
            "reflexCountPerEye": quality_reflex_count,
            "pupilStatusPerEye": quality_pupil_status,
            "reflexStatusPerEye": quality_reflex_status,
            "serverMeasured": measured,
            "detectorSuccess": {
                "face": face_detected,
                "eyes": eyes_detected,
                "iris": iris_detected,
                "pupil": pupil_detected,
                "reflex": reflex_detected,
            },
        },
    }


def _sample_time(sample: Dict[str, Any]) -> float:
    for key in ("realTimestampMs", "timestamp", "t"):
        value = sample.get(key)
        if _is_finite_number(value):
            return float(value)
    raise ResearchMeasurementError("TIMESERIES_INVALID", "Sample timestamp is missing or non-finite.")


def _corner_x(corner: Dict[str, Any], name: str) -> Optional[float]:
    point = corner.get(name)
    if not isinstance(point, dict):
        return None
    x = point.get("x")
    if not _is_finite_number(x):
        return None
    return float(x)


def _normalized_cover_position(sample: Dict[str, Any]) -> Optional[float]:
    iris_x = sample.get("iris_x")
    corner = sample.get("eye_corner")
    if not _is_finite_number(iris_x) or not isinstance(corner, dict):
        return None

    # Frontend stores tracked eye corners as {inner, outer}; map to nasal/temporal when explicit names are absent.
    nasal_x = _corner_x(corner, "nasal")
    temporal_x = _corner_x(corner, "temporal")
    if nasal_x is None:
        nasal_x = _corner_x(corner, "inner")
    if temporal_x is None:
        temporal_x = _corner_x(corner, "outer")
    if nasal_x is None or temporal_x is None:
        return None
    denominator = nasal_x - temporal_x
    if abs(denominator) < 1e-6:
        return None
    return float((float(iris_x) - temporal_x) / denominator)


def measure_cover(req: ResearchMeasurementRequest) -> Dict[str, Any]:
    samples = [s.model_dump() for s in (req.samples or [])]
    if len(samples) < MIN_COVER_SAMPLES:
        raise ResearchMeasurementError("TIMESERIES_INVALID", "COVER requires a non-empty timestamped sample sequence.")

    reason_codes = ["COVER_EXPERIMENTAL_ONLY"]
    times = [_sample_time(s) for s in samples]
    for idx in range(1, len(times)):
        if times[idx] <= times[idx - 1]:
            raise ResearchMeasurementError("TIMESERIES_INVALID", "Timestamp must be strictly increasing.")
    gaps = [times[idx] - times[idx - 1] for idx in range(1, len(times))]
    max_gap = max(gaps) if gaps else 0.0
    gap_breaks = [idx for idx, gap in enumerate(gaps, start=1) if gap > MAX_TIMESTAMP_GAP_MS]
    if gap_breaks:
        reason_codes.append("TIMESTAMP_GAP_SPLIT_REQUIRED")

    valid_samples = []
    blink_count = 0
    occlusion_count = 0
    quality_sum = 0.0
    quality_count = 0
    positions: List[Tuple[float, str, float]] = []

    for sample, timestamp_ms in zip(samples, times):
        visibility = sample.get("visibility") or {}
        tracked_status = str(visibility.get("trackedStatus") or "").upper()
        if tracked_status in {"BLINK"}:
            blink_count += 1
        if tracked_status in {"OCCLUDED_BY_COVER", "LOST"}:
            occlusion_count += 1
        tq = sample.get("trackingQuality")
        if _is_finite_number(tq):
            quality_sum += float(tq)
            quality_count += 1
        u_value = _normalized_cover_position(sample)
        frame_valid = sample.get("frameValid")
        phase = str(sample.get("phase") or "").upper()
        if u_value is not None and frame_valid is not False and tracked_status not in {"BLINK", "OCCLUDED_BY_COVER", "LOST"}:
            valid_samples.append(sample)
            positions.append((timestamp_ms, phase, u_value))

    valid_ratio = len(valid_samples) / len(samples)
    tracking_confidence = quality_sum / quality_count if quality_count else 0.0
    if valid_ratio < MIN_VALID_FRAME_RATIO or tracking_confidence < MIN_TRACKING_CONFIDENCE:
        return {
            "status": "INCONCLUSIVE",
            "result": "LOW_QUALITY_INPUT",
            "reasonCodes": reason_codes + ["LOW_QUALITY_INPUT"],
            "measurements": {
                "validFrameRatio": round(valid_ratio, 4),
                "trackingConfidence": round(tracking_confidence, 4),
                "maxTimestampGapMs": round(max_gap, 3),
            },
            "quality": {
                "validFrameRatio": round(valid_ratio, 4),
                "trackingConfidence": round(tracking_confidence, 4),
                "blinkRatio": round(blink_count / len(samples), 4),
                "occlusionRatio": round(occlusion_count / len(samples), 4),
            },
        }

    baseline_values = [u for _ts, phase, u in positions if phase == "BASELINE"]
    event_values = [(ts, phase, u) for ts, phase, u in positions if phase in {"TRACKING", "UNCOVER"}]
    if len(baseline_values) < MIN_BASELINE_SAMPLES or not event_values:
        return {
            "status": "INCONCLUSIVE",
            "result": "TIMESERIES_INVALID",
            "reasonCodes": reason_codes + ["BASELINE_OR_TRACKING_MISSING"],
            "measurements": {},
            "quality": {
                "validFrameRatio": round(valid_ratio, 4),
                "trackingConfidence": round(tracking_confidence, 4),
            },
        }

    baseline = float(median(baseline_values))
    deviations = [(ts, phase, u - baseline) for ts, phase, u in event_values]
    peak_ts, peak_phase, peak_dev = max(deviations, key=lambda item: abs(item[2]))
    velocities = []
    for prev, curr in zip(event_values, event_values[1:]):
        dt_sec = max(0.001, (curr[0] - prev[0]) / 1000.0)
        velocities.append((curr[2] - prev[2]) / dt_sec)

    direction = "possible_inward" if peak_dev > 0 else "possible_outward" if peak_dev < 0 else "stable"
    review_required = abs(peak_dev) >= COVER_AMPLITUDE_TODO_PILOT
    return {
        "status": "COMPLETED",
        "result": "REVIEW_REQUIRED" if review_required else "MEASUREMENT_ONLY",
        "reasonCodes": reason_codes + (["AMPLITUDE_ABOVE_TODO_PILOT"] if review_required else ["THRESHOLDS_TODO_PILOT"]),
        "measurements": {
            "baselineMedianU": round(baseline, 5),
            "peakAmplitudeU": round(float(peak_dev), 5),
            "peakAbsoluteAmplitudeU": round(abs(float(peak_dev)), 5),
            "direction": direction,
            "peakPhase": peak_phase,
            "latencyMs": round(float(peak_ts - event_values[0][0]), 3),
            "peakVelocityUPerSec": round(max((abs(v) for v in velocities), default=0.0), 5),
            "maxTimestampGapMs": round(max_gap, 3),
            "gapBreakCount": len(gap_breaks),
            "threshold": {
                "amplitudeU": COVER_AMPLITUDE_TODO_PILOT,
                "source": CONFIG_VERSION,
            },
        },
        "quality": {
            "validFrameRatio": round(valid_ratio, 4),
            "trackingConfidence": round(tracking_confidence, 4),
            "blinkRatio": round(blink_count / len(samples), 4),
            "occlusionRatio": round(occlusion_count / len(samples), 4),
        },
    }


def measure_research_request(req: ResearchMeasurementRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    request_id = _safe_request_id(req.requestId)
    common_reasons = _validate_common(req)

    if req.testType == "HIRSCHBERG":
        result = measure_hirschberg(req)
    elif req.testType == "COVER":
        result = measure_cover(req)
    else:
        raise ResearchMeasurementError("INVALID_REQUEST", "Unsupported testType.")

    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
    response = {
        "requestId": request_id,
        "sessionId": req.sessionId,
        "testType": req.testType,
        "status": result["status"],
        "result": result["result"],
        "reasonCodes": common_reasons + result.get("reasonCodes", []),
        "measurements": result.get("measurements", {}),
        "quality": result.get("quality", {}),
        "versions": {
            "schemaVersion": req.schemaVersion,
            "featureVersion": FEATURE_VERSION,
            "modelVersion": None,
            "preprocessingVersion": PREPROCESSING_VERSION,
            "configVersion": CONFIG_VERSION,
            "thresholdSource": CONFIG_VERSION,
        },
        "experimental": True,
        "latencyMs": latency_ms,
    }
    if "aiPrediction" in result:
        response["aiPrediction"] = result["aiPrediction"]
    logger.info(
        "[ResearchMeasurement] requestId=%s sessionId=%s testType=%s status=%s result=%s latencyMs=%.3f quality=%s",
        request_id,
        req.sessionId,
        req.testType,
        response["status"],
        response["result"],
        latency_ms,
        response["quality"],
    )
    return response





