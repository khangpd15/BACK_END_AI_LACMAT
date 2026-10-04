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
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    mp = None

from app.schemas.research_measurement import ResearchMeasurementRequest

logger = logging.getLogger("remicare.research_measurement")

SUPPORTED_SCHEMA_VERSION = "remicare-research-quality-v0.1"
FEATURE_VERSION = "research-geometry-v0.1"
PREPROCESSING_VERSION = "research-geometry-preprocess-v0.1"
CONFIG_VERSION = "TODO_PILOT"

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 2560 * 1440
MAX_TIMESTAMP_GAP_MS = 600.0
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

# Compact-glint reflex detector, synced with the manual/oracle research audit.
REFLEX_MIN_PIXELS_TODO_PILOT = 1
REFLEX_MAX_PIXELS_TODO_PILOT = 60
REFLEX_BROAD_SEARCH_RADIUS_FACTOR_TODO_PILOT = 1.60   # Relative to iris radius
REFLEX_STRICT_SEARCH_RADIUS_FACTOR_TODO_PILOT = 1.30  # Reject accepted candidate beyond this radius
REFLEX_MIN_PEAK_LUMA_TODO_PILOT = 180.0
REFLEX_THRESHOLD_DELTA_TODO_PILOT = 32.0
REFLEX_MAX_ASPECT_RATIO_TODO_PILOT = 2.8              # Exclude elongated glare streaks
REFLEX_CLUSTER_DISTANCE_PX_TODO_PILOT = 8.0
REFLEX_CLUSTER_SCORE_GAP_TODO_PILOT = 3.0

REFLEX_SUCCESS_STATUS = "DETECTED"
REFLEX_FAILURE_DETAIL_STATUSES = {
    "REFLEX_NOT_FOUND",
    "LOW_PEAK_BRIGHTNESS",
    "LARGE_OR_ELONGATED_GLARE",
    "CLUSTERED_REFLEX_CANDIDATES",
    "REFLEX_CANDIDATE_TOO_FAR",
    "LOW_QUALITY_INPUT",
    "MULTIPLE_REFLEX",
}

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


@dataclass
class EyeHirschbergMeasurement:
    """Standardized geometric Hirschberg measurement for an individual eye."""
    eye: str  # "left" (OS) or "right" (OD)
    pupil_center_x_px: float
    pupil_center_y_px: float
    iris_diameter_px: float
    clr_found: bool
    clr_x_px: Optional[float]
    clr_y_px: Optional[float]
    dx_px: Optional[float]  # clr_x - pupil_center_x
    dy_px: Optional[float]  # clr_y - pupil_center_y
    dx_mm: Optional[float]  # normalized by corneal diameter (~11.7mm)
    dy_mm: Optional[float]  # normalized by corneal diameter (~11.7mm)
    nasal_scleral_area: float
    temporal_scleral_area: float
    nasal_to_temporal_scleral_ratio: float
    status: str
    displacement_anatomical: Optional[str] = None


@dataclass
class QualityGateMetrics:
    """Multi-factor clinical quality gatekeeper metrics."""
    is_acceptable: bool
    blur_variance: float
    is_blurry: bool
    head_pose_pitch: float
    head_pose_yaw: float
    head_pose_roll: float
    head_pose_status: str  # "OPTIMAL", "MILD_TILT", "EXCEEDED_TOLERANCE"
    inter_pupillary_angle_deg: float
    face_detected: bool
    landmarks_count: int
    confidence_score: float
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class ROICrops:
    """Three levels of standardized ocular crops for AI vision backbones."""
    bino_periocular: Optional[np.ndarray] = None  # (H, W, 3) both eyes + bridge of nose
    left_eye_224: Optional[np.ndarray] = None     # (224, 224, 3) Left eye (OS)
    right_eye_224: Optional[np.ndarray] = None    # (224, 224, 3) Right eye (OD)


@dataclass
class ResearchMeasurementOutput:
    """Comprehensive output container for ResearchMeasurementService."""
    is_acceptable: bool
    confidence_score: float
    rejection_reasons: List[str]
    quality: QualityGateMetrics
    hirschberg_left: Optional[EyeHirschbergMeasurement]
    hirschberg_right: Optional[EyeHirschbergMeasurement]
    intercanthal_distance_px: Optional[float]
    intercanthal_distance_mm: Optional[float]
    symmetry_deviation_mm: Optional[float]
    is_likely_pseudostrabismus: Optional[bool]
    pseudostrabismus_notes: str
    rois: ROICrops
    leveled_image: Optional[np.ndarray] = None


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
    """Detect compact corneal light reflexes within an iris-centered ROI.

    Returns:
        (candidates, reflex_status, reflex_tier)
        reflex_status values are frontend-facing research reason codes, including:
        DETECTED, LOW_PEAK_BRIGHTNESS, LARGE_OR_ELONGATED_GLARE,
        CLUSTERED_REFLEX_CANDIDATES, REFLEX_CANDIDATE_TOO_FAR,
        REFLEX_NOT_FOUND, LOW_QUALITY_INPUT.
    """
    if image.ndim != 3 or image.shape[2] < 3 or iris_diameter <= 1.0:
        return [], "LOW_QUALITY_INPUT", None

    height, width = image.shape[:2]
    iris_radius = iris_diameter / 2.0
    broad_radius = max(16.0, iris_radius * REFLEX_BROAD_SEARCH_RADIUS_FACTOR_TODO_PILOT)
    strict_radius = max(12.0, iris_radius * REFLEX_STRICT_SEARCH_RADIUS_FACTOR_TODO_PILOT)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(broad_radius)), 255, -1)
    masked_values = gray[mask > 0]
    if masked_values.size == 0:
        return [], "LOW_QUALITY_INPUT", None

    max_val = float(np.max(masked_values))
    if max_val < REFLEX_MIN_PEAK_LUMA_TODO_PILOT:
        return [], "LOW_PEAK_BRIGHTNESS", None

    threshold = max(
        int(REFLEX_MIN_PEAK_LUMA_TODO_PILOT),
        int(max_val - REFLEX_THRESHOLD_DELTA_TODO_PILOT),
    )
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask)
    num_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates: List[Dict[str, Any]] = []
    rejected_large_or_long = 0
    rejected_too_far = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        aspect = max(box_w, box_h) / max(1, min(box_w, box_h))
        x = float(centroids[label_idx][0])
        y = float(centroids[label_idx][1])
        dist = float(math.hypot(x - cx, y - cy))

        if area > REFLEX_MAX_PIXELS_TODO_PILOT or (
            aspect > REFLEX_MAX_ASPECT_RATIO_TODO_PILOT and area >= 4
        ):
            rejected_large_or_long += 1
            continue
        if area < REFLEX_MIN_PIXELS_TODO_PILOT:
            continue
        if dist > strict_radius:
            rejected_too_far += 1
            continue

        component_mask = component_labels == label_idx
        peak = float(np.max(gray[component_mask])) if np.any(component_mask) else max_val
        score = (
            dist
            + area * 0.12
            + max(0.0, aspect - 1.0) * 1.5
            - (peak - threshold) * 0.03
        )
        candidates.append({
            "x": x,
            "y": y,
            "pixel_count": area,
            "area": area,
            "aspect_ratio": round(float(aspect), 4),
            "distance_from_iris_center_px": round(dist, 4),
            "score": round(float(score), 4),
            "tier": "compact_specular",
            "threshold_source": "TODO_PILOT",
        })

    if not candidates:
        if rejected_large_or_long:
            return [], "LARGE_OR_ELONGATED_GLARE", None
        if rejected_too_far:
            return [], "REFLEX_CANDIDATE_TOO_FAR", None
        return [], "REFLEX_NOT_FOUND", None

    candidates = sorted(candidates, key=lambda item: float(item["score"]))
    best = candidates[0]
    clustered = [
        candidate for candidate in candidates[1:]
        if math.hypot(float(candidate["x"]) - float(best["x"]), float(candidate["y"]) - float(best["y"]))
        <= REFLEX_CLUSTER_DISTANCE_PX_TODO_PILOT
        and float(candidate["score"]) - float(best["score"]) <= REFLEX_CLUSTER_SCORE_GAP_TODO_PILOT
    ]
    if clustered:
        return [], "CLUSTERED_REFLEX_CANDIDATES", None

    best["secondary_candidate_count"] = len(candidates) - 1
    tier = "detected_with_secondary_candidates" if len(candidates) > 1 else "compact_specular"
    return [best], REFLEX_SUCCESS_STATUS, tier


# Backwards compatibility alias
def _detect_reflexes(image: np.ndarray, cx: float, cy: float, radius: float) -> List[Tuple[float, float, int]]:
    candidates, _status, _tier = detect_reflexes_in_roi(image, cx, cy, radius * 2.0)
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

    if reflex_status == REFLEX_SUCCESS_STATUS and reflex_count == 1:
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
    if reflex_status == REFLEX_SUCCESS_STATUS and pupil_status == "DETECTED":
        status = "MEASURED"
    elif reflex_status in REFLEX_FAILURE_DETAIL_STATUSES:
        status = reflex_status
    elif pupil_status == "PUPIL_NOT_FOUND":
        status = "PUPIL_NOT_FOUND"
    elif pupil_status == "LOW_QUALITY_INPUT":
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
        reflex_detected = (
            right.reflex_status == REFLEX_SUCCESS_STATUS
            and left.reflex_status == REFLEX_SUCCESS_STATUS
            and right.reflex_count == 1
            and left.reflex_count == 1
        )
        measured = right.h_iris is not None and left.h_iris is not None

        reasons: List[str] = []
        if not reflex_detected:
            for reflex_status in (right.reflex_status, left.reflex_status):
                if reflex_status == REFLEX_SUCCESS_STATUS:
                    continue
                # Preserve the broad legacy reason so older clients can keep
                # their existing fallback copy, then add the detailed reason.
                if "REFLEX_NOT_FOUND" not in reasons:
                    reasons.append("REFLEX_NOT_FOUND")
                if reflex_status and reflex_status not in reasons:
                    reasons.append(reflex_status)
            if len(reasons) == 1 and "REFLEX_COUNT_NOT_EXACTLY_ONE_PER_EYE" not in reasons:
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


# ==============================================================================
# ResearchMeasurementService: Advanced Gatekeeper, Hirschberg & Pseudostrabismus
# ==============================================================================

# Standard 3D facial model for cv2.solvePnP (in mm, reference origin at nose tip)
FACIAL_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),     # Chin (landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye outer corner (landmark 263)
    (225.0, 170.0, -135.0),   # Right eye outer corner (landmark 33)
    (-150.0, -150.0, -125.0), # Left mouth corner (landmark 287)
    (150.0, -150.0, -125.0),  # Right mouth corner (landmark 57)
], dtype=np.float64)

class ResearchMeasurementService:
    """Clinical & Biomedical Ocular Measurement Service for RemiCare Strabismus AI.
    
    Modules:
    1. Quality Gatekeeper:
       - 468+10 MediaPipe Face Mesh & Iris landmark detection.
       - Laplacian blur variance verification (threshold >= 100.0).
       - 3D Head Pose estimation (Pitch, Yaw, Roll via solvePnP). Rejects if Pitch/Yaw > 10°, Roll > 5°.
       - Inter-pupillary line leveling via 2D Affine transformation.
    2. Hirschberg Geometric Features:
       - Pupil center extraction from refined iris landmarks & aperture analysis.
       - Corneal Light Reflex (CLR) Hunter algorithm via adaptive specular highlight detection.
       - Hirschberg decentration in pixels; mm requires an explicitly measured corneal diameter.
       - Nasal-to-Temporal Scleral Area Ratio for epicanthal fold / flat nasal bridge detection.
    3. AI ROI Extraction:
       - (a) Bino-periocular crop (both eyes + bridge of nose).
       - (b) Left eye crop (224x224).
       - (c) Right eye crop (224x224).
    4. Pseudostrabismus Safety Check:
       - Identifies pseudostrabismus when Hirschberg is orthophoric (|dx| < 0.40mm) despite narrow nasal sclera.
    """

    def __init__(
        self,
        min_blur_var: float = 100.0,
        max_pitch_deg: float = 10.0,
        max_yaw_deg: float = 10.0,
        max_roll_deg: float = 5.0,
        cornea_diameter_mm: Optional[float] = None,
    ):
        self.min_blur_var = min_blur_var
        self.max_pitch_deg = max_pitch_deg
        self.max_yaw_deg = max_yaw_deg
        self.max_roll_deg = max_roll_deg
        if cornea_diameter_mm is not None and cornea_diameter_mm <= 0:
            raise ValueError("cornea_diameter_mm must be a positive measured value")
        self.cornea_diameter_mm = cornea_diameter_mm

        self._face_mesh = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                logger.warning("Could not initialize MediaPipe FaceMesh: %s", exc)

    def compute_blur_variance(self, image: np.ndarray) -> float:
        """Calculates image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if (image.ndim == 3 and image.shape[2] >= 3) else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def estimate_head_pose(
        self,
        landmarks: List[Dict[str, float]],
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float, float, str, List[str]]:
        """Estimates 3D Head Pose (Pitch, Yaw, Roll in degrees) using cv2.solvePnP."""
        h, w = image_shape[:2]
        key_indices = [1, 152, 263, 33, 287, 57]
        reasons: List[str] = []

        if len(landmarks) < 468 or any(idx >= len(landmarks) for idx in key_indices):
            return 0.0, 0.0, 0.0, "UNKNOWN", ["LANDMARKS_INSUFFICIENT_FOR_HEAD_POSE"]

        image_points = []
        for idx in key_indices:
            pt = landmarks[idx]
            image_points.append([pt["x"] * w, pt["y"] * h])
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _tvec = cv2.solvePnP(
            FACIAL_MODEL_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0, "ESTIMATION_FAILED", ["HEAD_POSE_ESTIMATION_FAILED"]

        R, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

        is_exceeded = False
        if abs(pitch) > self.max_pitch_deg:
            reasons.append(f"HEAD_PITCH_EXCEEDED: |pitch|={abs(pitch):.1f}° > {self.max_pitch_deg}°")
            is_exceeded = True
        if abs(yaw) > self.max_yaw_deg:
            reasons.append(f"HEAD_YAW_EXCEEDED: |yaw|={abs(yaw):.1f}° > {self.max_yaw_deg}°")
            is_exceeded = True
        if abs(roll) > self.max_roll_deg:
            reasons.append(f"HEAD_ROLL_EXCEEDED: |roll|={abs(roll):.1f}° > {self.max_roll_deg}°")
            is_exceeded = True

        if is_exceeded:
            status = "EXCEEDED_TOLERANCE"
        elif abs(pitch) > (self.max_pitch_deg * 0.6) or abs(yaw) > (self.max_yaw_deg * 0.6):
            status = "MILD_TILT"
        else:
            status = "OPTIMAL"

        return pitch, yaw, roll, status, reasons

    def level_interpupillary_line(
        self,
        image: np.ndarray,
        right_pupil_px: Tuple[float, float],
        left_pupil_px: Tuple[float, float],
        landmarks: List[Dict[str, float]],
    ) -> Tuple[np.ndarray, List[Dict[str, float]], float, np.ndarray]:
        """Performs 2D Affine rotation so that the inter-pupillary line is horizontal."""
        h, w = image.shape[:2]
        xr, yr = right_pupil_px
        xl, yl = left_pupil_px

        dx = xl - xr
        dy = yl - yr
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        mid_x = (xr + xl) / 2.0
        mid_y = (yr + yl) / 2.0

        M = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
        leveled_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

        leveled_landmarks: List[Dict[str, float]] = []
        for lm in landmarks:
            px = lm["x"] * w
            py = lm["y"] * h
            rot_x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            rot_y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            leveled_landmarks.append({
                "x": float(rot_x / w),
                "y": float(rot_y / h),
                "z": lm.get("z", 0.0),
            })

        return leveled_img, leveled_landmarks, angle_deg, M

    def hunt_corneal_light_reflex(
        self,
        image: np.ndarray,
        iris_cx: float,
        iris_cy: float,
        iris_diameter: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Corneal Light Reflex (CLR) Hunter:
        Identifies peak specular glint within iris radius using Luma adaptive thresholding.
        """
        candidates, status, _tier = detect_reflexes_in_roi(image, iris_cx, iris_cy, iris_diameter)
        if candidates and status == REFLEX_SUCCESS_STATUS:
            return float(candidates[0]["x"]), float(candidates[0]["y"]), "DETECTED"
        return None, None, status

    def compute_scleral_area_ratio(
        self,
        landmarks: List[Dict[str, float]],
        eye: str,
        iris_cx: float,
        iris_cy: float,
        iris_diameter: float,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """Computes Nasal-to-Temporal Scleral Area Ratio for the specified eye."""
        h, w = image_shape[:2]
        r = iris_diameter / 2.0

        if eye == "left":
            nasal_idx = LEFT_CORNERS["nasal"]      # 362 (inner, toward nose)
            temporal_idx = LEFT_CORNERS["temporal"] # 263 (outer, toward ear)
            upper_lid_idx = 386
            lower_lid_idx = 374

            nasal_x = landmarks[nasal_idx]["x"] * w
            temporal_x = landmarks[temporal_idx]["x"] * w
            upper_y = landmarks[upper_lid_idx]["y"] * h
            lower_y = landmarks[lower_lid_idx]["y"] * h
            fissure_height = max(1.0, abs(lower_y - upper_y))

            medial_edge = iris_cx - r
            lateral_edge = iris_cx + r
            nasal_width = max(0.5, medial_edge - nasal_x)
            temporal_width = max(0.5, temporal_x - lateral_edge)

        else:
            nasal_idx = RIGHT_CORNERS["nasal"]      # 133 (inner, toward nose)
            temporal_idx = RIGHT_CORNERS["temporal"] # 33 (outer, toward ear)
            upper_lid_idx = 159
            lower_lid_idx = 145

            nasal_x = landmarks[nasal_idx]["x"] * w
            temporal_x = landmarks[temporal_idx]["x"] * w
            upper_y = landmarks[upper_lid_idx]["y"] * h
            lower_y = landmarks[lower_lid_idx]["y"] * h
            fissure_height = max(1.0, abs(lower_y - upper_y))

            medial_edge = iris_cx + r
            lateral_edge = iris_cx - r
            nasal_width = max(0.5, nasal_x - medial_edge)
            temporal_width = max(0.5, lateral_edge - temporal_x)

        area_nasal = float(0.5 * nasal_width * fissure_height)
        area_temporal = float(0.5 * temporal_width * fissure_height)
        ratio = round(float(area_nasal / max(0.1, area_temporal)), 4)
        return round(area_nasal, 2), round(area_temporal, 2), ratio

    def extract_rois(
        self,
        image_leveled: np.ndarray,
        left_eye_center: Tuple[float, float],
        right_eye_center: Tuple[float, float],
        left_corners: Tuple[float, float],
        right_corners: Tuple[float, float],
        avg_iris_diameter: float,
    ) -> ROICrops:
        """Extracts 3 ROI crop levels: Bino-periocular, Left eye 224x224, Right eye 224x224."""
        h, w = image_leveled.shape[:2]
        xr, yr = right_eye_center
        xl, yl = left_eye_center
        ipd = max(10.0, math.hypot(xl - xr, yl - yr))

        # (a) Bino-periocular crop
        temp_r = right_corners[1]
        temp_l = left_corners[1]
        x_min_span = min(temp_r, xr - avg_iris_diameter * 1.5)
        x_max_span = max(temp_l, xl + avg_iris_diameter * 1.5)
        margin_x = 0.15 * ipd
        x0_bino = max(0, int(round(x_min_span - margin_x)))
        x1_bino = min(w, int(round(x_max_span + margin_x)))

        mid_y = (yr + yl) / 2.0
        y0_bino = max(0, int(round(mid_y - 0.40 * ipd)))
        y1_bino = min(h, int(round(mid_y + 0.35 * ipd)))

        bino_crop = image_leveled[y0_bino:y1_bino, x0_bino:x1_bino] if (x1_bino > x0_bino and y1_bino > y0_bino) else None

        # (b) Left eye 224x224
        half_crop_l = max(16.0, avg_iris_diameter * 1.6)
        x0_l = max(0, int(round(xl - half_crop_l)))
        x1_l = min(w, int(round(xl + half_crop_l)))
        y0_l = max(0, int(round(yl - half_crop_l)))
        y1_l = min(h, int(round(yl + half_crop_l)))
        left_crop_raw = image_leveled[y0_l:y1_l, x0_l:x1_l]
        left_224 = cv2.resize(left_crop_raw, (224, 224), interpolation=cv2.INTER_LANCZOS4) if left_crop_raw.size > 0 else None

        # (c) Right eye 224x224
        half_crop_r = max(16.0, avg_iris_diameter * 1.6)
        x0_r = max(0, int(round(xr - half_crop_r)))
        x1_r = min(w, int(round(xr + half_crop_r)))
        y0_r = max(0, int(round(yr - half_crop_r)))
        y1_r = min(h, int(round(yr + half_crop_r)))
        right_crop_raw = image_leveled[y0_r:y1_r, x0_r:x1_r]
        right_224 = cv2.resize(right_crop_raw, (224, 224), interpolation=cv2.INTER_LANCZOS4) if right_crop_raw.size > 0 else None

        return ROICrops(
            bino_periocular=bino_crop,
            left_eye_224=left_224,
            right_eye_224=right_224,
        )

    def check_pseudostrabismus_rule(
        self,
        left_meas: Optional[EyeHirschbergMeasurement],
        right_meas: Optional[EyeHirschbergMeasurement],
    ) -> Tuple[Optional[bool], str]:
        """Evaluates clinical rule check to distinguish Pseudostrabismus from Esotropia."""
        if not left_meas or not right_meas or left_meas.dx_mm is None or right_meas.dx_mm is None:
            return None, "MEASUREMENT_INCOMPLETE_CANNOT_EVALUATE_RULE"

        dx_l = left_meas.dx_mm
        dx_r = right_meas.dx_mm
        ratio_l = left_meas.nasal_to_temporal_scleral_ratio
        ratio_r = right_meas.nasal_to_temporal_scleral_ratio
        symmetry_dev = abs(dx_l - dx_r)

        # Normal orthophoric decentration bound: |dx| < 0.40 mm
        is_orthophoric = (abs(dx_l) < 0.40) and (abs(dx_r) < 0.40) and (symmetry_dev < 0.50)
        # Narrow nasal scleral margin (epicanthal fold / flat nasal bridge)
        has_narrow_nasal_sclera = (ratio_l < 0.70) or (ratio_r < 0.70)

        if is_orthophoric and has_narrow_nasal_sclera:
            return True, (
                "Bilateral corneal light reflexes are orthophoric (|dx| < 0.40mm) despite narrow nasal sclera "
                f"(scleral ratios: OS={ratio_l:.2f}, OD={ratio_r:.2f}), confirming characteristic Pseudostrabismus."
            )
        elif (abs(dx_l) >= 0.40) or (abs(dx_r) >= 0.40) or (symmetry_dev >= 0.50):
            return False, (
                "Corneal light reflex shows significant decentration (|dx| >= 0.40mm or asymmetry >= 0.50mm); "
                "points to genuine strabismus (Esotropia / Exotropia), not Pseudostrabismus."
            )
        else:
            return False, "Orthophoric bilateral corneal reflexes with normal scleral exposure."

    def process_image(
        self,
        image_bgr_or_rgb: np.ndarray,
        landmarks_provided: Optional[List[Dict[str, float]]] = None,
    ) -> ResearchMeasurementOutput:
        """Runs end-to-end Quality Gate, Affine Leveling, Hirschberg CLR Hunter, Scleral Ratios, ROIs & Safety Check."""
        img = image_bgr_or_rgb
        if img.ndim != 3:
            raise ValueError("Input image must be a 3-channel image (H, W, 3).")
        h, w = img.shape[:2]

        rejection_reasons: List[str] = []

        # 1. Blur check
        blur_var = self.compute_blur_variance(img)
        is_blurry = blur_var < self.min_blur_var
        if is_blurry:
            rejection_reasons.append(f"IMAGE_BLURRY: var={blur_var:.1f} < {self.min_blur_var}")

        # 2. Extract landmarks
        landmarks = landmarks_provided
        if landmarks is None and self._face_mesh is not None:
            rgb_for_mp = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            results = self._face_mesh.process(rgb_for_mp)
            if results.multi_face_landmarks:
                landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in results.multi_face_landmarks[0].landmark
                ]

        face_detected = landmarks is not None and len(landmarks) >= 468
        if not face_detected:
            rejection_reasons.append("FACE_OR_LANDMARKS_NOT_DETECTED")
            quality_metrics = QualityGateMetrics(
                is_acceptable=False,
                blur_variance=round(blur_var, 1),
                is_blurry=is_blurry,
                head_pose_pitch=0.0,
                head_pose_yaw=0.0,
                head_pose_roll=0.0,
                head_pose_status="REJECTED_EXCEEDED_TOLERANCE",
                inter_pupillary_angle_deg=0.0,
                face_detected=False,
                landmarks_count=len(landmarks) if landmarks else 0,
                confidence_score=0.1,
                rejection_reasons=rejection_reasons,
            )
            return ResearchMeasurementOutput(
                is_acceptable=False,
                confidence_score=0.1,
                rejection_reasons=rejection_reasons,
                quality=quality_metrics,
                hirschberg_left=None,
                hirschberg_right=None,
                intercanthal_distance_px=None,
                intercanthal_distance_mm=None,
                symmetry_deviation_mm=None,
                is_likely_pseudostrabismus=False,
                pseudostrabismus_notes="Face detection failed.",
                rois=ROICrops(),
            )

        # 3. Head Pose Estimation
        pitch, yaw, roll, pose_status, pose_reasons = self.estimate_head_pose(landmarks, (h, w))
        rejection_reasons.extend(pose_reasons)

        # 4. Extract Iris centers for leveling
        left_geom = _iris_geometry(landmarks, LEFT_IRIS, w, h)
        right_geom = _iris_geometry(landmarks, RIGHT_IRIS, w, h)

        if not left_geom or not right_geom:
            rejection_reasons.append("IRIS_LANDMARKS_INVALID")
            quality_metrics = QualityGateMetrics(
                is_acceptable=False,
                blur_variance=round(blur_var, 1),
                is_blurry=is_blurry,
                head_pose_pitch=round(pitch, 2),
                head_pose_yaw=round(yaw, 2),
                head_pose_roll=round(roll, 2),
                head_pose_status=pose_status,
                inter_pupillary_angle_deg=0.0,
                face_detected=True,
                landmarks_count=len(landmarks),
                confidence_score=0.3,
                rejection_reasons=rejection_reasons,
            )
            return ResearchMeasurementOutput(
                is_acceptable=False,
                confidence_score=0.3,
                rejection_reasons=rejection_reasons,
                quality=quality_metrics,
                hirschberg_left=None,
                hirschberg_right=None,
                intercanthal_distance_px=None,
                intercanthal_distance_mm=None,
                symmetry_deviation_mm=None,
                is_likely_pseudostrabismus=False,
                pseudostrabismus_notes="Iris geometry invalid.",
                rois=ROICrops(),
            )

        left_cx, left_cy, left_d = left_geom
        right_cx, right_cy, right_d = right_geom

        # 5. Affine Inter-pupillary Line Leveling
        leveled_img, leveled_landmarks, angle_deg, M = self.level_interpupillary_line(
            img, (right_cx, right_cy), (left_cx, left_cy), landmarks
        )

        lev_right_cx = M[0, 0] * right_cx + M[0, 1] * right_cy + M[0, 2]
        lev_right_cy = M[1, 0] * right_cx + M[1, 1] * right_cy + M[1, 2]
        lev_left_cx = M[0, 0] * left_cx + M[0, 1] * left_cy + M[0, 2]
        lev_left_cy = M[1, 0] * left_cx + M[1, 1] * left_cy + M[1, 2]
        avg_d = (left_d + right_d) / 2.0

        # 6. Hunt CLR on leveled image
        clr_l_x, clr_l_y, clr_l_status = self.hunt_corneal_light_reflex(leveled_img, lev_left_cx, lev_left_cy, left_d)
        clr_r_x, clr_r_y, clr_r_status = self.hunt_corneal_light_reflex(leveled_img, lev_right_cx, lev_right_cy, right_d)

        if clr_l_status != "DETECTED":
            rejection_reasons.append(f"LEFT_EYE_REFLEX_{clr_l_status}")
        if clr_r_status != "DETECTED":
            rejection_reasons.append(f"RIGHT_EYE_REFLEX_{clr_r_status}")

        nasal_a_l, temp_a_l, ratio_l = self.compute_scleral_area_ratio(
            leveled_landmarks, "left", lev_left_cx, lev_left_cy, left_d, (h, w)
        )
        nasal_a_r, temp_a_r, ratio_r = self.compute_scleral_area_ratio(
            leveled_landmarks, "right", lev_right_cx, lev_right_cy, right_d, (h, w)
        )

        dx_l_px = (clr_l_x - lev_left_cx) if clr_l_x is not None else None
        dy_l_px = (clr_l_y - lev_left_cy) if clr_l_y is not None else None
        dx_l_mm = (
            round((dx_l_px / left_d) * self.cornea_diameter_mm, 4)
            if dx_l_px is not None and self.cornea_diameter_mm is not None else None
        )
        dy_l_mm = (
            round((dy_l_px / left_d) * self.cornea_diameter_mm, 4)
            if dy_l_px is not None and self.cornea_diameter_mm is not None else None
        )

        dx_r_px = (clr_r_x - lev_right_cx) if clr_r_x is not None else None
        dy_r_px = (clr_r_y - lev_right_cy) if clr_r_y is not None else None
        dx_r_mm = (
            round((dx_r_px / right_d) * self.cornea_diameter_mm, 4)
            if dx_r_px is not None and self.cornea_diameter_mm is not None else None
        )
        dy_r_mm = (
            round((dy_r_px / right_d) * self.cornea_diameter_mm, 4)
            if dy_r_px is not None and self.cornea_diameter_mm is not None else None
        )

        left_meas = EyeHirschbergMeasurement(
            eye="left",
            pupil_center_x_px=round(lev_left_cx, 2),
            pupil_center_y_px=round(lev_left_cy, 2),
            iris_diameter_px=round(left_d, 2),
            clr_found=clr_l_status == "DETECTED",
            clr_x_px=round(clr_l_x, 2) if clr_l_x else None,
            clr_y_px=round(clr_l_y, 2) if clr_l_y else None,
            dx_px=round(dx_l_px, 2) if dx_l_px else None,
            dy_px=round(dy_l_px, 2) if dy_l_px else None,
            dx_mm=dx_l_mm,
            dy_mm=dy_l_mm,
            nasal_scleral_area=nasal_a_l,
            temporal_scleral_area=temp_a_l,
            nasal_to_temporal_scleral_ratio=ratio_l,
            status=clr_l_status,
        )

        right_meas = EyeHirschbergMeasurement(
            eye="right",
            pupil_center_x_px=round(lev_right_cx, 2),
            pupil_center_y_px=round(lev_right_cy, 2),
            iris_diameter_px=round(right_d, 2),
            clr_found=clr_r_status == "DETECTED",
            clr_x_px=round(clr_r_x, 2) if clr_r_x else None,
            clr_y_px=round(clr_r_y, 2) if clr_r_y else None,
            dx_px=round(dx_r_px, 2) if dx_r_px else None,
            dy_px=round(dy_r_px, 2) if dy_r_px else None,
            dx_mm=dx_r_mm,
            dy_mm=dy_r_mm,
            nasal_scleral_area=nasal_a_r,
            temporal_scleral_area=temp_a_r,
            nasal_to_temporal_scleral_ratio=ratio_r,
            status=clr_r_status,
        )

        inner_l_x = leveled_landmarks[LEFT_CORNERS["nasal"]]["x"] * w
        inner_l_y = leveled_landmarks[LEFT_CORNERS["nasal"]]["y"] * h
        inner_r_x = leveled_landmarks[RIGHT_CORNERS["nasal"]]["x"] * w
        inner_r_y = leveled_landmarks[RIGHT_CORNERS["nasal"]]["y"] * h
        intercanthal_px = float(math.hypot(inner_l_x - inner_r_x, inner_l_y - inner_r_y))
        intercanthal_mm = (
            round((intercanthal_px / avg_d) * self.cornea_diameter_mm, 2)
            if self.cornea_diameter_mm is not None else None
        )

        sym_dev_mm = round(abs(dx_l_mm - dx_r_mm), 4) if (dx_l_mm is not None and dx_r_mm is not None) else None

        outer_l_x = leveled_landmarks[LEFT_CORNERS["temporal"]]["x"] * w
        outer_r_x = leveled_landmarks[RIGHT_CORNERS["temporal"]]["x"] * w
        rois = self.extract_rois(
            leveled_img,
            (lev_left_cx, lev_left_cy),
            (lev_right_cx, lev_right_cy),
            (inner_l_x, outer_l_x),
            (inner_r_x, outer_r_x),
            avg_d,
        )

        is_pseudo, pseudo_notes = self.check_pseudostrabismus_rule(left_meas, right_meas)

        is_acceptable = len(rejection_reasons) == 0
        conf_score = 0.95 if is_acceptable else max(0.1, 0.95 - 0.15 * len(rejection_reasons))

        quality_metrics = QualityGateMetrics(
            is_acceptable=is_acceptable,
            blur_variance=round(blur_var, 1),
            is_blurry=is_blurry,
            head_pose_pitch=round(pitch, 2),
            head_pose_yaw=round(yaw, 2),
            head_pose_roll=round(roll, 2),
            head_pose_status=pose_status,
            inter_pupillary_angle_deg=round(angle_deg, 2),
            face_detected=True,
            landmarks_count=len(landmarks),
            confidence_score=round(conf_score, 2),
            rejection_reasons=rejection_reasons,
        )

        return ResearchMeasurementOutput(
            is_acceptable=is_acceptable,
            confidence_score=round(conf_score, 2),
            rejection_reasons=rejection_reasons,
            quality=quality_metrics,
            hirschberg_left=left_meas,
            hirschberg_right=right_meas,
            intercanthal_distance_px=round(intercanthal_px, 2),
            intercanthal_distance_mm=intercanthal_mm,
            symmetry_deviation_mm=sym_dev_mm,
            is_likely_pseudostrabismus=is_pseudo,
            pseudostrabismus_notes=pseudo_notes,
            rois=rois,
            leveled_image=leveled_img,
        )






