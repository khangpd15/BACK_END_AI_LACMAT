"""Hirschberg AI Inference Service.

Loads the trained Hirschberg exploratory classifier (`hirschberg-candidate-v0.1`)
and the ONNX representation model (`best_model.onnx`).
Performs feature extraction (73 geometry and appearance features) and runs model prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("remicare.services.hirschberg_ai")

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_DIR / "models" / "research" / "hirschberg_candidate_v0.1.joblib"
ONNX_PATH = APP_DIR / "models" / "best_model.onnx"

_BUNDLE: Optional[Dict[str, Any]] = None
_ONNX_SESS: Optional[ort.InferenceSession] = None


def _get_model_and_onnx() -> Tuple[Optional[Dict[str, Any]], Optional[ort.InferenceSession]]:
    global _BUNDLE, _ONNX_SESS
    if _BUNDLE is None and MODEL_PATH.is_file():
        try:
            _BUNDLE = joblib.load(MODEL_PATH)
            logger.info("Loaded Hirschberg candidate model from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load Hirschberg candidate model: %s", exc)

    if _ONNX_SESS is None and ONNX_PATH.is_file():
        try:
            _ONNX_SESS = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
            logger.info("Loaded ONNX bilateral ROI representation model from %s", ONNX_PATH)
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)

    return _BUNDLE, _ONNX_SESS


def estimate_iris_center_in_crop(image: np.ndarray) -> Tuple[float, float, float]:
    """Estimate iris center (cx, cy, diameter) for a 224x224 cropped eye image."""
    h, w = image.shape[:2]
    default_cx, default_cy = w / 2.0, h / 2.0
    default_diameter = min(w, h) * 0.45

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=45,
        param2=24,
        minRadius=25,
        maxRadius=75,
    )

    if circles is not None and len(circles[0]) > 0:
        best_circle = None
        min_dist = float("inf")
        for c in circles[0]:
            dist = np.hypot(c[0] - default_cx, c[1] - default_cy)
            if dist < min_dist:
                min_dist = dist
                best_circle = c

        if best_circle is not None and min_dist < 45.0:
            cx = float(best_circle[0])
            cy = float(best_circle[1])
            diameter = float(best_circle[2] * 2.0)
            return cx, cy, diameter

    return default_cx, default_cy, default_diameter


def extract_features_from_crop(
    bgr: np.ndarray,
    onnx_sess: ort.InferenceSession,
) -> List[float]:
    """Extracts 73 features from an eye crop matching the Phase 6B feature contract."""
    from app.services.research_measurement_service import detect_pupil_in_roi, detect_reflexes_in_roi

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Iris estimation
    cx, cy, diameter = estimate_iris_center_in_crop(rgb)
    iris_cx_n = float(cx / w)
    iris_cy_n = float(cy / h)
    iris_d_n = float(diameter / max(w, h))

    # 2. Reflex detection
    reflexes, r_status, r_tier = detect_reflexes_in_roi(rgb, cx, cy, diameter)
    r_found = 1.0 if len(reflexes) > 0 else 0.0
    if len(reflexes) > 0:
        best_r = min(reflexes, key=lambda c: (c["x"] - cx) ** 2 + (c["y"] - cy) ** 2)
        rcx_n = float(best_r["x"]) / w
        rcy_n = float(best_r["y"]) / h
        r_area_n = float(best_r.get("pixel_count", 1)) / (w * h)
    else:
        rcx_n, rcy_n, r_area_n = 0.5, 0.5, 0.0

    tier_map = {"strict_specular": 3.0, "medium_specular": 2.0, "adaptive_specular": 1.0}
    r_tier_val = tier_map.get(r_tier, 0.0)

    # 3. Pupil detection
    pupil_c, pupil_d, p_status = detect_pupil_in_roi(rgb, cx, cy, diameter)
    p_found = 1.0 if p_status == "DETECTED" and pupil_c is not None else 0.0
    if p_found > 0.5 and pupil_c is not None:
        pcx_n = float(pupil_c["x"]) / w
        pcy_n = float(pupil_c["y"]) / h
        pd_n = float(pupil_d) / max(w, h)
    else:
        pcx_n, pcy_n, pd_n = iris_cx_n, iris_cy_n, iris_d_n * 0.4

    # 4. Hirschberg vectors
    both_detected = 1.0 if (r_found > 0.5 and p_found > 0.5) else 0.0
    h_iris_x = (rcx_n - iris_cx_n) / max(0.01, iris_d_n) if r_found > 0.5 else 0.0
    h_iris_y = (rcy_n - iris_cy_n) / max(0.01, iris_d_n) if r_found > 0.5 else 0.0
    h_pupil_x = (rcx_n - pcx_n) / max(0.01, iris_d_n) if both_detected > 0.5 else 0.0
    h_pupil_y = (rcy_n - pcy_n) / max(0.01, iris_d_n) if both_detected > 0.5 else 0.0

    # 5. Ocular dark centroid
    dark = 255.0 - gray.astype(np.float32)
    tot_dark = float(np.sum(dark))
    if tot_dark > 1e-4:
        yy, xx = np.indices(gray.shape)
        dark_cx = float(np.sum(xx * dark) / (tot_dark * w))
        dark_cy = float(np.sum(yy * dark) / (tot_dark * h))
    else:
        dark_cx, dark_cy = 0.5, 0.5
    dark_cx_offset = dark_cx - 0.5

    # 6. Asymmetry
    col_means = np.mean(gray, axis=0) / 255.0
    left_mean = float(np.mean(col_means[: int(w * 0.4)]))
    right_mean = float(np.mean(col_means[int(w * 0.6) :]))
    lr_ratio = left_mean / max(0.01, right_mean)
    lr_diff = left_mean - right_mean

    # 7. Horizontal profile (16 bins)
    bin_size = max(1, w // 16)
    x_profile = [float(np.mean(col_means[i * bin_size : (i + 1) * bin_size])) for i in range(16)]

    # 8. Spatial grid (4x4 = 16 cells x 2 stats = 32)
    spatial_feats = []
    gh, gw = max(1, h // 4), max(1, w // 4)
    for gi in range(4):
        for gj in range(4):
            cell = gray[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw].astype(np.float32) / 255.0
            spatial_feats.extend([float(np.mean(cell)), float(np.std(cell))])

    # 9. ResNet logits
    rgb_224 = cv2.resize(rgb, (224, 224))
    rgb_norm = rgb_224.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    tensor = ((rgb_norm.transpose(2, 0, 1)[np.newaxis, ...] - mean) / std).astype(np.float32)
    logits = onnx_sess.run(None, {"input_frame": tensor})[0][0]
    l0, l1 = float(logits[0]), float(logits[1])
    exp_l = np.exp([l0, l1] - np.max([l0, l1]))
    prob_strab = float(exp_l[1] / np.sum(exp_l))

    feats = [
        iris_cx_n,
        iris_cy_n,
        iris_d_n,
        r_found,
        rcx_n,
        rcy_n,
        r_area_n,
        r_tier_val,
        p_found,
        pcx_n,
        pcy_n,
        pd_n,
        both_detected,
        h_iris_x,
        h_iris_y,
        h_pupil_x,
        h_pupil_y,
        dark_cx,
        dark_cy,
        dark_cx_offset,
        lr_ratio,
        lr_diff,
        *x_profile,
        *spatial_feats,
        l0,
        l1,
        prob_strab,
    ]
    return feats


def crop_eye_from_landmarks(
    image: np.ndarray,
    landmarks: List[Dict[str, Any]],
    eye: str = "right",
    target_size: int = 224,
) -> np.ndarray:
    """Crops a square eye region centered on iris landmarks."""
    from app.services.research_measurement_service import LEFT_IRIS, RIGHT_IRIS, _iris_geometry

    h, w = image.shape[:2]
    indices = LEFT_IRIS if eye == "left" else RIGHT_IRIS
    geom = _iris_geometry(landmarks, indices, w, h)
    if not geom:
        # Fallback to half-face crop
        if eye == "right":
            return cv2.resize(image[:, : w // 2], (target_size, target_size))
        return cv2.resize(image[:, w // 2 :], (target_size, target_size))

    cx, cy, diameter = geom
    crop_size = max(target_size, int(diameter * 2.8))
    half = crop_size // 2

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(image, (target_size, target_size))
    return cv2.resize(crop, (target_size, target_size))


def predict_hirschberg(
    image_bgr: np.ndarray,
    landmarks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Runs Hirschberg AI model inference on an input image.

    Returns:
        Dict with status, predictedClass, confidence, probabilities, and model details.
    """
    bundle, onnx_sess = _get_model_and_onnx()
    if not bundle or not onnx_sess:
        return {
            "status": "UNAVAILABLE",
            "message": "Hirschberg AI model artifact not available on backend.",
        }

    pipeline = bundle.get("pipeline")
    classes = bundle.get("classes", ["esotropia", "exotropia", "normal"])

    # Prepare eye crops
    crops = []
    if landmarks and len(landmarks) >= 468:
        od_crop = crop_eye_from_landmarks(image_bgr, landmarks, "right")
        os_crop = crop_eye_from_landmarks(image_bgr, landmarks, "left")
        crops = [od_crop, os_crop]
    else:
        # Check if the image itself is approximately square / eye crop (e.g. 224x224)
        h, w = image_bgr.shape[:2]
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < 1.35 and min(w, h) <= 400:
            crops = [cv2.resize(image_bgr, (224, 224))]
        else:
            # Full face without landmarks: extract left and right eye regions by heuristics
            left_half = image_bgr[:, : w // 2]
            right_half = image_bgr[:, w // 2 :]
            crops = [
                cv2.resize(left_half, (224, 224)),
                cv2.resize(right_half, (224, 224)),
            ]

    # Predict on each crop and aggregate probabilities
    all_probs = []
    for crop in crops:
        feats = extract_features_from_crop(crop, onnx_sess)
        X = np.array(feats, dtype=np.float32).reshape(1, -1)
        prob = pipeline.predict_proba(X)[0]
        all_probs.append(prob)

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    pred_class = classes[pred_idx]
    confidence = float(avg_probs[pred_idx])

    prob_dict = {cls_name: round(float(avg_probs[i]), 4) for i, cls_name in enumerate(classes)}

    return {
        "status": "PREDICTED",
        "predictedClass": pred_class.upper(),
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "modelId": bundle.get("model_id", "hirschberg-candidate-v0.1"),
        "datasetVersion": bundle.get("dataset_version", "hirschberg-folder-labels-v0.1"),
        "evaluationSummary": "Balanced Accuracy 76%, Sensitivity 91%, Specificity 76%, ROC-AUC 0.912",
        "nonClinicalDeclaration": "Sàng lọc nghiên cứu - Không thay thế chẩn đoán bác sĩ chuyên khoa.",
    }
