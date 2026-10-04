"""Hirschberg AI Inference Service.

Loads the trained Hirschberg exploratory classifier (`hirschberg-candidate-v0.5-safe`)
and the ONNX representation model (`best_model.onnx`).
Performs feature extraction and runs model prediction with a conservative
research-only abstention policy.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import onnxruntime as ort

from app.services.onnx_runtime import get_cpu_session

logger = logging.getLogger("remicare.services.hirschberg_ai")

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = APP_DIR / "models" / "research" / "hirschberg_candidate_v0.6_ensemble_v2.joblib"
MODEL_PATH = Path(os.getenv("HIRSCHBERG_RESEARCH_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
ONNX_PATH = APP_DIR / "models" / "best_model.onnx"

_BUNDLE: Optional[Dict[str, Any]] = None
_ONNX_SESS: Optional[ort.InferenceSession] = None
_EFFNET: Any = None


def _unavailable_prediction(message: str, reason: str = "RUNTIME_ERROR") -> Dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "reason": reason,
        "message": message,
    }


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
            _ONNX_SESS = get_cpu_session(ONNX_PATH)
            logger.info("Loaded ONNX bilateral ROI representation model from %s", ONNX_PATH)
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)

    return _BUNDLE, _ONNX_SESS


class EfficientNetFeatureExtractor:
    def __init__(self):
        try:
            import timm
            import torch
            import torchvision.transforms as T
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "EfficientNet dependencies (timm, torch, torchvision, Pillow) are unavailable"
            ) from exc

        self._torch = torch
        self._Image = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        except Exception as exc:
            raise RuntimeError(
                "EfficientNet pretrained weights are unavailable; refusing random initialization"
            ) from exc
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(self._Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self.model(tensor).squeeze(0).cpu().numpy()
        bs = len(emb) // 16
        return np.array([float(np.mean(emb[i * bs : (i + 1) * bs])) for i in range(16)], dtype=np.float32)


def _get_effnet() -> EfficientNetFeatureExtractor:
    global _EFFNET
    if _EFFNET is None:
        _EFFNET = EfficientNetFeatureExtractor()
    return _EFFNET


@lru_cache(maxsize=1)
def _secondary_branch_error() -> Optional[str]:
    try:
        _get_effnet()
        return None
    except Exception as exc:
        return str(exc)


def hirschberg_runtime_status() -> Dict[str, Any]:
    return {
        "loaded": _BUNDLE is not None and _ONNX_SESS is not None,
        "configuredModelId": (_BUNDLE or {}).get("model_id"),
        "researchOnly": True,
    }


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


def _pair_feature_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, left - right, np.abs(left - right), (left + right) / 2.0]).astype(np.float32)


def _features_for_bundle(crops: List[np.ndarray], onnx_sess: ort.InferenceSession, bundle: Dict[str, Any]) -> np.ndarray:
    crop_features = [np.array(extract_features_from_crop(crop, onnx_sess), dtype=np.float32) for crop in crops]
    feature_contract = bundle.get("feature_contract")
    expected_count = len(bundle.get("feature_names") or [])

    if feature_contract == "hirschberg_pair_features_v0.5" or expected_count == 365:
        if len(crop_features) == 1:
            left = right = crop_features[0]
        else:
            left, right = crop_features[0], crop_features[1]
        return _pair_feature_vector(left, right).reshape(1, -1)

    feature_vector = np.mean(np.array(crop_features, dtype=np.float32), axis=0)
    return feature_vector.reshape(1, -1)


def predict_hirschberg(
    image_bgr: np.ndarray,
    landmarks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Runs Hirschberg AI model inference on an input image.

    Returns:
        Dict with status, predictedClass, confidence, probabilities, and model details.
    """
    try:
        bundle, onnx_sess = _get_model_and_onnx()
        if not bundle or not onnx_sess:
            return _unavailable_prediction(
                "Hirschberg AI model artifact not available on backend.",
                "MODEL_ARTIFACT_UNAVAILABLE",
            )

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

        model_type = bundle.get("model_type")
        version = bundle.get("version")
        is_ensemble_v2 = (version == "v0.6_ensemble_v2" or model_type == "ensemble_eye_crop_prior_normalized")
        fallback_reason: Optional[str] = None

        if is_ensemble_v2:
            # 1. Primary pipeline (v0.5 ONNX pair features)
            prim_pipeline = bundle.get("primary_pipeline")
            if prim_pipeline is None:
                v5_p = APP_DIR / "models" / "research" / "hirschberg_candidate_v0.5_safe.joblib"
                if v5_p.is_file():
                    prim_pipeline = joblib.load(v5_p).get("pipeline")
                else:
                    prim_pipeline = pipeline

            if prim_pipeline is None:
                return _unavailable_prediction(
                    "Hirschberg AI primary pipeline is not available on backend.",
                    "MODEL_PIPELINE_UNAVAILABLE",
                )

            X5 = _features_for_bundle(
                crops, onnx_sess, {"feature_contract": "hirschberg_pair_features_v0.5", "feature_names": [None] * 365}
            )
            p5 = prim_pipeline.predict_proba(X5)

            try:
                secondary_error = _secondary_branch_error()
                if secondary_error:
                    raise RuntimeError(secondary_error)
                from app.services.eye_crop_geometry_service import (
                    GEOMETRIC_FEATURE_NAMES,
                    extract_eye_crop_geometric_features,
                )

                # 2. Secondary pipeline (v0.2 Eye-Crop Geometry + EfficientNet)
                sec_pipeline = bundle.get("branch_v02_pipeline")
                if sec_pipeline is None:
                    v2_p = APP_DIR / "models" / "research" / "hirschberg_branch_v02_eyecrop.joblib"
                    if v2_p.is_file():
                        sec_pipeline = joblib.load(v2_p).get("pipeline")

                if sec_pipeline is None:
                    raise RuntimeError("secondary eye-crop pipeline is not available")

                gdict = extract_eye_crop_geometric_features(image_bgr)
                missing_features = [name for name in GEOMETRIC_FEATURE_NAMES if gdict[name] is None]
                if missing_features:
                    raise RuntimeError(
                        "secondary branch measurements unavailable: " + ", ".join(missing_features)
                    )
                gvec = [gdict[k] for k in GEOMETRIC_FEATURE_NAMES]
                effnet = _get_effnet()
                vvec = effnet.extract_features(cv2.resize(image_bgr, (224, 224)))
                X2 = np.array(gvec + list(vvec), dtype=np.float32).reshape(1, -1)
                p2 = sec_pipeline.predict_proba(X2)

                # 3. Prior-normalized weighted fusion
                fusion_policy = bundle.get("fusion_policy") or {}
                prior5 = np.array(fusion_policy.get("prior5", [0.65, 0.15, 0.20]), dtype=np.float32)
                p5_norm = p5 / prior5
                p5_norm = p5_norm / np.maximum(p5_norm.sum(axis=1, keepdims=True), 1e-8)
                alpha = float(fusion_policy.get("alpha", 0.25))

                fused = alpha * p5_norm + (1.0 - alpha) * p2
                fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-8)
                avg_probs = fused[0]

                threshold = float(fusion_policy.get("conf_threshold", 0.40))
                margin_threshold = float(fusion_policy.get("margin_threshold", 0.03))
                decision_policy = {
                    "type": fusion_policy.get("type", "prior_normalized_weighted"),
                    "threshold": threshold,
                    "marginThreshold": margin_threshold,
                }
                feature_contract = "ensemble_v0.6_v2_eyecrop_geometry_effnet_onnx_pair"
            except Exception as exc:
                fallback_reason = type(exc).__name__
                logger.warning(
                    "Hirschberg ensemble secondary branch unavailable; falling back to v0.5 primary branch: %s",
                    exc,
                )
                avg_probs = p5[0]
                decision_policy = bundle.get("decision_policy") or {}
                fusion_policy = bundle.get("fusion_policy") or {}
                threshold = float(
                    decision_policy.get("threshold", fusion_policy.get("conf_threshold", 0.40)) or 0.0
                )
                margin_threshold = float(
                    decision_policy.get("margin_threshold", fusion_policy.get("margin_threshold", 0.03)) or 0.0
                )
                feature_contract = "hirschberg_pair_features_v0.5"
        else:
            if pipeline is None:
                return _unavailable_prediction(
                    "Hirschberg AI pipeline is not available on backend.",
                    "MODEL_PIPELINE_UNAVAILABLE",
                )
            X = _features_for_bundle(crops, onnx_sess, bundle)
            avg_probs = pipeline.predict_proba(X)[0]
            decision_policy = bundle.get("decision_policy") or {}
            threshold = float(decision_policy.get("threshold", 0.0) or 0.0)
            margin_threshold = float(decision_policy.get("margin_threshold", 0.0) or 0.0)
            feature_contract = bundle.get("feature_contract", "hirschberg_crop_mean_features_v0.1")

        pred_idx = int(np.argmax(avg_probs))
        pred_class = classes[pred_idx]
        confidence = float(avg_probs[pred_idx])
        sorted_probs = np.sort(avg_probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else confidence
        prob_dict = {cls_name: round(float(avg_probs[i]), 4) for i, cls_name in enumerate(classes)}

        if not is_ensemble_v2:
            decision_policy = bundle.get("decision_policy") or {}
            threshold = float(decision_policy.get("threshold", 0.0) or 0.0)
            margin_threshold = float(decision_policy.get("margin_threshold", 0.0) or 0.0)

        model_id = (
            "hirschberg-candidate-v0.5-safe"
            if fallback_reason
            else bundle.get("model_id", "hirschberg-candidate-v0.6-ensemble-v2")
        )
        dataset_version = bundle.get("dataset_version", "hirschberg-folder-labels-v0.1")
        base_payload = {
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "probabilities": prob_dict,
            "modelId": model_id,
            "datasetVersion": dataset_version,
            "featureContract": feature_contract,
            "deploymentWarning": (
                "Research-only Hirschberg candidate with limited validation. "
                "Low-confidence cases intentionally return INCONCLUSIVE. "
                "Not suitable for clinical diagnosis or clearance."
            ),
            "nonClinicalDeclaration": "Sàng lọc nghiên cứu - Không thay thế chẩn đoán bác sĩ chuyên khoa.",
        }
        if fallback_reason:
            base_payload["fallbackReason"] = fallback_reason
            base_payload["fallbackFromModelId"] = bundle.get(
                "model_id", "hirschberg-candidate-v0.6-ensemble-v2"
            )

        if confidence < threshold or margin < margin_threshold:
            return {
                "status": "INCONCLUSIVE",
                "predictedClass": "INCONCLUSIVE",
                "reason": "LOW_CONFIDENCE_RESEARCH_MODEL",
                "decisionPolicy": {
                    "type": decision_policy.get("type", "confidence_margin_abstention"),
                    "threshold": threshold,
                    "marginThreshold": margin_threshold,
                },
                **base_payload,
            }

        return {
            "status": "PREDICTED",
            "predictedClass": pred_class.upper(),
            "decisionPolicy": {
                "type": decision_policy.get("type", "confidence_margin_abstention"),
                "threshold": threshold,
                "marginThreshold": margin_threshold,
            },
            **base_payload,
        }
    except Exception as exc:
        logger.exception("Hirschberg AI prediction failed.")
        return _unavailable_prediction(
            f"Hirschberg AI prediction failed: {type(exc).__name__}.",
            "AI_RUNTIME_ERROR",
        )
