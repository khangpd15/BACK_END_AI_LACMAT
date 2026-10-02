"""Strabismus Deep Learning Inference Service for RemiCare.

Provides high-performance, single-instance ONNX Runtime inference:
- Singleton lifecycle: model is loaded once at FastAPI startup and kept resident in RAM.
- Strict Quality Gate pre-evaluation (blur, illumination, bilateral ocular presence).
- Locked clinical screening threshold: 0.20 (>= 0.20 -> SUSPICIOUS, < 0.20 -> NORMAL).
- Zero raw biometric image storage; in-memory buffer cleanup via finally blocks.
"""

from io import BytesIO
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.config import PROJECT_ROOT
from cv.quality_gate import evaluate_quality

logger = logging.getLogger("remicare.ai.strabismus")

# Configuration constants
DEFAULT_MODEL_FILENAME = "best_model.onnx"
DEFAULT_MODEL_VERSION = "remicare-bilateral-resnet18-v1"
SCREENING_THRESHOLD = 0.20

# Normalization constants (ImageNet standard matching training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)



def validate_bilateral_eye_roi_contract(
    img_bgr: np.ndarray,
    orig_w: int,
    orig_h: int
) -> Tuple[bool, str, str]:
    """Safety guard: validates that the input image satisfies the Bilateral Eye ROI contract.
    
    MODEL INPUT CONTRACT:
    - Input: Bilateral ocular ROI (covers both eyes, inner/outer canthi, glabella; no full face).
    - Expected shape: 224x224x3 (or pre-cropped ocular strip with aspect ratio >= 1.4).
    - Color: RGB.
    - Normalization: ImageNet (Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]).
    - Inference augmentation: NONE.
    - Expected content: Both eyes visible, primary frontal gaze.
    - Defensive rejection: uncropped full-face webcam images (e.g. 640x480, 1280x720, or portrait < 0.8).
    """
    aspect_ratio = orig_w / float(max(1, orig_h))
    
    # 1. Reject full webcam face images: e.g. 640x480, 1280x720, or portrait orientation
    if (orig_w >= 450 and orig_h >= 340 and aspect_ratio < 2.0) or (aspect_ratio < 0.75):
        return False, "FULL_FACE_NOT_ACCEPTED", (
            f"Kích thước ảnh {orig_w}x{orig_h} (tỉ lệ {aspect_ratio:.2f}) là ảnh toàn khuôn mặt. "
            f"Mô hình yêu cầu vùng cắt 2 mắt (Bilateral Eye ROI)."
        )
    
    # 2. Reject tiny crops
    if orig_w < 80 or orig_h < 35:
        return False, "ROI_TOO_SMALL", f"Kích thước ROI {orig_w}x{orig_h} quá nhỏ để phân tích."
        
    # 3. Bilateral ocular symmetry & contrast check
    if cv2 is not None and len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        left_quarter = gray[:, :int(w * 0.40)]
        right_quarter = gray[:, int(w * 0.60):]
        
        left_std = float(np.std(left_quarter))
        right_std = float(np.std(right_quarter))
        
        if left_std < 6.0 or right_std < 6.0:
            return False, "LOW_BILATERAL_CONTRAST", (
                f"Độ tương phản 2 vùng mắt không đồng đều (L={left_std:.1f}, R={right_std:.1f})."
            )
            
    return True, "PASS", "Hợp lệ chuẩn Bilateral Eye ROI"


class StrabismusInferenceService:
    """Production service for bilateral strabismus screening using ONNX Runtime."""

    _instance: Optional["StrabismusInferenceService"] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        threshold: float = SCREENING_THRESHOLD,
    ) -> None:
        self.model_version = model_version
        self.threshold = threshold
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.resolved_model_path: Optional[str] = None
        self.is_loaded: bool = False

        self._resolve_and_load_model(model_path)

    def _resolve_and_load_model(self, custom_path: Optional[str] = None) -> None:
        """Resolves model path from env or parameters and initializes ONNX session."""
        env_model_path = os.getenv("MODEL_PATH")
        candidate_paths = []

        if custom_path:
            candidate_paths.append(Path(custom_path))
        if env_model_path:
            candidate_paths.append(Path(env_model_path))
            candidate_paths.append(PROJECT_ROOT / env_model_path)

        # Standard relative fallback locations
        candidate_paths.extend([
            PROJECT_ROOT / "app" / "models" / DEFAULT_MODEL_FILENAME,
            PROJECT_ROOT / "models" / DEFAULT_MODEL_FILENAME,
            Path("app/models") / DEFAULT_MODEL_FILENAME,
            Path("models") / DEFAULT_MODEL_FILENAME,
        ])

        found_path: Optional[Path] = None
        for p in candidate_paths:
            try:
                resolved = p.resolve()
                if resolved.is_file() and resolved.stat().st_size > 1000:
                    found_path = resolved
                    break
            except Exception:
                continue

        if not found_path:
            logger.error(
                "[StrabismusInferenceService] Model file not found in candidates: %s. "
                "Ensure MODEL_PATH points to a valid ONNX model file.",
                [str(p) for p in candidate_paths],
            )
            self.is_loaded = False
            return

        self.resolved_model_path = str(found_path)
        logger.info(
            "[StrabismusInferenceService] Initializing ONNX Runtime session from: %s",
            self.resolved_model_path,
        )

        if ort is None:
            logger.error("[StrabismusInferenceService] onnxruntime is not installed!")
            self.is_loaded = False
            return

        try:
            # Configure ONNX session options for optimized CPU inference
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = max(1, os.cpu_count() or 2)
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                self.resolved_model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.is_loaded = True
            logger.info(
                "[StrabismusInferenceService] Model successfully resident in RAM. input_name=%s, version=%s",
                self.input_name,
                self.model_version,
            )
        except Exception as e:
            logger.error(
                "[StrabismusInferenceService] Failed to load ONNX session: %s", e, exc_info=True
            )
            self.is_loaded = False

    def preprocess_image(self, pil_image: Image.Image) -> np.ndarray:
        """Preprocesses PIL image into normalized float32 tensor of shape (1, 3, 224, 224)."""
        # Resize to 224x224 RGB
        img = pil_image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]

        # Convert to CHW: (3, 224, 224)
        tensor = np.transpose(img_np, (2, 0, 1))
        # Add batch dim: (1, 3, 224, 224)
        tensor = np.expand_dims(tensor, axis=0)

        # Normalize with ImageNet mean and std
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor.astype(np.float32)

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Performs Defensive ROI Contract validation, Quality Gate check and Deep Learning screening.
        
        Zero biometric image storage: buffers are held transiently in RAM and discarded.
        
        Returns:
            Dict matching specification:
            - status: "NORMAL" | "SUSPICIOUS" | "INCONCLUSIVE"
            - prediction: "NORMAL" | "STRABISMUS" | "INCONCLUSIVE"
            - confidence: Optional[float]
            - confidence_type: "MODEL_SOFTMAX"
            - screening_status: "AI_SIGNAL"
            - strabismus_probability: Optional[float]
            - prob_normal: Optional[float]
            - prob_strabismus: Optional[float]
            - quality: "PASS" | "FAIL"
            - quality_score: float
            - threshold: float (0.20)
            - model_version: str
            - inference_latency_ms: Optional[int]
            - disclaimer: str
            - reason: Optional[str] (when INCONCLUSIVE)
        """
        start_time = time.perf_counter()

        # 1. Decode image into memory
        pil_img: Optional[Image.Image] = None
        img_bgr: Optional[np.ndarray] = None

        try:
            pil_img = Image.open(BytesIO(image_bytes))
            pil_img.load()  # Force decode
            orig_w, orig_h = pil_img.size
            # Convert to BGR array for OpenCV quality gate and defensive check
            rgb_arr = np.array(pil_img.convert("RGB"))
            if cv2 is not None:
                img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = rgb_arr[:, :, ::-1]  # RGB to BGR slice
        except Exception as decode_err:
            logger.warning("[StrabismusInferenceService] Malformed image decode error: %s", decode_err)
            return {
                "status": "INCONCLUSIVE",
                "prediction": "INCONCLUSIVE",
                "reason": "IMAGE_QUALITY_FAILED",
                "quality": "FAIL",
                "quality_score": 0.0,
                "confidence": None,
                "confidence_type": "MODEL_SOFTMAX",
                "screening_status": "AI_SIGNAL",
                "strabismus_probability": None,
                "prob_normal": None,
                "prob_strabismus": None,
                "threshold": self.threshold,
                "model_version": self.model_version,
                "inference_latency_ms": int(round((time.perf_counter() - start_time) * 1000.0)),
                "disclaimer": "Độ tự tin thể hiện mức độ tự tin toán học của mô hình đối với mẫu ảnh, không phải xác suất mắc bệnh.",
            }

        try:
            # 2. Defensive Preprocessing: Validate Bilateral Eye ROI Contract
            is_valid_roi, roi_reason, roi_msg = validate_bilateral_eye_roi_contract(img_bgr, orig_w, orig_h)
            if not is_valid_roi:
                logger.warning(
                    "[StrabismusInferenceService] Rejected by ROI Safety Guard: reason=%s, msg=%s, dims=%dx%d",
                    roi_reason,
                    roi_msg,
                    orig_w,
                    orig_h,
                )
                return {
                    "status": "INCONCLUSIVE",
                    "prediction": "INCONCLUSIVE",
                    "reason": roi_reason,
                    "message": roi_msg,
                    "quality": "FAIL_ROI_CONTRACT",
                    "quality_score": 0.0,
                    "confidence": None,
                    "confidence_type": "MODEL_SOFTMAX",
                    "screening_status": "AI_SIGNAL",
                    "strabismus_probability": None,
                    "prob_normal": None,
                    "prob_strabismus": None,
                    "threshold": self.threshold,
                    "model_version": self.model_version,
                    "inference_latency_ms": int(round((time.perf_counter() - start_time) * 1000.0)),
                    "disclaimer": "Độ tự tin thể hiện mức độ tự tin toán học của mô hình đối với mẫu ảnh, không phải xác suất mắc bệnh.",
                }

            # 3. Quality Gate Evaluation (blur, brightness)
            passed_qg, q_score, q_metrics, q_reason = evaluate_quality(
                img_bgr,
                min_blur_var=50.0,
                min_brightness=35.0,
                max_brightness=240.0,
            )

            if not passed_qg:
                logger.info(
                    "[StrabismusInferenceService] Quality gate rejected: reason=%s, score=%.2f, metrics=%s",
                    q_reason,
                    q_score,
                    q_metrics,
                )
                return {
                    "status": "INCONCLUSIVE",
                    "prediction": "INCONCLUSIVE",
                    "reason": "IMAGE_QUALITY_FAILED",
                    "quality": "FAIL_QUALITY_GATE",
                    "quality_score": round(q_score, 2),
                    "confidence": None,
                    "confidence_type": "MODEL_SOFTMAX",
                    "screening_status": "AI_SIGNAL",
                    "strabismus_probability": None,
                    "prob_normal": None,
                    "prob_strabismus": None,
                    "threshold": self.threshold,
                    "model_version": self.model_version,
                    "inference_latency_ms": int(round((time.perf_counter() - start_time) * 1000.0)),
                    "disclaimer": "Độ tự tin thể hiện mức độ tự tin toán học của mô hình đối với mẫu ảnh, không phải xác suất mắc bệnh.",
                }

            # 4. Model Readiness Check
            if not self.is_loaded or self.session is None or not self.input_name:
                logger.error("[StrabismusInferenceService] Inference requested but ONNX session is not loaded.")
                raise RuntimeError("Strabismus ONNX screening model is not ready.")

            # Log contract confirmation
            logger.info(
                "[STRABISMUS AI] Input type: BILATERAL_EYE_ROI | Size: %dx%d | RGB: true | Contract: PASS | Quality: PASS",
                orig_w,
                orig_h,
            )

            # 5. Deep Learning Inference
            input_tensor = self.preprocess_image(pil_img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0]  # Shape: (1, 2)

            # Softmax calculation
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            prob_normal = float(probs[0, 0])
            prob_strabismus = float(probs[0, 1])

            # 6. Threshold Rule: locked at 0.20
            # probability >= 0.20 -> SUSPICIOUS
            # probability < 0.20 -> NORMAL
            if prob_strabismus >= self.threshold:
                status = "SUSPICIOUS"
                prediction = "STRABISMUS"
                confidence = prob_strabismus
            else:
                status = "NORMAL"
                prediction = "NORMAL"
                confidence = prob_normal

            latency_ms = int(round((time.perf_counter() - start_time) * 1000.0))

            # Phase 13 Compliance: Structured Model Inference Log (zero PII, zero base64)
            logger.info(
                "[MODEL] Model: %s | Input: %dx%d | Normal: %.4f | Strabismus: %.4f | Prediction: %s | Confidence: %.2f%%",
                self.model_version,
                orig_w,
                orig_h,
                prob_normal,
                prob_strabismus,
                prediction,
                confidence * 100.0,
            )

            return {
                "status": status,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "confidence_type": "MODEL_SOFTMAX",
                "screening_status": "AI_SIGNAL",
                "strabismus_probability": round(prob_strabismus, 2),
                "prob_normal": round(prob_normal, 2),
                "prob_strabismus": round(prob_strabismus, 2),
                "quality": "PASS",
                "quality_score": round(q_score, 2),
                "threshold": self.threshold,
                "model_version": self.model_version,
                "inference_latency_ms": latency_ms,
                "disclaimer": "Độ tự tin thể hiện mức độ tự tin toán học của mô hình đối với mẫu ảnh, không phải xác suất mắc bệnh.",
            }

        finally:
            # Strict cleanup: release memory references
            del pil_img
            del img_bgr


# Global singleton instance
_inference_service: Optional[StrabismusInferenceService] = None


def get_strabismus_inference_service() -> StrabismusInferenceService:
    """Returns singleton instance of StrabismusInferenceService held resident in RAM."""
    global _inference_service
    if _inference_service is None:
        _inference_service = StrabismusInferenceService()
    return _inference_service
