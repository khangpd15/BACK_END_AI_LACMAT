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
        """Performs Quality Gate check and Deep Learning screening on raw image bytes.
        
        Zero biometric image storage: buffers are held transiently in RAM and discarded.
        
        Returns:
            Dict matching specification:
            - status: "NORMAL" | "SUSPICIOUS" | "INCONCLUSIVE"
            - strabismus_probability: Optional[float]
            - confidence: Optional[float]
            - quality_score: float
            - threshold: float (0.20)
            - model_version: str
            - inference_latency_ms: Optional[int]
            - reason: Optional[str] (when INCONCLUSIVE)
        """
        start_time = time.perf_counter()

        # 1. Decode image into memory
        pil_img: Optional[Image.Image] = None
        img_bgr: Optional[np.ndarray] = None

        try:
            pil_img = Image.open(BytesIO(image_bytes))
            pil_img.load()  # Force decode
            # Convert to BGR array for OpenCV quality gate
            rgb_arr = np.array(pil_img.convert("RGB"))
            if cv2 is not None:
                img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = rgb_arr[:, :, ::-1]  # RGB to BGR slice
        except Exception as decode_err:
            logger.warning("[StrabismusInferenceService] Malformed image decode error: %s", decode_err)
            return {
                "status": "INCONCLUSIVE",
                "reason": "IMAGE_QUALITY_FAILED",
                "quality_score": 0.0,
                "strabismus_probability": None,
                "confidence": None,
                "threshold": self.threshold,
                "model_version": self.model_version,
                "inference_latency_ms": int(round((time.perf_counter() - start_time) * 1000.0)),
            }

        try:
            # 2. Quality Gate Evaluation
            passed_qg, q_score, q_metrics, q_reason = evaluate_quality(
                img_bgr,
                min_blur_var=60.0,
                min_brightness=40.0,
                max_brightness=235.0,
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
                    "reason": "IMAGE_QUALITY_FAILED",
                    "quality_score": round(q_score, 2),
                    "strabismus_probability": None,
                    "confidence": None,
                    "threshold": self.threshold,
                    "model_version": self.model_version,
                    "inference_latency_ms": int(round((time.perf_counter() - start_time) * 1000.0)),
                }

            # 3. Model Readiness Check
            if not self.is_loaded or self.session is None or not self.input_name:
                logger.error("[StrabismusInferenceService] Inference requested but ONNX session is not loaded.")
                raise RuntimeError("Strabismus ONNX screening model is not ready.")

            # 4. Deep Learning Inference
            input_tensor = self.preprocess_image(pil_img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0]  # Shape: (1, 2)

            # Softmax calculation
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            prob_normal = float(probs[0, 0])
            prob_strabismus = float(probs[0, 1])

            # 5. Threshold Rule: locked at 0.20
            # probability >= 0.20 -> SUSPICIOUS
            # probability < 0.20 -> NORMAL
            if prob_strabismus >= self.threshold:
                status = "SUSPICIOUS"
                confidence = prob_strabismus
            else:
                status = "NORMAL"
                confidence = prob_normal

            latency_ms = int(round((time.perf_counter() - start_time) * 1000.0))

            return {
                "status": status,
                "strabismus_probability": round(prob_strabismus, 2),
                "confidence": round(confidence, 2),
                "quality_score": round(q_score, 2),
                "threshold": self.threshold,
                "model_version": self.model_version,
                "inference_latency_ms": latency_ms,
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
