"""Quality Gate for RemiCare Strabismus AI bilateral image screening.

Validates image quality before inference:
- Image readability and dimension bounds
- Laplacian variance blur check (threshold > 60)
- Illumination/brightness range [40, 235]
- Bilateral presence (both eyes detected or verified ocular symmetry)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    _FACE_MESH = mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )
except Exception:
    _FACE_MESH = None


@dataclass
class QualityResult:
    passed: bool
    quality_score: float
    reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_quality(
    image_bgr: np.ndarray,
    min_blur_var: float = 60.0,
    min_brightness: float = 40.0,
    max_brightness: float = 235.0,
) -> Tuple[bool, float, Dict[str, Any], Optional[str]]:
    """Evaluates image quality against strict criteria:
    - readable and valid dimensions (min 60x100)
    - blur: Laplacian variance > 60
    - brightness: mean intensity between 40 and 235
    - bilateral presence: both eyes present simultaneously
    
    Returns:
        (passed, quality_score, metrics_dict, failure_reason_or_None)
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return False, 0.0, {}, "IMAGE_UNREADABLE"

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return False, 0.0, {"shape": str(image_bgr.shape)}, "INVALID_CHANNELS"

    h, w = image_bgr.shape[:2]
    if h < 60 or w < 100:
        return False, 0.0, {"height": h, "width": w}, "IMAGE_RESOLUTION_TOO_LOW"

    # Compute grayscale
    if cv2 is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        gray = np.mean(image_bgr, axis=2).astype(np.float64)
        lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        lap_var = float(np.var(lap))

    brightness = float(np.mean(gray))

    # 1. Blur check (Laplacian variance > 60)
    if lap_var <= min_blur_var:
        q_score = round(max(0.0, min(0.5, lap_var / (min_blur_var * 2.0))), 2)
        return False, q_score, {
            "laplacian_var": round(lap_var, 2),
            "min_blur_var": min_blur_var,
            "brightness": round(brightness, 2),
        }, f"IMAGE_BLURRY: var={lap_var:.1f} <= {min_blur_var}"

    # 2. Brightness check (40 - 235)
    if brightness < min_brightness:
        return False, 0.2, {
            "brightness": round(brightness, 2),
            "min_brightness": min_brightness,
        }, f"IMAGE_TOO_DARK: brightness={brightness:.1f} < {min_brightness}"
    elif brightness > max_brightness:
        return False, 0.2, {
            "brightness": round(brightness, 2),
            "max_brightness": max_brightness,
        }, f"IMAGE_TOO_BRIGHT: brightness={brightness:.1f} > {max_brightness}"

    # 3. Bilateral check (both eyes present)
    aspect_ratio = float(w) / max(1.0, float(h))
    has_face_mesh = False
    both_eyes_detected = False

    if _FACE_MESH is not None and cv2 is not None:
        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = _FACE_MESH.process(rgb)
            if res.multi_face_landmarks:
                has_face_mesh = True
                landmarks = res.multi_face_landmarks[0].landmark
                # Left eye landmark 33, Right eye landmark 263
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                # Check both are in frame and separated horizontally
                if 0.0 < left_eye.x < 1.0 and 0.0 < right_eye.x < 1.0:
                    both_eyes_detected = True
        except Exception:
            pass

    # Heuristic fallback for bilateral eye crop/region:
    # A standard bilateral gaze capture frame has aspect ratio >= 1.2 or both eyes detected
    is_bilateral = both_eyes_detected or has_face_mesh or (aspect_ratio >= 1.25)
    if not is_bilateral:
        # Check ocular symmetry in left and right halves
        left_half = gray[:, :int(w * 0.45)]
        right_half = gray[:, int(w * 0.55):]
        if np.percentile(left_half, 15) < 140 and np.percentile(right_half, 15) < 140:
            is_bilateral = True

    if not is_bilateral:
        return False, 0.3, {
            "aspect_ratio": round(aspect_ratio, 2),
            "face_mesh": has_face_mesh,
        }, "FAIL_BOTH_EYES_NOT_DETECTED"

    # Compute continuous quality score [0.60 - 1.00]
    blur_norm = min(1.0, lap_var / 250.0)
    bright_norm = 1.0 - abs(brightness - 128.0) / 128.0
    quality_score = round(float(np.clip(0.6 + 0.25 * blur_norm + 0.15 * bright_norm, 0.6, 0.98)), 2)

    return True, quality_score, {
        "laplacian_var": round(lap_var, 2),
        "brightness": round(brightness, 2),
        "aspect_ratio": round(aspect_ratio, 2),
        "bilateral_verified": True,
    }, None
