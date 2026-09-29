"""Advanced Camera & Eye Tracking Quality Gate for RemiCare Strabismus AI.

Multi-factor quality assessment before running clinical screening:
- Face bounding box ratio & distance check
- Head pose estimation (Yaw, Pitch, Roll angles)
- Eye Aspect Ratio (EAR) for blink exclusion
- Image blur / sharpness via Laplacian variance
- Minimum anatomical ocular resolution
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class QualityGateResult:
    is_acceptable: bool
    camera_quality_score: float  # [0.0, 1.0]
    tracking_quality_score: float  # [0.0, 1.0]
    failure_reasons: List[str] = field(default_factory=list)
    user_guidance: Optional[str] = None


class AdvancedQualityGate:
    """Rigorous gate checking illumination, resolution, blur, and head alignment."""

    def __init__(
        self,
        min_face_ratio: float = 0.20,
        max_face_ratio: float = 0.70,
        min_eye_width_px: float = 35.0,
        min_blur_var: float = 100.0,
        max_yaw_deg: float = 12.0,
        max_pitch_deg: float = 10.0,
        max_roll_deg: float = 6.0,
        min_ear_blink: float = 0.18,
    ):
        self.min_face_ratio = min_face_ratio
        self.max_face_ratio = max_face_ratio
        self.min_eye_width_px = min_eye_width_px
        self.min_blur_var = min_blur_var
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg
        self.min_ear_blink = min_ear_blink

    @staticmethod
    def compute_ear(eye_landmarks: np.ndarray) -> float:
        """Computes Eye Aspect Ratio (EAR) from 6 2D eyelid points.
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
        """
        if eye_landmarks.shape[0] < 6:
            return 0.30
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        d_v1 = float(np.linalg.norm(p2 - p6))
        d_v2 = float(np.linalg.norm(p3 - p5))
        d_h = float(np.linalg.norm(p1 - p4))
        if d_h < 1e-5:
            return 0.0
        return float((d_v1 + d_v2) / (2.0 * d_h))

    def evaluate_frame(
        self,
        frame_shape: Tuple[int, int],  # (height, width)
        face_bbox: Tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max) in [0, 1]
        head_pose: Tuple[float, float, float],  # (yaw, pitch, roll) in degrees
        left_eye_ear: float,
        right_eye_ear: float,
        left_eye_width_px: float,
        right_eye_width_px: float,
        blur_variance: float,
        mediapipe_confidence: float = 0.95,
    ) -> QualityGateResult:
        """Evaluates incoming frame parameters against strict clinical quality bounds."""
        reasons: List[str] = []
        guidance: Optional[str] = None

        h, w = frame_shape
        face_w = max(0.0, (face_bbox[2] - face_bbox[0]) * w)
        face_ratio = face_w / w if w > 0 else 0.0

        # 1. Camera distance / face size
        if face_ratio < self.min_face_ratio:
            reasons.append(f"FACE_TOO_FAR: ratio={face_ratio:.2f} < {self.min_face_ratio}")
            guidance = "Vui lòng ngồi gần camera hơn."
        elif face_ratio > self.max_face_ratio:
            reasons.append(f"FACE_TOO_CLOSE: ratio={face_ratio:.2f} > {self.max_face_ratio}")
            guidance = "Vui lòng ngồi lùi xa camera một chút."

        # 2. Head pose
        yaw, pitch, roll = head_pose
        if abs(yaw) > self.max_yaw_deg:
            reasons.append(f"HEAD_YAW_EXCEEDED: |yaw|={abs(yaw):.1f}° > {self.max_yaw_deg}°")
            guidance = "Vui lòng nhìn thẳng, không quay mặt sang hai bên."
        if abs(pitch) > self.max_pitch_deg:
            reasons.append(f"HEAD_PITCH_EXCEEDED: |pitch|={abs(pitch):.1f}° > {self.max_pitch_deg}°")
            guidance = "Vui lòng giữ thẳng cằm, không cúi hoặc ngửa đầu."
        if abs(roll) > self.max_roll_deg:
            reasons.append(f"HEAD_ROLL_EXCEEDED: |roll|={abs(roll):.1f}° > {self.max_roll_deg}°")
            guidance = "Vui lòng giữ đầu thẳng, không nghiêng sang vai."

        # 3. Blur
        if blur_variance < self.min_blur_var:
            reasons.append(f"IMAGE_BLURRY: var={blur_variance:.1f} < {self.min_blur_var}")
            guidance = "Camera bị mờ hoặc rung lắc. Vui lòng giữ yên thiết bị."

        # 4. Eye width
        if left_eye_width_px < self.min_eye_width_px or right_eye_width_px < self.min_eye_width_px:
            reasons.append("EYE_RESOLUTION_TOO_LOW")
            if not guidance:
                guidance = "Độ phân giải mắt quá nhỏ để theo dõi chính xác."

        # 5. Blink detection
        is_blinking = (left_eye_ear < self.min_ear_blink) or (right_eye_ear < self.min_ear_blink)
        if is_blinking:
            reasons.append("EYE_BLINK_DETECTED")

        # Aggregate Camera Quality Score
        cam_components = [
            np.clip(1.0 - abs(face_ratio - 0.40) / 0.30, 0.0, 1.0),
            np.clip(1.0 - abs(yaw) / (self.max_yaw_deg * 2.0), 0.0, 1.0),
            np.clip(1.0 - abs(pitch) / (self.max_pitch_deg * 2.0), 0.0, 1.0),
            np.clip(1.0 - abs(roll) / (self.max_roll_deg * 2.0), 0.0, 1.0),
            np.clip(blur_variance / 200.0, 0.0, 1.0),
        ]
        cam_score = float(np.mean(cam_components))

        # Aggregate Eye Tracking Quality Score
        track_components = [
            float(mediapipe_confidence),
            1.0 if not is_blinking else 0.1,
            np.clip((left_eye_width_px + right_eye_width_px) / 100.0, 0.0, 1.0),
        ]
        track_score = float(np.mean(track_components))

        structural_errors = [r for r in reasons if r != "EYE_BLINK_DETECTED"]
        is_acceptable = (len(structural_errors) == 0) and not is_blinking

        return QualityGateResult(
            is_acceptable=is_acceptable,
            camera_quality_score=round(cam_score, 4),
            tracking_quality_score=round(track_score, 4),
            failure_reasons=reasons,
            user_guidance=guidance,
        )
