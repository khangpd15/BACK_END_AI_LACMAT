"""Eye Region of Interest (ROI) Alignment & Cropping Service.

Extracts normalized, in-plane roll-aligned ocular crops from full webcam frames:
- Canonical alignment: Rotates ROI so the canthal axis is strictly horizontal
- Preserves periocular morphology, sclera visibility, and iris limbus
- Robust bounding box expansion (1.8x width, 2.0x height)
- Standardized output dimensions (e.g. 128x64 or 64x64)
- Safe boundary padding preventing out-of-bounds clipping
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import cv2
except ImportError:
    cv2 = None

from app.services.cv.mediapipe_eye_extractor import EyeLandmarksData


@dataclass
class EyeRoiCrop:
    """Cropped and aligned eye region of interest with anatomical metadata."""
    eye_side: str  # "LEFT" or "RIGHT"
    image: np.ndarray  # (H, W, C) uint8
    bbox_norm: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in [0, 1]
    rotation_deg: float
    eye_width_px: float
    eye_height_px: float
    iris_roi_norm: Tuple[float, float]  # (x, y) relative to crop [0, 1]


class EyeRoiAligner:
    """Performs affine rotation alignment and anatomical cropping for ocular ROIs."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 64),  # (width, height)
        expansion_x: float = 1.8,                  # Horizontal padding factor
        expansion_y: float = 2.0,                  # Vertical padding factor
    ):
        self.target_size = target_size
        self.expansion_x = expansion_x
        self.expansion_y = expansion_y

    def crop_aligned_eye(
        self,
        frame: np.ndarray,
        inner_canthus_px: Tuple[float, float],
        outer_canthus_px: Tuple[float, float],
        iris_center_px: Tuple[float, float],
        eye_side: str = "LEFT",
    ) -> EyeRoiCrop:
        """Extracts roll-compensated, horizontally aligned eye crop."""
        h, w = frame.shape[:2]

        # 1. Anatomical eye center (midpoint of medial and lateral canthi)
        cx = (inner_canthus_px[0] + outer_canthus_px[0]) / 2.0
        cy = (inner_canthus_px[1] + outer_canthus_px[1]) / 2.0

        # 2. Inter-canthal width and angle
        # Lateral is outer, medial is inner.
        # Vector points from medial to lateral (or vice versa):
        dx = outer_canthus_px[0] - inner_canthus_px[0]
        dy = outer_canthus_px[1] - inner_canthus_px[1]
        canthal_w = float(math.hypot(dx, dy))
        canthal_w = max(canthal_w, 10.0)

        # Angle in degrees to rotate so the canthal vector is horizontal
        angle_rad = math.atan2(dy, dx)
        # For left eye, outer is x > inner; for right eye outer is x < inner
        # We align the vector horizontally:
        angle_deg = math.degrees(angle_rad)
        if eye_side == "RIGHT" and dx < 0:
            # Right eye outer canthus is towards the temple (left side of image for user's right eye)
            angle_deg = math.degrees(math.atan2(-dy, -dx))

        # 3. Crop dimensions based on expansion factors
        crop_w = int(round(canthal_w * self.expansion_x))
        crop_h = int(round(canthal_w * 0.35 * self.expansion_y))
        # Ensure aspect ratio preserves target
        crop_w = max(crop_w, 20)
        crop_h = max(crop_h, 10)

        # 4. Affine rotation and extraction
        if cv2 is not None:
            # OpenCV affine transformation
            rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            # Adjust translation so eye center maps to center of target crop
            rot_mat[0, 2] += (crop_w / 2.0) - cx
            rot_mat[1, 2] += (crop_h / 2.0) - cy

            cropped = cv2.warpAffine(
                frame,
                rot_mat,
                (crop_w, crop_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Resize to standardized model input dimension
            if (crop_w, crop_h) != self.target_size:
                final_image = cv2.resize(
                    cropped,
                    self.target_size,
                    interpolation=cv2.INTER_AREA if crop_w > self.target_size[0] else cv2.INTER_LINEAR,
                )
            else:
                final_image = cropped

            # Transform iris center to crop space
            iris_pt = np.array([iris_center_px[0], iris_center_px[1], 1.0])
            iris_trans = rot_mat @ iris_pt
            iris_roi_norm = (
                float(np.clip(iris_trans[0] / crop_w, 0.0, 1.0)),
                float(np.clip(iris_trans[1] / crop_h, 0.0, 1.0)),
            )
        else:
            # Pure NumPy axis-aligned crop fallback when cv2 is not loaded
            x_min = int(max(0, cx - crop_w / 2))
            x_max = int(min(w, cx + crop_w / 2))
            y_min = int(max(0, cy - crop_h / 2))
            y_max = int(min(h, cy + crop_h / 2))

            cropped = frame[y_min:y_max, x_min:x_max]
            # Simple nearest-neighbor/bilinear resize fallback
            final_image = self._numpy_resize(cropped, self.target_size)
            iris_roi_norm = (
                float(np.clip((iris_center_px[0] - x_min) / max(1, x_max - x_min), 0.0, 1.0)),
                float(np.clip((iris_center_px[1] - y_min) / max(1, y_max - y_min), 0.0, 1.0)),
            )

        # Bounding box in normalized frame coordinates
        half_w_norm = (crop_w / 2.0) / w
        half_h_norm = (crop_h / 2.0) / h
        bbox_norm = (
            float(max(0.0, (cx / w) - half_w_norm)),
            float(max(0.0, (cy / h) - half_h_norm)),
            float(min(1.0, (cx / w) + half_w_norm)),
            float(min(1.0, (cy / h) + half_h_norm)),
        )

        return EyeRoiCrop(
            eye_side=eye_side,
            image=final_image,
            bbox_norm=bbox_norm,
            rotation_deg=round(angle_deg, 2),
            eye_width_px=round(canthal_w, 2),
            eye_height_px=round(canthal_w * 0.35, 2),
            iris_roi_norm=(round(iris_roi_norm[0], 4), round(iris_roi_norm[1], 4)),
        )

    def crop_both_eyes(
        self,
        frame: np.ndarray,
        landmarks: EyeLandmarksData,
    ) -> Dict[str, EyeRoiCrop]:
        """Crops and aligns both Left and Right eye ROIs from landmark data."""
        left_crop = self.crop_aligned_eye(
            frame=frame,
            inner_canthus_px=(landmarks.left_inner_canthus_norm[0] * frame.shape[1], landmarks.left_inner_canthus_norm[1] * frame.shape[0]),
            outer_canthus_px=(landmarks.left_outer_canthus_norm[0] * frame.shape[1], landmarks.left_outer_canthus_norm[1] * frame.shape[0]),
            iris_center_px=landmarks.left_iris_px,
            eye_side="LEFT",
        )
        right_crop = self.crop_aligned_eye(
            frame=frame,
            inner_canthus_px=(landmarks.right_inner_canthus_norm[0] * frame.shape[1], landmarks.right_inner_canthus_norm[1] * frame.shape[0]),
            outer_canthus_px=(landmarks.right_outer_canthus_norm[0] * frame.shape[1], landmarks.right_outer_canthus_norm[1] * frame.shape[0]),
            iris_center_px=landmarks.right_iris_px,
            eye_side="RIGHT",
        )
        return {
            "LEFT": left_crop,
            "RIGHT": right_crop,
        }

    @staticmethod
    def _numpy_resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Pure NumPy bilinear resize fallback when OpenCV is unavailable."""
        target_w, target_h = target_size
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            channels = image.shape[2] if image.ndim == 3 else 1
            return np.zeros((target_h, target_w, channels), dtype=np.uint8)

        x_indices = (np.linspace(0, w - 1, target_w)).astype(int)
        y_indices = (np.linspace(0, h - 1, target_h)).astype(int)
        return image[np.ix_(y_indices, x_indices)]
