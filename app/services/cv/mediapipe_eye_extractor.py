"""MediaPipe Face & Iris Landmark Extractor for RemiCare Strabismus AI.

Extracts high-fidelity anatomical eye landmarks using MediaPipe Face Mesh / Tasks:
- Left & Right Iris Centers (landmarks 468, 473)
- Left & Right Iris Boundaries (landmarks 469-472, 474-477)
- Palpebral Fissures: Medial/Lateral Canthi (Left: 362, 263; Right: 133, 33)
- Superior/Inferior Palpebral Margins (Left: 386, 374; Right: 159, 145)
- Full 16-point eye contours for precise periocular segmentation
- Eye-to-camera distance estimation via iris distance: D_cm = 4095 / irisDistancePx
- Head pose estimation (Yaw, Pitch, Roll)
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import cv2
except ImportError:
    cv2 = None

try:
    # pyrefly: ignore [missing-import]
    import mediapipe as mp
except ImportError:
    mp = None


# Canonical MediaPipe Face Mesh Landmark Indices
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# Left Eye: Medial (inner) 362, Lateral (outer) 263
LEFT_INNER_CANTHUS = 362
LEFT_OUTER_CANTHUS = 263
LEFT_TOP_EYELID = 386
LEFT_BOT_EYELID = 374
LEFT_VERTICAL_1 = (385, 380)
LEFT_VERTICAL_2 = (386, 374)

# Right Eye: Lateral (outer) 33, Medial (inner) 133
RIGHT_OUTER_CANTHUS = 33
RIGHT_INNER_CANTHUS = 133
RIGHT_TOP_EYELID = 159
RIGHT_BOT_EYELID = 145
RIGHT_VERTICAL_1 = (158, 153)
RIGHT_VERTICAL_2 = (159, 145)

# Full contours (16 points per eye)
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Key face landmarks for bounding box and head pose
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_LEFT_EDGE = 454
FACE_RIGHT_EDGE = 234
FACE_FOREHEAD = 10
MOUTH_LEFT = 291
MOUTH_RIGHT = 61


@dataclass
class EyeLandmarksData:
    """Structured container for single-frame eye and facial landmark metrics."""
    # Iris centers (normalized [0, 1] and pixel coordinates)
    left_iris_norm: Tuple[float, float]
    right_iris_norm: Tuple[float, float]
    left_iris_px: Tuple[float, float]
    right_iris_px: Tuple[float, float]

    # Canthi (inner/medial and outer/lateral)
    left_inner_canthus_norm: Tuple[float, float]
    left_outer_canthus_norm: Tuple[float, float]
    right_inner_canthus_norm: Tuple[float, float]
    right_outer_canthus_norm: Tuple[float, float]

    # Eyelids (superior and inferior margins)
    left_eyelid_top_norm: Tuple[float, float]
    left_eyelid_bot_norm: Tuple[float, float]
    right_eyelid_top_norm: Tuple[float, float]
    right_eyelid_bot_norm: Tuple[float, float]

    # Eye Aspect Ratio (EAR)
    left_ear: float
    right_ear: float

    # Dimensions in pixels
    left_eye_width_px: float
    right_eye_width_px: float
    ipd_px: float
    ipd_norm: float

    # Estimated eye-to-screen physical distance (cm)
    estimated_distance_cm: float

    # Head pose angles (degrees)
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll)

    # Face bounding box (xmin, ymin, xmax, ymax) in [0, 1]
    face_bbox: Tuple[float, float, float, float]

    # Validity flags
    left_eye_valid: bool = True
    right_eye_valid: bool = True
    confidence: float = 0.95

    # Raw landmark arrays (optional)
    left_contour_px: Optional[List[Tuple[float, float]]] = None
    right_contour_px: Optional[List[Tuple[float, float]]] = None


class MediaPipeEyeExtractor:
    """High-accuracy eye landmark extractor integrating MediaPipe Face/Iris models."""

    def __init__(
        self,
        ref_distance_constant: float = 4095.0,  # D_cm = 4095 / iris_distance_px
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.ref_distance_constant = ref_distance_constant
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._mesh = None

    def _init_mesh(self):
        """Lazy-load MediaPipe FaceMesh with Iris landmarks refinement."""
        if self._mesh is None and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Enables 478 iris landmarks
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    @staticmethod
    def compute_ear_from_landmarks(
        landmarks: List[Any],
        corners: Tuple[int, int],
        v1: Tuple[int, int],
        v2: Tuple[int, int],
        frame_w: float = 1.0,
        frame_h: float = 1.0,
    ) -> float:
        """Computes Eye Aspect Ratio (EAR) using the dual-vertical projection formula.
        
        EAR = (||v1_top - v1_bot|| + ||v2_top - v2_bot||) / (2.0 * ||inner - outer||)
        """
        def get_xy(idx):
            lm = landmarks[idx]
            if hasattr(lm, "x"):
                return np.array([lm.x * frame_w, lm.y * frame_h])
            return np.array([lm[0] * frame_w, lm[1] * frame_h])

        p_inner = get_xy(corners[0])
        p_outer = get_xy(corners[1])
        p_v1_top = get_xy(v1[0])
        p_v1_bot = get_xy(v1[1])
        p_v2_top = get_xy(v2[0])
        p_v2_bot = get_xy(v2[1])

        d_v1 = float(np.linalg.norm(p_v1_top - p_v1_bot))
        d_v2 = float(np.linalg.norm(p_v2_top - p_v2_bot))
        d_h = float(np.linalg.norm(p_inner - p_outer))

        if d_h < 1e-5:
            return 0.0
        return float((d_v1 + d_v2) / (2.0 * d_h))

    @staticmethod
    def estimate_head_pose_from_landmarks(
        landmarks: List[Any],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        """Estimates head pose (Yaw, Pitch, Roll) in degrees from key 3D facial landmarks.
        
        Uses solvePnP when cv2 is present, or direct projective trigonometry.
        """
        def get_pt(idx):
            lm = landmarks[idx]
            if hasattr(lm, "x"):
                return np.array([lm.x * frame_w, lm.y * frame_h, getattr(lm, "z", 0.0) * frame_w])
            z = lm[2] * frame_w if len(lm) > 2 else 0.0
            return np.array([lm[0] * frame_w, lm[1] * frame_h, z])

        # Key points: Nose tip (1), Chin (152), Left eye outer (263), Right eye outer (33), Left mouth (291), Right mouth (61)
        p_nose = get_pt(FACE_NOSE_TIP)
        p_chin = get_pt(FACE_CHIN)
        p_left_eye = get_pt(LEFT_OUTER_CANTHUS)
        p_right_eye = get_pt(RIGHT_OUTER_CANTHUS)
        p_left_mouth = get_pt(MOUTH_LEFT)
        p_right_mouth = get_pt(MOUTH_RIGHT)

        if cv2 is not None:
            # Standard 3D anthropometric face model (mm)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye outer corner
                (225.0, 170.0, -135.0),      # Right eye outer corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype=np.float64)

            image_points = np.array([
                [p_nose[0], p_nose[1]],
                [p_chin[0], p_chin[1]],
                [p_left_eye[0], p_left_eye[1]],
                [p_right_eye[0], p_right_eye[1]],
                [p_left_mouth[0], p_left_mouth[1]],
                [p_right_mouth[0], p_right_mouth[1]],
            ], dtype=np.float64)

            focal_length = frame_w
            center = (frame_w / 2.0, frame_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vector, _ = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                # Compute Euler angles from rotation matrix
                sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    pitch = math.atan2(rmat[2, 1], rmat[2, 2])
                    yaw = math.atan2(-rmat[2, 0], sy)
                    roll = math.atan2(rmat[1, 0], rmat[0, 0])
                else:
                    pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
                    yaw = math.atan2(-rmat[2, 0], sy)
                    roll = 0.0
                return (
                    float(math.degrees(yaw)),
                    float(math.degrees(pitch)),
                    float(math.degrees(roll)),
                )

        # Trigonometric fallback if cv2 not available or solvePnP fails
        eye_dx = p_left_eye[0] - p_right_eye[0]
        eye_dy = p_left_eye[1] - p_right_eye[1]
        roll = math.degrees(math.atan2(eye_dy, eye_dx))

        # Yaw approximation from nose-to-eye distances
        dist_left = np.linalg.norm(p_left_eye[:2] - p_nose[:2])
        dist_right = np.linalg.norm(p_right_eye[:2] - p_nose[:2])
        total_dist = dist_left + dist_right
        yaw = float(math.degrees((dist_left - dist_right) / total_dist * 0.75)) if total_dist > 1e-4 else 0.0

        # Pitch approximation from nose-to-chin vertical ratio
        eye_mid_y = (p_left_eye[1] + p_right_eye[1]) / 2.0
        face_h = abs(p_chin[1] - eye_mid_y)
        nose_rel_y = (p_nose[1] - eye_mid_y) / face_h if face_h > 1e-4 else 0.5
        pitch = float((nose_rel_y - 0.45) * 50.0)

        return (round(yaw, 2), round(pitch, 2), round(roll, 2))

    def extract_from_landmarks(
        self,
        landmarks: List[Any],
        frame_shape: Tuple[int, int],  # (height, width)
        confidence: float = 0.95,
    ) -> EyeLandmarksData:
        """Parses a 468 or 478-landmark array into a full EyeLandmarksData structure."""
        frame_h, frame_w = frame_shape

        def get_norm(idx: int) -> Tuple[float, float]:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if hasattr(lm, "x"):
                    return (float(lm.x), float(lm.y))
                return (float(lm[0]), float(lm[1]))
            return (0.5, 0.5)

        def norm_to_px(xy: Tuple[float, float]) -> Tuple[float, float]:
            return (xy[0] * frame_w, xy[1] * frame_h)

        # Iris centers: if 478 landmarks available, use 468 and 473. Else approximate from eye centers.
        has_iris = len(landmarks) >= 478
        if has_iris:
            left_iris_norm = get_norm(LEFT_IRIS_CENTER)
            right_iris_norm = get_norm(RIGHT_IRIS_CENTER)
        else:
            # Fallback to midpoint of canthi
            l_in = get_norm(LEFT_INNER_CANTHUS)
            l_out = get_norm(LEFT_OUTER_CANTHUS)
            r_in = get_norm(RIGHT_INNER_CANTHUS)
            r_out = get_norm(RIGHT_OUTER_CANTHUS)
            left_iris_norm = ((l_in[0] + l_out[0]) / 2.0, (l_in[1] + l_out[1]) / 2.0)
            right_iris_norm = ((r_in[0] + r_out[0]) / 2.0, (r_in[1] + r_out[1]) / 2.0)

        left_iris_px = norm_to_px(left_iris_norm)
        right_iris_px = norm_to_px(right_iris_norm)

        # Canthi & Eyelids
        left_inner_canthus_norm = get_norm(LEFT_INNER_CANTHUS)
        left_outer_canthus_norm = get_norm(LEFT_OUTER_CANTHUS)
        right_inner_canthus_norm = get_norm(RIGHT_INNER_CANTHUS)
        right_outer_canthus_norm = get_norm(RIGHT_OUTER_CANTHUS)

        left_eyelid_top_norm = get_norm(LEFT_TOP_EYELID)
        left_eyelid_bot_norm = get_norm(LEFT_BOT_EYELID)
        right_eyelid_top_norm = get_norm(RIGHT_TOP_EYELID)
        right_eyelid_bot_norm = get_norm(RIGHT_BOT_EYELID)

        # EAR computation
        left_ear = self.compute_ear_from_landmarks(
            landmarks, (LEFT_INNER_CANTHUS, LEFT_OUTER_CANTHUS),
            LEFT_VERTICAL_1, LEFT_VERTICAL_2, frame_w, frame_h
        )
        right_ear = self.compute_ear_from_landmarks(
            landmarks, (RIGHT_INNER_CANTHUS, RIGHT_OUTER_CANTHUS),
            RIGHT_VERTICAL_1, RIGHT_VERTICAL_2, frame_w, frame_h
        )

        # Eye Widths in pixels
        l_in_px = norm_to_px(left_inner_canthus_norm)
        l_out_px = norm_to_px(left_outer_canthus_norm)
        r_in_px = norm_to_px(right_inner_canthus_norm)
        r_out_px = norm_to_px(right_outer_canthus_norm)

        left_eye_width_px = float(math.hypot(l_out_px[0] - l_in_px[0], l_out_px[1] - l_in_px[1]))
        right_eye_width_px = float(math.hypot(r_out_px[0] - r_in_px[0], r_out_px[1] - r_in_px[1]))

        # Interpupillary distance (IPD)
        ipd_px = float(math.hypot(right_iris_px[0] - left_iris_px[0], right_iris_px[1] - left_iris_px[1]))
        ipd_norm = float(math.hypot(right_iris_norm[0] - left_iris_norm[0], right_iris_norm[1] - left_iris_norm[1]))

        # Estimated physical distance (cm) via D = 4095 / irisDistancePx (insightEye calibration formula)
        if ipd_px > 1.0:
            estimated_distance_cm = float(self.ref_distance_constant / ipd_px)
        else:
            estimated_distance_cm = 50.0

        # Head Pose
        head_pose = self.estimate_head_pose_from_landmarks(landmarks, frame_w, frame_h)

        # Face Bounding Box
        all_xs = [lm.x if hasattr(lm, "x") else lm[0] for lm in landmarks]
        all_ys = [lm.y if hasattr(lm, "y") else lm[1] for lm in landmarks]
        face_bbox = (
            float(max(0.0, min(all_xs))),
            float(max(0.0, min(all_ys))),
            float(min(1.0, max(all_xs))),
            float(min(1.0, max(all_ys))),
        )

        # Contour pixel points
        left_contour_px = [norm_to_px(get_norm(i)) for i in LEFT_EYE_CONTOUR]
        right_contour_px = [norm_to_px(get_norm(i)) for i in RIGHT_EYE_CONTOUR]

        # Basic validity checks
        left_valid = (left_eye_width_px >= 10.0) and (left_ear >= 0.05)
        right_valid = (right_eye_width_px >= 10.0) and (right_ear >= 0.05)

        return EyeLandmarksData(
            left_iris_norm=left_iris_norm,
            right_iris_norm=right_iris_norm,
            left_iris_px=left_iris_px,
            right_iris_px=right_iris_px,
            left_inner_canthus_norm=left_inner_canthus_norm,
            left_outer_canthus_norm=left_outer_canthus_norm,
            right_inner_canthus_norm=right_inner_canthus_norm,
            right_outer_canthus_norm=right_outer_canthus_norm,
            left_eyelid_top_norm=left_eyelid_top_norm,
            left_eyelid_bot_norm=left_eyelid_bot_norm,
            right_eyelid_top_norm=right_eyelid_top_norm,
            right_eyelid_bot_norm=right_eyelid_bot_norm,
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            left_eye_width_px=round(left_eye_width_px, 2),
            right_eye_width_px=round(right_eye_width_px, 2),
            ipd_px=round(ipd_px, 2),
            ipd_norm=round(ipd_norm, 4),
            estimated_distance_cm=round(estimated_distance_cm, 1),
            head_pose=head_pose,
            face_bbox=face_bbox,
            left_eye_valid=left_valid,
            right_eye_valid=right_valid,
            confidence=confidence,
            left_contour_px=left_contour_px,
            right_contour_px=right_contour_px,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[EyeLandmarksData]:
        """Processes a BGR/RGB image array and extracts eye landmarks if a face is detected."""
        self._init_mesh()
        if self._mesh is None:
            return None

        h, w = frame.shape[:2]
        # MediaPipe expects RGB format
        if cv2 is not None and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        results = self._mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        return self.extract_from_landmarks(landmarks, (h, w), confidence=0.98)
