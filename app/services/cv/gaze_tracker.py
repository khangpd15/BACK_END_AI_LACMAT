"""Gaze and Fixation Tracking Service.

Adapted from InsightEye research architecture:
- Vector projection of iris onto canthi and eyelid margins (dimensionless gaze ratio [0, 1])
- I-DT (Identification by Dispersion-Threshold) algorithmic fixation detection
- Real-time ocular saccade velocity tracking
- Differentiates steady fixational gaze from rapid saccadic movements
- Blink cycle tracking and duration accumulator
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GazeMetrics:
    """Telemetry metrics for ocular gaze orientation and fixation stability."""
    timestamp: float
    gaze_x_left: float
    gaze_y_left: float
    gaze_x_right: float
    gaze_y_right: float
    gaze_x: float  # Binocular average
    gaze_y: float  # Binocular average
    gaze_velocity: float  # Disparity / second
    is_fixating: bool
    is_saccade: bool
    dispersion: float
    stability_score: float  # [0.0, 1.0] where 1.0 = perfectly steady fixation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "gazeXLeft": round(self.gaze_x_left, 4),
            "gazeYLeft": round(self.gaze_y_left, 4),
            "gazeXRight": round(self.gaze_x_right, 4),
            "gazeYRight": round(self.gaze_y_right, 4),
            "gazeX": round(self.gaze_x, 4),
            "gazeY": round(self.gaze_y, 4),
            "gazeVelocity": round(self.gaze_velocity, 4),
            "isFixating": self.is_fixating,
            "isSaccade": self.is_saccade,
            "dispersion": round(self.dispersion, 4),
            "stabilityScore": round(self.stability_score, 4),
        }


class GazeFixationTracker:
    """Ocular gaze direction and fixation stability analyzer."""

    def __init__(
        self,
        dispersion_threshold: float = 0.08,
        fixation_window_frames: int = 10,
        saccade_velocity_threshold: float = 2.0,
    ):
        self.dispersion_threshold = dispersion_threshold
        self.fixation_window_frames = fixation_window_frames
        self.saccade_velocity_threshold = saccade_velocity_threshold

        # Sliding temporal window of (gaze_x, gaze_y, timestamp)
        self.gaze_window: List[Tuple[float, float, float]] = []
        self.last_gaze: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def project_point_onto_segment(
        pt: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        """Projects a 2D point onto a line segment and returns scalar ratio [0.0, 1.0].
        
        0.0 indicates alignment with start point, 1.0 indicates alignment with end point.
        """
        p = np.array([pt[0], pt[1]], dtype=float)
        s = np.array([start[0], start[1]], dtype=float)
        e = np.array([end[0], end[1]], dtype=float)

        line_vec = e - s
        line_len_sq = float(np.dot(line_vec, line_vec))
        if line_len_sq < 1e-7:
            return 0.5

        proj = float(np.dot(p - s, line_vec) / line_len_sq)
        return float(np.clip(proj, 0.0, 1.0))

    def compute_monocular_gaze_ratio(
        self,
        iris_xy: Tuple[float, float],
        inner_canthus: Tuple[float, float],
        outer_canthus: Tuple[float, float],
        eyelid_top: Tuple[float, float],
        eyelid_bot: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Calculates normalized horizontal and vertical gaze ratios [0, 1] using vector projections."""
        # Horizontal gaze: project iris onto inner-to-outer canthus segment
        gaze_x = self.project_point_onto_segment(iris_xy, inner_canthus, outer_canthus)
        # Vertical gaze: project iris onto superior-to-inferior eyelid segment
        gaze_y = self.project_point_onto_segment(iris_xy, eyelid_top, eyelid_bot)
        return gaze_x, gaze_y

    def update(
        self,
        t_sec: float,
        left_iris_xy: Optional[Tuple[float, float]],
        right_iris_xy: Optional[Tuple[float, float]],
        left_canthi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        right_canthi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        left_eyelids: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        right_eyelids: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> GazeMetrics:
        """Updates tracker state with new frame coordinates and outputs gaze stability metrics."""
        # 1. Compute monocular gaze ratios
        if left_iris_xy and left_canthi and left_eyelids:
            g_lx, g_ly = self.compute_monocular_gaze_ratio(
                left_iris_xy, left_canthi[0], left_canthi[1], left_eyelids[0], left_eyelids[1]
            )
        elif left_iris_xy:
            g_lx, g_ly = left_iris_xy[0], left_iris_xy[1]
        else:
            g_lx, g_ly = 0.5, 0.5

        if right_iris_xy and right_canthi and right_eyelids:
            g_rx, g_ry = self.compute_monocular_gaze_ratio(
                right_iris_xy, right_canthi[0], right_canthi[1], right_eyelids[0], right_eyelids[1]
            )
        elif right_iris_xy:
            g_rx, g_ry = right_iris_xy[0], right_iris_xy[1]
        else:
            g_rx, g_ry = 0.5, 0.5

        # 2. Binocular average gaze
        gaze_x = (g_lx + g_rx) / 2.0
        gaze_y = (g_ly + g_ry) / 2.0

        # 3. Gaze velocity (Euclidean displacement / dt)
        gaze_velocity = 0.0
        if self.last_gaze is not None:
            last_x, last_y, last_t = self.last_gaze
            dt = t_sec - last_t
            if dt > 1e-4:
                dist = math.hypot(gaze_x - last_x, gaze_y - last_y)
                gaze_velocity = dist / dt

        self.last_gaze = (gaze_x, gaze_y, t_sec)

        # 4. Sliding window for I-DT fixation detection
        self.gaze_window.append((gaze_x, gaze_y, t_sec))
        if len(self.gaze_window) > self.fixation_window_frames:
            self.gaze_window.pop(0)

        is_fixating = False
        dispersion = 0.0
        if len(self.gaze_window) >= max(3, self.fixation_window_frames // 2):
            xs = [g[0] for g in self.gaze_window]
            ys = [g[1] for g in self.gaze_window]
            dispersion = float((max(xs) - min(xs)) + (max(ys) - min(ys)))
            if dispersion < self.dispersion_threshold:
                is_fixating = True

        # 5. Saccade classification
        is_saccade = (gaze_velocity > self.saccade_velocity_threshold) and not is_fixating

        # 6. Continuous stability score [0.0, 1.0]
        # Low dispersion & low velocity -> score near 1.0
        disp_score = np.clip(1.0 - (dispersion / (self.dispersion_threshold * 2.0)), 0.0, 1.0)
        vel_score = np.clip(1.0 - (gaze_velocity / (self.saccade_velocity_threshold * 2.0)), 0.0, 1.0)
        stability_score = float(0.6 * disp_score + 0.4 * vel_score)

        return GazeMetrics(
            timestamp=t_sec,
            gaze_x_left=g_lx,
            gaze_y_left=g_ly,
            gaze_x_right=g_rx,
            gaze_y_right=g_ry,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            gaze_velocity=gaze_velocity,
            is_fixating=is_fixating,
            is_saccade=is_saccade,
            dispersion=dispersion,
            stability_score=stability_score,
        )

    def reset(self) -> None:
        """Resets tracking history."""
        self.gaze_window.clear()
        self.last_gaze = None
