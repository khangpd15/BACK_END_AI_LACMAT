"""Kinematics & Refixation Event Detector for Cover Test trajectories.

Differentiates true pathological refixation saccades from:
- Natural physiological ocular tremor / drift
- Microsaccades
- Tracking jitter and blink artifacts

Computes high-order kinematic derivatives:
- Velocity v(t) = dr/dt
- Acceleration a(t) = dv/dt
- Jerk j(t) = da/dt
- Settling time and overshoot ratio
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RefixationEvent:
    has_refixation: bool
    confidence: float              # [0.0, 1.0]
    peak_displacement: float       # Peak absolute displacement
    peak_velocity: float           # Max velocity (coords/sec)
    peak_acceleration: float       # Max acceleration (coords/sec^2)
    peak_jerk: float               # Max jerk (coords/sec^3)
    time_to_peak_ms: float         # Latency from uncover to peak displacement
    movement_duration_ms: float    # Duration of active saccade movement
    settling_time_ms: float        # Time required to stabilize after saccade
    overshoot_ratio: float         # Overshoot beyond final settled position
    direction_sign: int            # +1 or -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hasRefixation": self.has_refixation,
            "confidence": self.confidence,
            "peakDisplacement": self.peak_displacement,
            "peakVelocity": self.peak_velocity,
            "peakAcceleration": self.peak_acceleration,
            "peakJerk": self.peak_jerk,
            "timeToPeakMs": self.time_to_peak_ms,
            "movementDurationMs": self.movement_duration_ms,
            "settlingTimeMs": self.settling_time_ms,
            "overshootRatio": self.overshoot_ratio,
            "directionSign": self.direction_sign,
        }


class RefixationEventDetector:
    """Analyzes post-uncover ocular trajectories to detect saccadic refixation events."""

    def __init__(
        self,
        min_saccade_velocity: float = 0.06,        # Threshold for active saccadic movement
        min_displacement_threshold: float = 0.012, # Threshold to exceed physiological micro-jitter
        settling_window_samples: int = 4,
    ):
        self.min_saccade_velocity = min_saccade_velocity
        self.min_displacement_threshold = min_displacement_threshold
        self.settling_window_samples = settling_window_samples

    def analyze_uncover_trajectory(
        self,
        t_ms: np.ndarray,
        displacements: np.ndarray,
    ) -> RefixationEvent:
        """Evaluates 1D horizontal or composite displacement curve.
        
        Args:
            t_ms: Timestamps in milliseconds (monotonic).
            displacements: Signed deviations from baseline.
        """
        n = len(displacements)
        if n < 5 or len(t_ms) != n:
            return RefixationEvent(
                has_refixation=False,
                confidence=0.0,
                peak_displacement=0.0,
                peak_velocity=0.0,
                peak_acceleration=0.0,
                peak_jerk=0.0,
                time_to_peak_ms=0.0,
                movement_duration_ms=0.0,
                settling_time_ms=0.0,
                overshoot_ratio=0.0,
                direction_sign=0,
            )

        # 1. Temporal derivatives via central difference with respect to time axis
        t_sec = np.asarray(t_ms, dtype=float) / 1000.0
        for i in range(1, len(t_sec)):
            if t_sec[i] <= t_sec[i - 1]:
                t_sec[i] = t_sec[i - 1] + 0.0333

        velocities = np.gradient(displacements, t_sec)
        accelerations = np.gradient(velocities, t_sec)
        jerks = np.gradient(accelerations, t_sec)

        abs_disp = np.abs(displacements)
        abs_vel = np.abs(velocities)
        abs_acc = np.abs(accelerations)
        abs_jerk = np.abs(jerks)

        peak_idx = int(np.argmax(abs_disp))
        peak_disp_val = float(abs_disp[peak_idx])
        peak_vel_val = float(np.max(abs_vel))
        peak_acc_val = float(np.max(abs_acc))
        peak_jerk_val = float(np.max(abs_jerk))

        start_t = float(t_ms[0])
        peak_t = float(t_ms[peak_idx])
        time_to_peak = max(0.0, peak_t - start_t)

        # 2. Saccade active duration based on velocity profile
        vel_threshold = self.min_saccade_velocity * 0.45
        active_indices = np.where(abs_vel >= vel_threshold)[0]
        if len(active_indices) > 0:
            duration_ms = max(0.0, float(t_ms[active_indices[-1]] - t_ms[active_indices[0]]))
        else:
            duration_ms = 0.0

        # 3. Settling time & Overshoot
        k_settle = min(self.settling_window_samples, n)
        settled_val = float(np.median(displacements[-k_settle:]))
        overshoot = max(0.0, peak_disp_val - abs(settled_val))
        overshoot_ratio = float(overshoot / peak_disp_val) if peak_disp_val > 1e-4 else 0.0
        settling_time = max(0.0, float(t_ms[-1] - peak_t))

        # 4. Multi-criteria clinical refixation saccade verification
        # - Exceeds micro-drift threshold
        # - Has characteristic rapid saccadic velocity burst
        # - Time-to-peak within physiological saccadic latency window (50ms - 1500ms)
        is_true_saccade = bool(
            peak_disp_val >= self.min_displacement_threshold
            and peak_vel_val >= self.min_saccade_velocity
            and 40.0 <= time_to_peak <= 1500.0
        )

        score_disp = np.clip(peak_disp_val / (self.min_displacement_threshold * 2.5), 0.0, 1.0)
        score_vel = np.clip(peak_vel_val / (self.min_saccade_velocity * 2.0), 0.0, 1.0)
        score_timing = 1.0 if (70.0 <= time_to_peak <= 800.0) else 0.5
        confidence = float(0.45 * score_disp + 0.40 * score_vel + 0.15 * score_timing) if is_true_saccade else 0.0

        direction = int(np.sign(displacements[peak_idx])) if peak_disp_val > 1e-5 else 0

        return RefixationEvent(
            has_refixation=is_true_saccade,
            confidence=round(confidence, 4),
            peak_displacement=round(peak_disp_val, 6),
            peak_velocity=round(peak_vel_val, 6),
            peak_acceleration=round(peak_acc_val, 6),
            peak_jerk=round(peak_jerk_val, 6),
            time_to_peak_ms=round(time_to_peak, 2),
            movement_duration_ms=round(duration_ms, 2),
            settling_time_ms=round(settling_time, 2),
            overshoot_ratio=round(overshoot_ratio, 4),
            direction_sign=direction,
        )
