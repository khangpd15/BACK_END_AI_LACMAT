"""One Euro Filter for 2D ocular time-series trajectory smoothing.

Designed specifically for eye tracking during the Cover Test:
- Minimizes jitter during steady fixation periods (f_cutoff -> min_cutoff)
- Reduces phase delay/lag during fast refixation saccades (f_cutoff increases dynamically with velocity)
- Preserves peak displacement and saccadic velocity without flattening signal peaks.

Reference: Casiez, Roussel, & Vogel (CHI 2012)
"""

import math
from typing import Optional, Tuple


class LowPassFilter:
    """First-order low-pass filter (Exponential Smoothing Filter)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def filter(self, value: float) -> float:
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self) -> None:
        self.s = None


class OneEuroFilter1D:
    """1-Dimensional One Euro Filter for a single spatial coordinate."""

    def __init__(
        self,
        min_cutoff: float = 1.0,  # Minimum cutoff frequency (Hz) at rest
        beta: float = 0.007,      # Speed coefficient: adapts cutoff to velocity
        d_cutoff: float = 1.0,    # Cutoff frequency for derivative calculation
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None
        self.last_value: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.0
        return 1.0 / (1.0 + tau / te) if (te + tau) > 0 else 1.0

    def filter(self, x: float, timestamp_sec: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp_sec
            self.last_value = x
            return x

        dt = timestamp_sec - self.last_time
        if dt <= 1e-5:
            return self.last_value if self.last_value is not None else x

        rate = 1.0 / dt

        # Estimate smoothed derivative dx/dt
        dx = (x - self.last_value) * rate if self.last_value is not None else 0.0
        self.dx_filt.alpha = self._alpha(rate, self.d_cutoff)
        edx = self.dx_filt.filter(dx)

        # Dynamic cutoff frequency according to instantaneous velocity
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the position coordinate
        self.x_filt.alpha = self._alpha(rate, cutoff)
        filtered_x = self.x_filt.filter(x)

        self.last_time = timestamp_sec
        self.last_value = filtered_x
        return filtered_x

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None
        self.last_value = None


class EyeTrajectoryOneEuroFilter:
    """Bilateral (Left/Right) 2D trajectory One Euro Filter.
    
    Filters (X, Y) coordinates independently for both eyes while preserving
    timestamps and saccadic velocity transitions.
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.007):
        self.left_x = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.left_y = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.right_x = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.right_y = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)

    def process_frame(
        self,
        t_sec: float,
        left_xy: Optional[Tuple[float, float]],
        right_xy: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Filters eye coordinate tuple at a given timestamp."""
        filt_left = None
        if left_xy and math.isfinite(left_xy[0]) and math.isfinite(left_xy[1]):
            filt_left = (
                self.left_x.filter(left_xy[0], t_sec),
                self.left_y.filter(left_xy[1], t_sec),
            )

        filt_right = None
        if right_xy and math.isfinite(right_xy[0]) and math.isfinite(right_xy[1]):
            filt_right = (
                self.right_x.filter(right_xy[0], t_sec),
                self.right_y.filter(right_xy[1], t_sec),
            )

        return filt_left, filt_right

    def reset(self) -> None:
        self.left_x.reset()
        self.left_y.reset()
        self.right_x.reset()
        self.right_y.reset()
