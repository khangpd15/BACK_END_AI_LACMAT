"""Preprocessing service for RemiCare Strabismus AI.

Constructs an immutable processed representation of Cover Test time-series:
- Computes baseline coordinates using median of valid BASELINE phase samples.
- Calculates relative coordinates: relativeX = rightX - leftX, relativeY = rightY - leftY.
- Computes signed displacements (signedDx, signedDy) and absolute values (absDx, absDy).
- Preserves raw samples untouched without mutation.
- Strictly avoids premature or uncalibrated clinical transformations (e.g. prism diopters).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from app.schemas import CoverCycle, EyeSample, ScreeningRequest


@dataclass(frozen=True)
class ProcessedSample:
    """Preprocessed time-series sample preserving raw values and adding technical metrics."""
    index: int
    t: float
    phase: str

    # Raw preserved values
    leftX: Optional[float]
    leftY: Optional[float]
    leftValid: bool
    rightX: Optional[float]
    rightY: Optional[float]
    rightValid: bool
    trackingQuality: float

    # Relative coordinates (inter-ocular distance/displacement vectors)
    relativeX: Optional[float]
    relativeY: Optional[float]

    # Tracked eye specific coordinates
    trackedEyeX: Optional[float]
    trackedEyeY: Optional[float]
    trackedEyeValid: bool

    # Displacements from baseline (signed and absolute)
    signedDx: Optional[float]
    signedDy: Optional[float]
    absDx: Optional[float]
    absDy: Optional[float]

    # Relative displacement from baseline relative coordinate
    relativeSignedDx: Optional[float]
    relativeSignedDy: Optional[float]


@dataclass(frozen=True)
class CycleBaseline:
    """Baseline reference statistics calculated strictly using median of valid BASELINE phase."""
    baselineLeftX: Optional[float]
    baselineLeftY: Optional[float]
    baselineRightX: Optional[float]
    baselineRightY: Optional[float]
    baselineTrackedX: float
    baselineTrackedY: float
    baselineStdX: float
    baselineStdY: float
    baselineRelativeX: Optional[float]
    baselineRelativeY: Optional[float]
    validBaselineCount: int


@dataclass(frozen=True)
class ProcessedCycle:
    """Preprocessed cycle with baseline and enriched samples."""
    cycle: int
    coveredEye: str
    trackedEye: str
    baseline: CycleBaseline
    samples: List[ProcessedSample]


@dataclass(frozen=True)
class ProcessedData:
    """Complete preprocessed representation of a Cover Test screening request."""
    sampleId: str
    test: str
    cycles: List[ProcessedCycle]


def compute_cycle_baseline(cycle: CoverCycle) -> CycleBaseline:
    """Compute baseline coordinates using median of valid samples during BASELINE phase.
    
    Median is robust against single-frame eye blinks, tracking jitter, or head movement.
    """
    baseline_samples = [s for s in cycle.samples if s.phase == "BASELINE"]
    tracked_eye = cycle.trackedEye

    left_xs = [s.leftX for s in baseline_samples if s.leftValid and s.leftX is not None]
    left_ys = [s.leftY for s in baseline_samples if s.leftValid and s.leftY is not None]
    right_xs = [s.rightX for s in baseline_samples if s.rightValid and s.rightX is not None]
    right_ys = [s.rightY for s in baseline_samples if s.rightValid and s.rightY is not None]

    # Collect tracked eye coordinates
    if tracked_eye == "LEFT":
        tracked_xs = left_xs
        tracked_ys = left_ys
    else:
        tracked_xs = right_xs
        tracked_ys = right_ys

    if len(tracked_xs) == 0 or len(tracked_ys) == 0:
        raise ValueError(f"Cycle {cycle.cycle}: No valid tracked eye baseline samples")

    baseline_tracked_x = float(np.median(tracked_xs))
    baseline_tracked_y = float(np.median(tracked_ys))
    baseline_std_x = float(np.std(tracked_xs, ddof=0)) if len(tracked_xs) > 1 else 0.0
    baseline_std_y = float(np.std(tracked_ys, ddof=0)) if len(tracked_ys) > 1 else 0.0

    baseline_left_x = float(np.median(left_xs)) if len(left_xs) > 0 else None
    baseline_left_y = float(np.median(left_ys)) if len(left_ys) > 0 else None
    baseline_right_x = float(np.median(right_xs)) if len(right_xs) > 0 else None
    baseline_right_y = float(np.median(right_ys)) if len(right_ys) > 0 else None

    # Compute relative baseline if both eyes have valid samples
    rel_xs = [
        s.rightX - s.leftX
        for s in baseline_samples
        if s.leftValid and s.rightValid and s.leftX is not None and s.rightX is not None
    ]
    rel_ys = [
        s.rightY - s.leftY
        for s in baseline_samples
        if s.leftValid and s.rightValid and s.leftY is not None and s.rightY is not None
    ]

    baseline_rel_x = float(np.median(rel_xs)) if len(rel_xs) > 0 else None
    baseline_rel_y = float(np.median(rel_ys)) if len(rel_ys) > 0 else None

    return CycleBaseline(
        baselineLeftX=baseline_left_x,
        baselineLeftY=baseline_left_y,
        baselineRightX=baseline_right_x,
        baselineRightY=baseline_right_y,
        baselineTrackedX=baseline_tracked_x,
        baselineTrackedY=baseline_tracked_y,
        baselineStdX=baseline_std_x,
        baselineStdY=baseline_std_y,
        baselineRelativeX=baseline_rel_x,
        baselineRelativeY=baseline_rel_y,
        validBaselineCount=len(tracked_xs),
    )


def preprocess_cycle(cycle: CoverCycle) -> ProcessedCycle:
    """Preprocess a single CoverCycle without altering raw input."""
    baseline = compute_cycle_baseline(cycle)
    tracked_eye = cycle.trackedEye

    processed_samples: List[ProcessedSample] = []

    for raw_s in cycle.samples:
        # Determine tracked eye coordinate and validity
        if tracked_eye == "LEFT":
            t_x = raw_s.leftX
            t_y = raw_s.leftY
            t_valid = bool(raw_s.leftValid and t_x is not None and t_y is not None)
        else:
            t_x = raw_s.rightX
            t_y = raw_s.rightY
            t_valid = bool(raw_s.rightValid and t_x is not None and t_y is not None)

        # Relative inter-ocular coordinates: right - left
        if (
            raw_s.leftValid
            and raw_s.rightValid
            and raw_s.leftX is not None
            and raw_s.rightX is not None
            and raw_s.leftY is not None
            and raw_s.rightY is not None
        ):
            rel_x = raw_s.rightX - raw_s.leftX
            rel_y = raw_s.rightY - raw_s.leftY
        else:
            rel_x = None
            rel_y = None

        # Signed & absolute displacement from baseline
        if t_valid and t_x is not None and t_y is not None:
            signed_dx = t_x - baseline.baselineTrackedX
            signed_dy = t_y - baseline.baselineTrackedY
            abs_dx = abs(signed_dx)
            abs_dy = abs(signed_dy)
        else:
            signed_dx = None
            signed_dy = None
            abs_dx = None
            abs_dy = None

        # Relative signed displacements
        if rel_x is not None and baseline.baselineRelativeX is not None:
            rel_signed_dx = rel_x - baseline.baselineRelativeX
        else:
            rel_signed_dx = None

        if rel_y is not None and baseline.baselineRelativeY is not None:
            rel_signed_dy = rel_y - baseline.baselineRelativeY
        else:
            rel_signed_dy = None

        processed_s = ProcessedSample(
            index=raw_s.index,
            t=raw_s.t,
            phase=raw_s.phase,
            leftX=raw_s.leftX,
            leftY=raw_s.leftY,
            leftValid=raw_s.leftValid,
            rightX=raw_s.rightX,
            rightY=raw_s.rightY,
            rightValid=raw_s.rightValid,
            trackingQuality=raw_s.trackingQuality,
            relativeX=rel_x,
            relativeY=rel_y,
            trackedEyeX=t_x,
            trackedEyeY=t_y,
            trackedEyeValid=t_valid,
            signedDx=signed_dx,
            signedDy=signed_dy,
            absDx=abs_dx,
            absDy=abs_dy,
            relativeSignedDx=rel_signed_dx,
            relativeSignedDy=rel_signed_dy,
        )
        processed_samples.append(processed_s)

    return ProcessedCycle(
        cycle=cycle.cycle,
        coveredEye=cycle.coveredEye,
        trackedEye=cycle.trackedEye,
        baseline=baseline,
        samples=processed_samples,
    )


def preprocess_screening_request(request: ScreeningRequest) -> ProcessedData:
    """Preprocess all cycles in a ScreeningRequest into ProcessedData."""
    processed_cycles = [preprocess_cycle(c) for c in request.cycles]
    return ProcessedData(
        sampleId=request.sampleId,
        test=request.test,
        cycles=processed_cycles,
    )
