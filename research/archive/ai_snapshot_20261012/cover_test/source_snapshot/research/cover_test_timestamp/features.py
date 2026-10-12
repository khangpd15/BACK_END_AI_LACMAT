"""Native timestamps in seconds; no interpolation across missing observations."""

from typing import Any

import numpy as np


CONTRACT = "binocular-timestamp-seconds-v1"
TARGET_RATES = (30, 45, 60)
FEATURE_NAMES = [
    f"{target}_{stat}"
    for target in ("left", "right", "disparity")
    for stat in ("spread_x", "spread_y", "radius_p90", "speed_median", "speed_p90", "speed_p99")
] + ["horizontal_speed_correlation", "vertical_speed_correlation"]


def split_clock_segments(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Separate clock resets without assigning phases, patient IDs or new times."""
    if not samples:
        raise ValueError("No timestamped samples")
    times = np.asarray([s["t"] for s in samples], float)
    if not np.isfinite(times).all():
        raise ValueError("Non-finite timestamp")
    boundaries = [0, *(np.flatnonzero(np.diff(times) <= 0) + 1).tolist(), len(times)]
    return [samples[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]


def validate_samples(samples: list[dict[str, Any]], timestamp_unit: str) -> np.ndarray:
    if timestamp_unit != "seconds":
        raise ValueError("This contract requires explicit timestamp_unit='seconds'. Convert milliseconds first.")
    if len(samples) < 2:
        raise ValueError("At least two timestamped samples are required")
    times = np.asarray([s["t"] for s in samples], dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing; no sorting or silent deduplication")
    for sample in samples:
        for eye in ("left", "right"):
            if not isinstance(sample.get(f"{eye}Valid"), bool):
                raise ValueError("Validity flags must be booleans")
            if sample[f"{eye}Valid"]:
                coords = [sample.get(f"{eye}X"), sample.get(f"{eye}Y")]
                if any(v is None for v in coords) or not np.isfinite(np.asarray(coords, float)).all():
                    raise ValueError("A valid eye must have finite X/Y coordinates")
    return times


def sampling_quality(samples: list[dict[str, Any]], timestamp_unit: str = "seconds") -> dict:
    times = validate_samples(samples, timestamp_unit)
    intervals = np.diff(times)
    both = np.array([s["leftValid"] and s["rightValid"] for s in samples], bool)
    phases = sorted({str(s.get("phase", "UNKNOWN")) for s in samples})
    return {
        "sample_count": len(samples),
        "duration_seconds": float(times[-1] - times[0]),
        "median_hz": float(1 / np.median(intervals)),
        "mean_hz": float((len(times) - 1) / (times[-1] - times[0])),
        "gap_over_100ms_count": int((intervals > 0.1).sum()),
        "both_valid_ratio": float(both.mean()),
        "phases": phases,
        "cover_phase_labels_present": {"COVER", "UNCOVER"}.issubset(phases),
    }


def downsample(samples: list[dict[str, Any]], target_hz: int) -> list[dict[str, Any]]:
    if target_hz not in TARGET_RATES:
        raise ValueError("Target rate must be 30, 45 or 60 Hz")
    times = validate_samples(samples, "seconds")
    source_hz = 1 / np.median(np.diff(times))
    if source_hz < target_hz * 0.95:
        raise ValueError("Cannot fabricate a higher frame rate from slower observations")
    # One original observation nearest each grid instant; never invent coordinates.
    grid = np.arange(times[0], times[-1] + 1e-9, 1 / target_hz)
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    chosen = np.where(abs(times[left] - grid) <= abs(times[right] - grid), left, right)
    close = abs(times[chosen] - grid) <= 0.55 / target_hz
    indices = np.unique(chosen[close])
    return [samples[int(i)] for i in indices]


def _motion(coords: np.ndarray, times: np.ndarray, valid: np.ndarray) -> tuple[list[float], np.ndarray]:
    values = coords[valid]
    if len(values) < 30:
        raise ValueError("Fewer than 30 jointly valid observations")
    centered = values - np.median(values, axis=0)
    # Derivatives only join consecutive valid observations separated by <=100ms.
    dt = np.diff(times)
    pairs = valid[:-1] & valid[1:] & (dt <= 0.1)
    if pairs.sum() < 15:
        raise ValueError("Insufficient consecutive observations for timestamp derivatives")
    velocity = np.diff(coords, axis=0)[pairs] / dt[pairs, None]
    speed = np.linalg.norm(velocity, axis=1)
    stats = [
        float(np.quantile(values[:, 0], 0.9) - np.quantile(values[:, 0], 0.1)),
        float(np.quantile(values[:, 1], 0.9) - np.quantile(values[:, 1], 0.1)),
        float(np.quantile(np.linalg.norm(centered, axis=1), 0.9)),
        *[float(np.quantile(speed, q)) for q in (0.5, 0.9, 0.99)],
    ]
    return stats, velocity


def extract_features(samples: list[dict[str, Any]], timestamp_unit: str = "seconds") -> np.ndarray:
    times = validate_samples(samples, timestamp_unit)
    quality = sampling_quality(samples, timestamp_unit)
    if not 27 <= quality["median_hz"] <= 65:
        raise ValueError("Observed sampling rate outside supported 30-60 FPS tolerance (27-65 Hz)")
    if quality["duration_seconds"] < 2 or quality["both_valid_ratio"] < 0.5:
        raise ValueError("Need >=2 seconds and >=50% jointly valid tracking")
    valid = np.asarray([s["leftValid"] and s["rightValid"] for s in samples], bool)
    left = np.full((len(samples), 2), np.nan)
    right = np.full_like(left, np.nan)
    for i, sample in enumerate(samples):
        if valid[i]:
            left[i] = [sample["leftX"], sample["leftY"]]
            right[i] = [sample["rightX"], sample["rightY"]]
    result = []
    velocities = []
    for coords in (left, right, right - left):
        stats, velocity = _motion(coords, times, valid)
        result.extend(stats)
        velocities.append(velocity)
    for axis in range(2):
        a, b = velocities[0][:, axis], velocities[1][:, axis]
        # Correlation of constant signals is undefined; reject instead of imputing.
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            raise ValueError("Velocity correlation undefined for a constant signal")
        result.append(float(np.corrcoef(a, b)[0, 1]))
    vector = np.asarray(result, float)
    if len(vector) != len(FEATURE_NAMES) or not np.isfinite(vector).all():
        raise ValueError("Non-finite or incomplete timestamp feature vector")
    return vector


def predict_recording(bundle: dict, samples: list[dict[str, Any]], *, timestamp_unit: str,
                      source: str) -> dict:
    if bundle.get("contract") != CONTRACT or bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Artifact feature contract mismatch")
    if source != bundle.get("source_domain"):
        return {"status": "INCONCLUSIVE", "reason": "SOURCE_DOMAIN_MISMATCH"}
    try:
        vector = extract_features(samples, timestamp_unit)
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "INCONCLUSIVE", "reason": "INVALID_TRAJECTORY", "detail": str(exc)}
    model = bundle["model"]
    probabilities = model.predict_proba(vector[None])[0]
    return {
        "status": "RESEARCH_PREDICTION",
        "prediction": str(model.classes_[int(np.argmax(probabilities))]),
        "class_probability": dict(zip(model.classes_.tolist(), probabilities.tolist())),
        "probabilities_calibrated": False,
        "model_name": bundle["name"],
        "version": bundle["version"],
        "quality": sampling_quality(samples, timestamp_unit),
        "clinical_validation": False,
        "cover_response_trained": False,
    }
