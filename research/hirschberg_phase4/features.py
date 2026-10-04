"""Dimensionless A0 features; never invent missing glints or physical scale."""

import math
import numpy as np

FEATURE_NAMES = (
    "OD_nasal", "OD_vertical", "OS_nasal", "OS_vertical",
    "binocular_camera_horizontal", "binocular_vertical",
)


def extract(row):
    vectors = []
    for eye in ("OD", "OS"):
        points = row["landmarks"][eye]
        center, glint = points["limbus"], points["glint"]
        diameter = points.get("limbus_diameter_px")
        if center is None or glint is None or diameter is None:
            return None, "MISSING_GEOMETRY"
        vectors.append((np.asarray(glint) - np.asarray(center)) / diameter)
    od, os = vectors
    # Unmirrored frontal camera: OD nasal points right, OS nasal points left.
    values = [od[0], od[1], -os[0], os[1], od[0] - os[0], od[1] - os[1]]
    return np.asarray(values, dtype=float), None


def displacement_mm(center, glint, diameter_px, measured_wtw_mm=None):
    if measured_wtw_mm is None:
        return None
    if not all(math.isfinite(v) and v > 0 for v in (diameter_px, measured_wtw_mm)):
        raise ValueError("Physical scale requires valid measured WTW and diameter")
    if center is None or glint is None:
        return None
    return ((np.asarray(glint) - np.asarray(center)) * measured_wtw_mm / diameter_px).tolist()


def to_native(point, inverse_transform):
    transformed = np.asarray(inverse_transform, dtype=float) @ np.array([*point, 1.0])
    if not np.isfinite(transformed).all() or transformed[2] == 0:
        raise ValueError("Invalid coordinate transform")
    return (transformed[:2] / transformed[2]).tolist()


def canthal_distance(medial_od, medial_os):
    if medial_od is None or medial_os is None:
        return None
    return float(np.linalg.norm(np.asarray(medial_od) - np.asarray(medial_os)))
