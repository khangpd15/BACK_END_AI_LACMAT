"""Adapter service for the Korean Strabismus Eye-Tracking Dataset.

Source Repository: https://github.com/hyunwoongko/strabismus-recognition
Classes: Strictly binary:
    - NORMAL (from test_normal/* or normal/*) -> label 0
    - STRABISMUS (from test_strabismus/* or strabismus folders) -> label 1
NO third/sub classes (e.g. EXOTROPIA, ESOTROPIA, HYPERTROPIA) are created.

Handles:
- Recursive discovery and parsing of Korean Eye-Tracker CSVs.
- Precise column mapping:
    TIME -> t
    LPCX -> leftX, LPCY -> leftY, LPV -> leftValid
    RPCX -> rightX, RPCY -> rightY, RPV -> rightValid
- Preserves raw values: no 0 substitution, no sentinel 0.5, no fabricated phases (phase = 'UNKNOWN').
- Retains original timestamp and calculates sampling statistics (estimatedHz, meanIntervalMs, medianIntervalMs, durationMs).
- Technical feature extraction without unverified clinical assumptions.
"""

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("remicare.korean_adapter")

# Required columns in Korean gaze CSVs
REQUIRED_COLUMNS = ["LPCX", "LPCY", "LPV", "RPCX", "RPCY", "RPV"]


@dataclass
class SamplingStats:
    estimatedHz: float
    meanIntervalMs: float
    medianIntervalMs: float
    durationMs: float


@dataclass
class NormalizedSample:
    index: int
    t: float
    phase: str
    leftX: Optional[float]
    leftY: Optional[float]
    leftValid: bool
    rightX: Optional[float]
    rightY: Optional[float]
    rightValid: bool


@dataclass
class NormalizedKoreanRecord:
    sampleId: str
    source: str
    deviceType: str
    label: str
    sampling: Dict[str, float]
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampleId": self.sampleId,
            "source": self.source,
            "deviceType": self.deviceType,
            "label": self.label,
            "sampling": self.sampling,
            "samples": self.samples,
        }


def resolve_label_from_folder(file_path: str) -> Optional[str]:
    """Map directory path strictly to NORMAL or STRABISMUS.
    
    Rule:
    - Containing 'test_normal' or 'normal' -> 'NORMAL'
    - Containing 'test_strabismus' or 'strabismus' or 'tropia' -> 'STRABISMUS'
    - Never generates sub-types (EXOTROPIA, ESOTROPIA, HYPERTROPIA).
    """
    path_norm = os.path.normpath(file_path).replace("\\", "/")
    parts = path_norm.split("/")
    
    for part in parts:
        part_lower = part.lower()
        if part_lower in ("test_normal", "normal"):
            return "NORMAL"
        if part_lower in ("test_strabismus", "strabismus", "esotropia", "exotropia", "hypertropia"):
            return "STRABISMUS"
            
    return None


def find_time_column(df_columns: List[str]) -> Optional[str]:
    """Find the timestamp column which may appear as 'TIME' or 'TIME(YYYY/MM/DD ...)'."""
    for col in df_columns:
        if col == "TIME" or col.startswith("TIME("):
            return col
    return None


def calculate_sampling_stats(ts: np.ndarray) -> SamplingStats:
    """Compute sampling telemetry from raw timestamps in seconds."""
    if len(ts) < 2:
        return SamplingStats(
            estimatedHz=0.0,
            meanIntervalMs=0.0,
            medianIntervalMs=0.0,
            durationMs=0.0,
        )

    # Convert seconds to milliseconds
    diffs_ms = np.diff(ts) * 1000.0
    positive_diffs = diffs_ms[diffs_ms > 0]

    if len(positive_diffs) == 0:
        mean_int = 0.0
        med_int = 0.0
        hz = 0.0
    else:
        mean_int = float(np.mean(positive_diffs))
        med_int = float(np.median(positive_diffs))
        hz = float(1000.0 / med_int) if med_int > 0 else 0.0

    duration_ms = float((ts[-1] - ts[0]) * 1000.0)

    return SamplingStats(
        estimatedHz=round(hz, 2),
        meanIntervalMs=round(mean_int, 4),
        medianIntervalMs=round(med_int, 4),
        durationMs=round(duration_ms, 2),
    )


def parse_korean_csv(file_path: str, label: Optional[str] = None) -> Tuple[Optional[NormalizedKoreanRecord], List[str]]:
    """Parse a single Korean gaze CSV file into a NormalizedKoreanRecord.
    
    Preserves exact coordinates without replacing missing values with 0 or 0.5.
    Sets phase to 'UNKNOWN' since Korean dataset lacks phase tags.
    """
    errors: List[str] = []

    if not os.path.exists(file_path):
        return None, [f"File not found: {file_path}"]

    if label is None:
        label = resolve_label_from_folder(file_path)

    if label not in ("NORMAL", "STRABISMUS"):
        return None, [f"Could not resolve binary label from path: {file_path}"]

    try:
        df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        return None, [f"Failed to read CSV {file_path}: {e}"]

    # Validate required columns
    cols = df.columns.tolist()
    time_col = find_time_column(cols)

    missing_cols = []
    if not time_col:
        missing_cols.append("TIME")
    for req in REQUIRED_COLUMNS:
        if req not in cols:
            missing_cols.append(req)

    if missing_cols:
        return None, [f"Missing required columns in {file_path}: {missing_cols}"]

    # Filename-based unique sample identifier
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    safe_name = re.sub(r"[^\w\-_]", "_", base_name)
    sample_id = f"korean_{label.lower()}_{safe_name}"

    # Extract timestamps
    raw_ts = df[time_col].to_numpy(dtype=float)
    sampling_stats = calculate_sampling_stats(raw_ts)

    samples: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        # Timestamp
        t_val = float(row[time_col])

        # Left Eye
        left_valid = bool(int(row["LPV"]) == 1) if pd.notna(row["LPV"]) else False
        lx_val = row["LPCX"]
        ly_val = row["LPCY"]
        left_x = float(lx_val) if pd.notna(lx_val) and math.isfinite(float(lx_val)) else None
        left_y = float(ly_val) if pd.notna(ly_val) and math.isfinite(float(ly_val)) else None

        # Right Eye
        right_valid = bool(int(row["RPV"]) == 1) if pd.notna(row["RPV"]) else False
        rx_val = row["RPCX"]
        ry_val = row["RPCY"]
        right_x = float(rx_val) if pd.notna(rx_val) and math.isfinite(float(rx_val)) else None
        right_y = float(ry_val) if pd.notna(ry_val) and math.isfinite(float(ry_val)) else None

        sample_obj = {
            "index": int(idx),
            "t": t_val,
            "phase": "UNKNOWN",
            "leftX": left_x,
            "leftY": left_y,
            "leftValid": left_valid,
            "rightX": right_x,
            "rightY": right_y,
            "rightValid": right_valid,
        }
        samples.append(sample_obj)

    record = NormalizedKoreanRecord(
        sampleId=sample_id,
        source="KOREAN_EYE_TRACKER",
        deviceType="INFRARED_EYE_TRACKER",
        label=label,
        sampling=asdict(sampling_stats),
        samples=samples,
        metadata={
            "originalFile": file_path,
            "rowCount": len(df),
        },
    )

    return record, errors


def extract_features_from_korean_record(record: NormalizedKoreanRecord) -> Dict[str, Any]:
    """Extract verifiable technical features from the normalized Korean record.
    
    Only computes features supported by empirical data:
    - Inter-ocular pupil disparity (deltaX = rightX - leftX, deltaY = rightY - leftY)
    - Coordinate statistics
    - Movement velocity dynamics
    - Eye detection validity ratios
    - Sampling intervals
    Does NOT fabricate baseline displacement or cover-test phases.
    """
    samples = record.samples
    total_samples = len(samples)

    if total_samples == 0:
        return {
            "sampleId": record.sampleId,
            "label_name": record.label,
            "label": 1 if record.label == "STRABISMUS" else 0,
        }

    left_valid_samples = [s for s in samples if s["leftValid"] and s["leftX"] is not None and s["leftY"] is not None]
    right_valid_samples = [s for s in samples if s["rightValid"] and s["rightX"] is not None and s["rightY"] is not None]
    both_valid = [
        s for s in samples
        if s["leftValid"] and s["rightValid"]
        and s["leftX"] is not None and s["rightX"] is not None
        and s["leftY"] is not None and s["rightY"] is not None
    ]

    both_valid_count = len(both_valid)
    left_valid_count = len(left_valid_samples)
    right_valid_count = len(right_valid_samples)

    # Inter-ocular disparity (right - left)
    if both_valid_count > 0:
        dxs = np.array([s["rightX"] - s["leftX"] for s in both_valid], dtype=float)
        dys = np.array([s["rightY"] - s["leftY"] for s in both_valid], dtype=float)

        mean_dx = float(np.mean(dxs))
        median_dx = float(np.median(dxs))
        std_dx = float(np.std(dxs))
        min_dx = float(np.min(dxs))
        max_dx = float(np.max(dxs))
        range_dx = float(max_dx - min_dx)
        mean_abs_dx = float(np.mean(np.abs(dxs)))

        mean_dy = float(np.mean(dys))
        median_dy = float(np.median(dys))
        std_dy = float(np.std(dys))
        min_dy = float(np.min(dys))
        max_dy = float(np.max(dys))
        range_dy = float(max_dy - min_dy)
        mean_abs_dy = float(np.mean(np.abs(dys)))
    else:
        mean_dx = median_dx = std_dx = min_dx = max_dx = range_dx = mean_abs_dx = 0.0
        mean_dy = median_dy = std_dy = min_dy = max_dy = range_dy = mean_abs_dy = 0.0

    # Coordinate positions for individual eyes
    if left_valid_count > 0:
        l_xs = np.array([s["leftX"] for s in left_valid_samples], dtype=float)
        l_ys = np.array([s["leftY"] for s in left_valid_samples], dtype=float)
        mean_l_x = float(np.mean(l_xs))
        std_l_x = float(np.std(l_xs))
        mean_l_y = float(np.mean(l_ys))
        std_l_y = float(np.std(l_ys))
    else:
        mean_l_x = std_l_x = mean_l_y = std_l_y = 0.0

    if right_valid_count > 0:
        r_xs = np.array([s["rightX"] for s in right_valid_samples], dtype=float)
        r_ys = np.array([s["rightY"] for s in right_valid_samples], dtype=float)
        mean_r_x = float(np.mean(r_xs))
        std_r_x = float(np.std(r_xs))
        mean_r_y = float(np.mean(r_ys))
        std_r_y = float(np.std(r_ys))
    else:
        mean_r_x = std_r_x = mean_r_y = std_r_y = 0.0

    # Velocities (displacements over dt in seconds)
    l_velocities: List[float] = []
    for i in range(len(left_valid_samples) - 1):
        dt = left_valid_samples[i + 1]["t"] - left_valid_samples[i]["t"]
        if dt > 0.001:
            dist = math.hypot(
                left_valid_samples[i + 1]["leftX"] - left_valid_samples[i]["leftX"],
                left_valid_samples[i + 1]["leftY"] - left_valid_samples[i]["leftY"],
            )
            l_velocities.append(dist / dt)

    r_velocities: List[float] = []
    for i in range(len(right_valid_samples) - 1):
        dt = right_valid_samples[i + 1]["t"] - right_valid_samples[i]["t"]
        if dt > 0.001:
            dist = math.hypot(
                right_valid_samples[i + 1]["rightX"] - right_valid_samples[i]["rightX"],
                right_valid_samples[i + 1]["rightY"] - right_valid_samples[i]["rightY"],
            )
            r_velocities.append(dist / dt)

    mean_l_vel = float(np.mean(l_velocities)) if l_velocities else 0.0
    peak_l_vel = float(np.max(l_velocities)) if l_velocities else 0.0
    mean_r_vel = float(np.mean(r_velocities)) if r_velocities else 0.0
    peak_r_vel = float(np.max(r_velocities)) if r_velocities else 0.0

    binary_label = 1 if record.label == "STRABISMUS" else 0

    return {
        "sampleId": record.sampleId,
        "label_name": record.label,
        "label": binary_label,
        # Sampling & validity
        "sampleCount": total_samples,
        "leftValidRatio": round(left_valid_count / total_samples, 4),
        "rightValidRatio": round(right_valid_count / total_samples, 4),
        "bothValidRatio": round(both_valid_count / total_samples, 4),
        "estimatedHz": record.sampling.get("estimatedHz", 0.0),
        "durationMs": record.sampling.get("durationMs", 0.0),
        "meanIntervalMs": record.sampling.get("meanIntervalMs", 0.0),
        # Inter-ocular disparity (right - left)
        "meanDeltaX": round(mean_dx, 6),
        "medianDeltaX": round(median_dx, 6),
        "stdDeltaX": round(std_dx, 6),
        "minDeltaX": round(min_dx, 6),
        "maxDeltaX": round(max_dx, 6),
        "rangeDeltaX": round(range_dx, 6),
        "meanAbsDeltaX": round(mean_abs_dx, 6),
        "meanDeltaY": round(mean_dy, 6),
        "medianDeltaY": round(median_dy, 6),
        "stdDeltaY": round(std_dy, 6),
        "minDeltaY": round(min_dy, 6),
        "maxDeltaY": round(max_dy, 6),
        "rangeDeltaY": round(range_dy, 6),
        "meanAbsDeltaY": round(mean_abs_dy, 6),
        # Eye coordinates
        "meanLeftX": round(mean_l_x, 6),
        "stdLeftX": round(std_l_x, 6),
        "meanLeftY": round(mean_l_y, 6),
        "stdLeftY": round(std_l_y, 6),
        "meanRightX": round(mean_r_x, 6),
        "stdRightX": round(std_r_x, 6),
        "meanRightY": round(mean_r_y, 6),
        "stdRightY": round(std_r_y, 6),
        # Dynamics
        "meanLeftVelocity": round(mean_l_vel, 6),
        "peakLeftVelocity": round(peak_l_vel, 6),
        "meanRightVelocity": round(mean_r_vel, 6),
        "peakRightVelocity": round(peak_r_vel, 6),
        "velocityDisparity": round(abs(mean_r_vel - mean_l_vel), 6),
    }


def convert_korean_test_folders(
    normal_dir: str,
    strabismus_dir: str,
    normalized_out_dir: str,
    features_out_csv: str,
    report_out_json: str,
) -> Dict[str, Any]:
    """Execute complete batch conversion of test_normal and test_strabismus directories."""
    normal_path = Path(normal_dir)
    strabismus_path = Path(strabismus_dir)

    normal_files = sorted(list(normal_path.rglob("*.csv")))
    strabismus_files = sorted(list(strabismus_path.rglob("*.csv")))

    total_files = len(normal_files) + len(strabismus_files)
    successful_files = 0
    failed_files = 0
    normal_count = 0
    strabismus_count = 0
    total_samples = 0
    invalid_rows = 0
    missing_columns = 0
    errors: List[str] = []

    features_list: List[Dict[str, Any]] = []

    # Ensure output directories exist
    norm_normal_dir = Path(normalized_out_dir) / "NORMAL"
    norm_strabismus_dir = Path(normalized_out_dir) / "STRABISMUS"
    norm_normal_dir.mkdir(parents=True, exist_ok=True)
    norm_strabismus_dir.mkdir(parents=True, exist_ok=True)

    all_file_tasks = [(f, "NORMAL") for f in normal_files] + [(f, "STRABISMUS") for f in strabismus_files]

    for f_path, label in all_file_tasks:
        record, errs = parse_korean_csv(str(f_path), label=label)
        if errs or record is None:
            failed_files += 1
            errors.extend(errs)
            if any("Missing required columns" in e for e in errs):
                missing_columns += 1
            continue

        successful_files += 1
        if label == "NORMAL":
            normal_count += 1
            out_target = norm_normal_dir / f"{record.sampleId}.json"
        else:
            strabismus_count += 1
            out_target = norm_strabismus_dir / f"{record.sampleId}.json"

        # Save normalized JSON
        with open(out_target, "w", encoding="utf-8") as fp:
            json.dump(record.to_dict(), fp, indent=2)

        # Track sample statistics
        total_samples += len(record.samples)
        for s in record.samples:
            if not s["leftValid"] and not s["rightValid"]:
                invalid_rows += 1

        # Extract features
        feats = extract_features_from_korean_record(record)
        features_list.append(feats)

    # Save features CSV
    if features_list:
        df_features = pd.DataFrame(features_list)
        Path(features_out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(features_out_csv, index=False, encoding="utf-8")

    report = {
        "totalFiles": total_files,
        "successfulFiles": successful_files,
        "failedFiles": failed_files,
        "normalFiles": normal_count,
        "strabismusFiles": strabismus_count,
        "totalSamples": total_samples,
        "invalidRows": invalid_rows,
        "missingColumns": missing_columns,
        "errors": errors,
    }

    # Save conversion report
    Path(report_out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_json, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report
