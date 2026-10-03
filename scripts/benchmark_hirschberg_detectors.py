"""Exploratory benchmark for Phase 6A Hirschberg pupil and reflex detectors.

SAFETY & PROTOCOL NOTICE:
This benchmark measures only detector technical coverage and success rates on cropped 224x224 images.
It deliberately does NOT report sensitivity, specificity, or clinical diagnostic metrics.
Thresholds remain TODO_PILOT. No clinical validation is claimed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.services.research_measurement_service import (
    detect_pupil_in_roi,
    detect_reflexes_in_roi,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("remicare.benchmark_detectors")


def estimate_iris_center_in_crop(image: np.ndarray) -> tuple[float, float, float]:
    """Estimate iris center (cx, cy, diameter) for a 224x224 cropped eye image.

    In standardized 224x224 eye crops, the eye is approximately centered.
    We refine the approximate center using Hough circles or dark center heuristic within central ROI.
    """
    h, w = image.shape[:2]
    default_cx, default_cy = w / 2.0, h / 2.0
    default_diameter = min(w, h) * 0.45  # ~100px for 224x224

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # Search for circular iris boundary in range 25px - 75px radius
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=45,
        param2=24,
        minRadius=25,
        maxRadius=75,
    )

    if circles is not None and len(circles[0]) > 0:
        # Pick the circle closest to the image center
        best_circle = None
        min_dist = float("inf")
        for c in circles[0]:
            dist = np.hypot(c[0] - default_cx, c[1] - default_cy)
            if dist < min_dist:
                min_dist = dist
                best_circle = c

        if best_circle is not None and min_dist < 45.0:
            cx = float(best_circle[0])
            cy = float(best_circle[1])
            diameter = float(best_circle[2] * 2.0)
            return cx, cy, diameter

    # Fallback to central default if circle fit is uncertain
    return default_cx, default_cy, default_diameter


def resolve_image_path(raw_path_str: str) -> Path | None:
    p = Path(raw_path_str)
    if p.is_file():
        return p
    candidate_1 = PROJECT_ROOT / p
    if candidate_1.is_file():
        return candidate_1
    candidate_2 = Path("D:/REMICARE-STRABISMUS-AI") / p
    if candidate_2.is_file():
        return candidate_2
    return None


def run_benchmark(
    manifest_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples and max_samples < len(records):
        records = records[:max_samples]

    total_images = len(records)
    logger.info("Running benchmark on %d images from %s", total_images, manifest_path.name)

    pupil_counter = Counter()
    reflex_counter = Counter()
    reflex_tier_counter = Counter()
    joint_coverage_count = 0

    per_split_stats: Dict[str, Dict[str, int]] = {}
    valid_count = 0

    for idx, rec in enumerate(records, 1):
        raw_path = rec.get("absolute_path") or rec.get("relative_path")
        img_path = resolve_image_path(raw_path)
        if img_path is None:
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid_count += 1

        cx, cy, diameter = estimate_iris_center_in_crop(rgb)
        reflexes, r_status, r_tier = detect_reflexes_in_roi(rgb, cx, cy, diameter)
        pupil_c, pupil_d, p_status = detect_pupil_in_roi(rgb, cx, cy, diameter)

        pupil_counter[p_status] += 1
        reflex_counter[r_status] += 1
        if r_tier:
            reflex_tier_counter[r_tier] += 1

        is_joint = (r_status == "DETECTED") and (p_status == "DETECTED")
        if is_joint:
            joint_coverage_count += 1

        split = rec.get("generated_split", "all")
        if split not in per_split_stats:
            per_split_stats[split] = {"total": 0, "joint": 0, "pupil": 0, "reflex": 0}
        per_split_stats[split]["total"] += 1
        if p_status == "DETECTED":
            per_split_stats[split]["pupil"] += 1
        if r_status == "DETECTED":
            per_split_stats[split]["reflex"] += 1
        if is_joint:
            per_split_stats[split]["joint"] += 1

        if idx % 100 == 0 or idx == total_images:
            logger.info("Processed %d/%d images...", idx, total_images)

    pupil_success_rate = (pupil_counter["DETECTED"] / valid_count) if valid_count else 0.0
    reflex_success_rate = (reflex_counter["DETECTED"] / valid_count) if valid_count else 0.0
    joint_coverage_rate = (joint_coverage_count / valid_count) if valid_count else 0.0

    summary = {
        "benchmarkName": "Phase 6A Hirschberg Detector Technical Coverage Benchmark",
        "datasetVersion": "hirschberg-folder-labels-v0.1",
        "manifestPath": str(manifest_path).replace("\\", "/"),
        "totalImagesEvaluated": valid_count,
        "detectorSuccessRates": {
            "pupilCenterDetection": {
                "detectedCount": pupil_counter["DETECTED"],
                "successRatePercent": round(pupil_success_rate * 100.0, 2),
                "pupilNotFoundCount": pupil_counter["PUPIL_NOT_FOUND"],
                "lowQualityInputCount": pupil_counter["LOW_QUALITY_INPUT"],
            },
            "cornealReflexDetection": {
                "detectedCount": reflex_counter["DETECTED"],
                "successRatePercent": round(reflex_success_rate * 100.0, 2),
                "reflexNotFoundCount": reflex_counter["REFLEX_NOT_FOUND"],
                "multipleReflexCount": reflex_counter["MULTIPLE_REFLEX"],
                "lowQualityInputCount": reflex_counter["LOW_QUALITY_INPUT"],
                "tierBreakdown": dict(reflex_tier_counter),
            },
            "jointMeasurementCoverage": {
                "jointDetectedCount": joint_coverage_count,
                "coveragePercent": round(joint_coverage_rate * 100.0, 2),
            },
        },
        "perSplitCoverage": {
            split: {
                "total": stats["total"],
                "pupilSuccessRatePercent": round(stats["pupil"] / stats["total"] * 100.0, 2) if stats["total"] else 0.0,
                "reflexSuccessRatePercent": round(stats["reflex"] / stats["total"] * 100.0, 2) if stats["total"] else 0.0,
                "jointCoveragePercent": round(stats["joint"] / stats["total"] * 100.0, 2) if stats["total"] else 0.0,
            }
            for split, stats in sorted(per_split_stats.items())
        },
        "safetyAndComplianceDeclarations": [
            "All thresholds are TODO_PILOT and pending doctor/pilot review.",
            "Reports technical detector coverage / success rate ONLY.",
            "NO clinical sensitivity, specificity, accuracy, or diagnostic metrics are calculated.",
            "NO clinical threshold is established.",
            "NO clinical diagnosis is made.",
            "Data is exploratory-only cropped images (not production full-frame protocol).",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Benchmark summary written to %s", output_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Hirschberg pupil & reflex detector coverage.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("D:/AI_Check_Lac/manifests/hirschberg_folder_labels.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("D:/AI_Check_Lac/manifests/hirschberg_detector_benchmark.json"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    summary = run_benchmark(args.manifest, args.output, max_samples=args.max_samples)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
