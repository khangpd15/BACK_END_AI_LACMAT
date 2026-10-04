#!/usr/bin/env python3
"""Prepare Phase A QA artifacts for the Pedseye Hirschberg research dataset.

Outputs a JSONL manifest, validation summary, data-gap report, and detector
overlays. This is research-only and does not modify production endpoints/models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "remicare_matplotlib_cache"))

from app.services.hirschberg_ai_service import estimate_iris_center_in_crop
from app.services.research_measurement_service import detect_pupil_in_roi, detect_reflexes_in_roi

DATASET_DIR = PROJECT_ROOT / "dataset_by_pedseye_manifest"
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "pedseye_hirschberg_manifest_v0.1.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "pedseye_hirschberg_manifest_v0.1_summary.json"
GAP_REPORT_PATH = PROJECT_ROOT / "reports" / "pedseye_data_gap_report.md"
OVERLAY_DIR = PROJECT_ROOT / "reports" / "pedseye_detector_overlays"
PHASH_THRESHOLD = 4
SEED = 20261004

LABELS = ["normal", "esotropia", "exotropia", "pseudostrabismus", "poor_quality"]
TARGET_RESEARCH_MIN = {
    "normal": 300,
    "esotropia": 300,
    "exotropia": 300,
    "pseudostrabismus": 300,
    "poor_quality": 200,
}


class UnionFind:
    def __init__(self, items: Iterable[Path]):
        self.parent = {item: item for item in items}

    def find(self, item: Path) -> Path:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: Path, b: Path) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def infer_label(path: Path) -> Optional[str]:
    folder = path.parent.name.lower()
    name = path.name.lower()
    if "esotropia" in folder or "estropia" in name:
        return "esotropia"
    if "exotropia" in folder:
        return "exotropia"
    if "normal" in folder:
        return "normal"
    if "pseudo" in folder:
        return "pseudostrabismus"
    if "poor" in folder or "reject" in folder:
        return "poor_quality"
    return None


def image_files(dataset_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> Tuple[int, ...]:
    with Image.open(path) as img:
        gray = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        if hasattr(gray, "get_flattened_data"):
            values = list(gray.get_flattened_data())
        else:
            values = list(gray.getdata())
    avg = sum(values) / max(1, len(values))
    return tuple(1 if value >= avg else 0 for value in values)


def hamming(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    return sum(x != y for x, y in zip(a, b))


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def crop_regions_for_service_contract(bgr: np.ndarray) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
    h, w = bgr.shape[:2]
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < 1.35 and min(w, h) <= 400:
        return [("single_crop", cv2.resize(bgr, (224, 224)), (0, 0, w, h))]
    return [
        ("left_half", cv2.resize(bgr[:, : w // 2], (224, 224)), (0, 0, w // 2, h)),
        ("right_half", cv2.resize(bgr[:, w // 2 :], (224, 224)), (w // 2, 0, w, h)),
    ]


def analyze_image_quality(path: Path) -> Dict[str, Any]:
    bgr = read_bgr(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    h, w = bgr.shape[:2]

    crop_results = []
    any_reflex = False
    for name, crop_bgr, _region in crop_regions_for_service_contract(bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        cx, cy, diameter = estimate_iris_center_in_crop(rgb)
        reflexes, reflex_status, reflex_tier = detect_reflexes_in_roi(rgb, cx, cy, diameter)
        pupil, pupil_diameter, pupil_status = detect_pupil_in_roi(rgb, cx, cy, diameter)
        any_reflex = any_reflex or bool(reflexes)
        crop_results.append(
            {
                "crop": name,
                "iris_center_px": [round(float(cx), 2), round(float(cy), 2)],
                "iris_diameter_px": round(float(diameter), 2),
                "reflex_status": reflex_status,
                "reflex_tier": reflex_tier,
                "reflex_count": len(reflexes),
                "pupil_status": pupil_status,
                "pupil_found": pupil is not None,
                "pupil_diameter_px": round(float(pupil_diameter), 2) if pupil_diameter else None,
            }
        )

    return {
        "resolution": {"width": int(w), "height": int(h)},
        "blur_variance": round(blur, 4),
        "mean_brightness": round(brightness, 4),
        "quality_flags": {
            "blur": blur < 100.0,
            "too_dark": brightness < 40.0,
            "eyes_closed": None,
            "face_off_axis": None,
            "no_corneal_reflex": not any_reflex,
        },
        "detector": crop_results,
    }


def build_near_duplicate_clusters(paths: List[Path]) -> Tuple[Dict[Path, str], List[Dict[str, Any]]]:
    hashes = {path: average_hash(path) for path in paths}
    uf = UnionFind(paths)
    collisions: List[Dict[str, Any]] = []
    for i, p1 in enumerate(paths):
        for p2 in paths[i + 1 :]:
            distance = hamming(hashes[p1], hashes[p2])
            if distance <= PHASH_THRESHOLD:
                uf.union(p1, p2)
                collisions.append(
                    {
                        "distance": distance,
                        "path1": p1.relative_to(PROJECT_ROOT).as_posix(),
                        "path2": p2.relative_to(PROJECT_ROOT).as_posix(),
                        "label1": infer_label(p1),
                        "label2": infer_label(p2),
                        "same_label": infer_label(p1) == infer_label(p2),
                    }
                )

    groups: Dict[Path, List[Path]] = defaultdict(list)
    for path in paths:
        groups[uf.find(path)].append(path)

    participant_map: Dict[Path, str] = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda items: str(sorted(items)[0])), 1):
        pid = f"PEDSEYE_INFERRED_G{idx:04d}"
        for member in members:
            participant_map[member] = pid
    return participant_map, collisions


def assign_splits(paths: List[Path], participant_map: Dict[Path, str]) -> Dict[str, str]:
    pid_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        pid_to_paths[participant_map[path]].append(path)

    pid_to_label: Dict[str, str] = {}
    for pid, members in pid_to_paths.items():
        counts = Counter(infer_label(path) or "unknown" for path in members)
        pid_to_label[pid] = counts.most_common(1)[0][0]

    split_by_pid: Dict[str, str] = {}
    for label in LABELS:
        pids = sorted(pid for pid, pid_label in pid_to_label.items() if pid_label == label)
        n = len(pids)
        if n == 0:
            continue
        train_cut = max(1, int(round(n * 0.70)))
        val_cut = train_cut + max(1, int(round(n * 0.15))) if n >= 3 else train_cut
        for i, pid in enumerate(pids):
            if i < train_cut:
                split_by_pid[pid] = "train"
            elif i < val_cut:
                split_by_pid[pid] = "val"
            else:
                split_by_pid[pid] = "test"
    return split_by_pid


def create_records(paths: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    participant_map, near_duplicates = build_near_duplicate_clusters(paths)
    split_by_pid = assign_splits(paths, participant_map)

    records: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths, 1):
        label = infer_label(path)
        if label is None:
            continue
        quality = analyze_image_quality(path)
        pid = participant_map[path]
        capture_meta = {
            "device": None,
            "resolution": quality["resolution"],
            "flash": None,
            "distance_cm": None,
            "crop": "unknown_source_image_service_fallback_crop",
            "orientation": None,
            "participant_id_source": "inferred_near_duplicate_cluster_not_real_patient_id",
        }
        notes = [
            "Label inferred from user-provided Pedseye folder name.",
            "Participant ID is inferred from near-duplicate image clustering; not a real patient identifier.",
        ]
        cluster_labels = sorted(
            {
                infer_label(other)
                for other, other_pid in participant_map.items()
                if other_pid == pid and infer_label(other) is not None
            }
        )
        if len(cluster_labels) > 1:
            notes.append(f"Near-duplicate cluster contains multiple labels: {cluster_labels}. Needs manual review.")
        records.append(
            {
                "image_id": f"pedseye_{idx:04d}",
                "sample_id": str(uuid.uuid5(uuid.NAMESPACE_URL, path.relative_to(PROJECT_ROOT).as_posix())),
                "file_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "participant_id": pid,
                "participant_id_source": "inferred_near_duplicate_cluster_not_real_patient_id",
                "label": label,
                "label_source": "pedseye_folder",
                "label_verified_by": None,
                "label_verified_at": None,
                "clinician_confirmed": False,
                "capture_meta": capture_meta,
                "quality_flags": quality["quality_flags"],
                "detector_summary": quality["detector"],
                "split": split_by_pid.get(pid, "train"),
                "notes": " ".join(notes),
            }
        )

    summary = validate_manifest_records(records, near_duplicates)
    return records, summary


def validate_manifest_records(records: List[Dict[str, Any]], near_duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    by_path = Counter(row["file_path"] for row in records)
    duplicate_paths = [path for path, count in by_path.items() if count > 1]
    if duplicate_paths:
        errors.append(f"Duplicate manifest paths: {duplicate_paths[:10]}")

    sha_counts = Counter(row["sha256"] for row in records)
    duplicate_sha = [sha for sha, count in sha_counts.items() if count > 1]
    if duplicate_sha:
        errors.append(f"Duplicate SHA-256 images: {duplicate_sha[:10]}")

    pid_splits: Dict[str, set[str]] = defaultdict(set)
    for row in records:
        if not (PROJECT_ROOT / row["file_path"]).is_file():
            errors.append(f"Missing image file: {row['file_path']}")
        pid_splits[row["participant_id"]].add(row["split"])

    leaked = {pid: sorted(splits) for pid, splits in pid_splits.items() if len(splits) > 1}
    if leaked:
        errors.append(f"Participant IDs appear in multiple splits: {dict(list(leaked.items())[:10])}")

    counts = Counter(row["label"] for row in records)
    for label in LABELS:
        if counts.get(label, 0) == 0:
            warnings.append(f"Missing class: {label}")
        elif counts[label] < TARGET_RESEARCH_MIN[label]:
            warnings.append(f"Class {label} below research minimum: {counts[label]} < {TARGET_RESEARCH_MIN[label]}")
    warnings.append("participant_id values are inferred clusters, not true patient IDs; patient-level leakage risk remains.")
    if near_duplicates:
        warnings.append(f"Near-duplicate image pairs at threshold <= {PHASH_THRESHOLD}: {len(near_duplicates)}")

    split_counts = Counter(row["split"] for row in records)
    class_split_counts: Dict[str, Dict[str, int]] = {}
    for label in LABELS:
        class_split_counts[label] = dict(Counter(row["split"] for row in records if row["label"] == label))

    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_WARNINGS" if not errors else "FAILED",
        "total_records": len(records),
        "class_counts": dict(counts),
        "split_counts": dict(split_counts),
        "class_split_counts": class_split_counts,
        "participant_count": len(pid_splits),
        "participant_id_source": "inferred_near_duplicate_cluster_not_real_patient_id",
        "participant_leakage": {
            "is_disjoint": not leaked,
            "violating_participants": leaked,
        },
        "exact_duplicate_sha256_count": len(duplicate_sha),
        "near_duplicate_threshold": PHASH_THRESHOLD,
        "near_duplicate_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def draw_overlay_for_image(path: Path, output_root: Path) -> Dict[str, Any]:
    bgr = read_bgr(path)
    overlay = bgr.copy()
    h, w = overlay.shape[:2]
    detections = []

    for crop_name, crop_bgr, region in crop_regions_for_service_contract(bgr):
        x0, y0, x1, y1 = region
        sx = (x1 - x0) / 224.0
        sy = (y1 - y0) / 224.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        cx, cy, diameter = estimate_iris_center_in_crop(rgb)
        reflexes, reflex_status, reflex_tier = detect_reflexes_in_roi(rgb, cx, cy, diameter)
        pupil, pupil_diameter, pupil_status = detect_pupil_in_roi(rgb, cx, cy, diameter)

        def map_point(px: float, py: float) -> Tuple[int, int]:
            return int(round(x0 + px * sx)), int(round(y0 + py * sy))

        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 2)
        iris_pt = map_point(cx, cy)
        iris_r = max(2, int(round((diameter / 2.0) * ((sx + sy) / 2.0))))
        cv2.circle(overlay, iris_pt, iris_r, (255, 0, 0), 2)
        cv2.circle(overlay, iris_pt, 3, (255, 0, 0), -1)

        pupil_pt = None
        if pupil is not None:
            pupil_pt = map_point(float(pupil["x"]), float(pupil["y"]))
            cv2.circle(overlay, pupil_pt, 4, (0, 255, 255), -1)

        reflex_pts = []
        for reflex in reflexes:
            rp = map_point(float(reflex["x"]), float(reflex["y"]))
            reflex_pts.append(rp)
            cv2.circle(overlay, rp, 5, (0, 255, 0), -1)
            cv2.arrowedLine(overlay, iris_pt, rp, (0, 255, 0), 2, tipLength=0.2)

        text = f"{crop_name}: reflex={reflex_status} pupil={pupil_status}"
        cv2.putText(overlay, text, (max(5, x0 + 5), min(h - 10, y0 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        detections.append(
            {
                "crop": crop_name,
                "reflex_status": reflex_status,
                "reflex_tier": reflex_tier,
                "reflex_count": len(reflexes),
                "pupil_status": pupil_status,
                "iris_center_original_px": list(iris_pt),
                "pupil_center_original_px": list(pupil_pt) if pupil_pt else None,
                "reflex_points_original_px": [list(pt) for pt in reflex_pts],
            }
        )

    rel = path.relative_to(PROJECT_ROOT)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return {"source": rel.as_posix(), "overlay": out_path.relative_to(PROJECT_ROOT).as_posix(), "detections": detections}


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_gap_report(path: Path, summary: Dict[str, Any], manifest_path: Path, overlay_count: int) -> None:
    counts = summary["class_counts"]
    rows = "\n".join(
        f"| {label} | {counts.get(label, 0)} | {TARGET_RESEARCH_MIN[label]} | {max(0, TARGET_RESEARCH_MIN[label] - counts.get(label, 0))} |"
        for label in LABELS
    )
    warnings = "\n".join(f"- {warning}" for warning in summary["warnings"])
    md = f"""# Pedseye Hirschberg Phase A Data QA Report

Created: {summary['createdAt']}

Status: `{summary['status']}`

## Outputs

- Manifest: `{manifest_path.relative_to(PROJECT_ROOT).as_posix()}`
- Validation summary: `{SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- Detector overlays: `{OVERLAY_DIR.relative_to(PROJECT_ROOT).as_posix()}` ({overlay_count} images)

## Dataset Counts

| Class | Current images | Research minimum | Gap |
|---|---:|---:|---:|
{rows}

## Split Counts

`{summary['split_counts']}`

## Participant IDs

Participant IDs were inferred from near-duplicate image clusters:
`{summary['participant_id_source']}`.

These IDs are not real patient identifiers and cannot prove patient-level independence.

## Duplicate Findings

- Exact SHA-256 duplicate count: `{summary['exact_duplicate_sha256_count']}`
- Near-duplicate threshold: `{summary['near_duplicate_threshold']}`
- Near-duplicate pair count: `{summary['near_duplicate_count']}`

## Warnings

{warnings}

## Research-Only Recommendation

Do not deploy a Hirschberg model from this dataset. Use this manifest and overlays
for manual review, label correction, pseudostrabismus/poor-quality collection, and
future participant-level data governance.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Phase A QA artifacts for Pedseye Hirschberg data.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--gap-report-output", type=Path, default=GAP_REPORT_PATH)
    parser.add_argument("--overlay-output-dir", type=Path, default=OVERLAY_DIR)
    args = parser.parse_args()

    paths = image_files(args.dataset)
    records, summary = create_records(paths)
    write_jsonl(args.manifest_output, records)

    overlay_rows = [draw_overlay_for_image(path, args.overlay_output_dir) for path in paths]
    summary["overlay_count"] = len(overlay_rows)
    summary["overlay_samples"] = overlay_rows[:10]

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gap_report(args.gap_report_output, summary, args.manifest_output, len(overlay_rows))

    print(json.dumps(
        {
            "manifest": str(args.manifest_output),
            "summary": str(args.summary_output),
            "gap_report": str(args.gap_report_output),
            "overlay_dir": str(args.overlay_output_dir),
            "status": summary["status"],
            "records": len(records),
            "warnings": len(summary["warnings"]),
            "errors": len(summary["errors"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
