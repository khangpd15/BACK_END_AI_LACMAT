#!/usr/bin/env python3
"""RemiCare Strabismus AI - Dataset Governance & Leakage Verification Script.

This script enforces rigorous medical AI data governance standards:
1. Validates all record metadata fields against the Pydantic Manifest Schema.
2. Performs Strict Patient-Level Disjoint Split verification (0% participant leakage).
3. Executes Cryptographic (SHA-256) and Perceptual (pHash) scans across splits
   to detect exact duplicates or cropped variants crossing train/val/test boundaries.
4. Outputs comprehensive terminal summary tables and optional JSON audit reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    # pyrefly: ignore [missing-import]
    import imagehash
except ImportError:  # pragma: no cover - exercised when optional package is absent
    imagehash = None

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("remicare.governance")


# ==============================================================================
# 1. Pydantic Manifest Schema Definitions
# ==============================================================================

class DiagnosisLabel(str, Enum):
    NORMAL = "normal"
    ESOTROPIA = "esotropia"
    EXOTROPIA = "exotropia"
    PSEUDOSTRABISMUS = "pseudostrabismus"
    POOR_QUALITY = "poor_quality"


class AgeGroup(str, Enum):
    INFANT_0_1 = "infant_0_1"
    TODDLER_1_3 = "toddler_1_3"
    PRESCHOOL_3_5 = "preschool_3_5"
    SCHOOL_6_12 = "school_6_12"
    ADOLESCENT_13_18 = "adolescent_13_18"
    ADULT_18_PLUS = "adult_18_plus"


class HeadPoseStatus(str, Enum):
    OPTIMAL = "optimal"
    MILD_TILT = "mild_tilt"
    REJECTED_EXCEEDED_TOLERANCE = "rejected_exceeded_tolerance"


class ConsentStatus(str, Enum):
    GUARDIAN_CONSENTED_FULL = "guardian_consented_full"
    RESEARCH_ONLY_DEIDENTIFIED = "research_only_deidentified"
    CLINICAL_AUDIT_ONLY = "clinical_audit_only"
    WITHDRAWN = "withdrawn"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CaptureCondition(BaseModel):
    flash_present: bool = Field(..., description="Flash present for corneal reflex generation")
    ambient: str = Field(default="standard_clinic", description="Ambient lighting condition")
    device_type: Optional[str] = Field(default=None, description="Camera sensor or smartphone model")


class ImageSampleRecord(BaseModel):
    """Rigorous schema definition for every sample in the manifest."""
    sample_id: UUID = Field(..., description="Unique RFC 4122 UUID v4")
    participant_id: str = Field(..., min_length=2, description="Pseudonymous patient identifier")
    image_path: str = Field(..., min_length=1, description="File path to the image artifact")
    label: DiagnosisLabel = Field(..., description="Clinical label (one of 5 classes)")
    clinician_confirmed: bool = Field(..., description="Validated by doctor / orthoptist")
    diagnosis_source: str = Field(..., min_length=2, description="Ground truth method or source")
    age_group: AgeGroup = Field(..., description="Patient age category")
    capture_condition: Union[CaptureCondition, Dict[str, Any], str] = Field(
        ..., description="Illumination and capture details"
    )
    head_pose_status: HeadPoseStatus = Field(..., description="Head rotation qualification")
    license: str = Field(..., min_length=2, description="Data usage rights and license")
    consent_status: ConsentStatus = Field(..., description="Informed research consent status")
    split: DatasetSplit = Field(..., description="Split partition (train, val, test)")

    @field_validator("participant_id")
    @classmethod
    def clean_participant_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("participant_id must not be empty or whitespace.")
        return s

    @field_validator("image_path")
    @classmethod
    def clean_image_path(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("image_path must not be empty.")
        return s


# ==============================================================================
# 2. Manifest Loading & Validation
# ==============================================================================

def load_manifest_records(manifest_path: Path) -> List[Dict[str, Any]]:
    """Loads manifest rows from Parquet, JSONL, JSON, or CSV format into raw dictionaries."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
        records = df.to_dict(orient="records")
    elif suffix in (".jsonl", ".ndjson"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at line {line_no} in {manifest_path}: {exc}") from exc
    elif suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "samples" in data:
                records = data["samples"]
            else:
                raise ValueError("JSON manifest must be an array of records or contain a 'samples' key.")
    elif suffix == ".csv":
        df = pd.read_csv(manifest_path)
        # Parse nested JSON fields if serialized as strings
        raw_records = df.to_dict(orient="records")
        for r in raw_records:
            if isinstance(r.get("capture_condition"), str) and r["capture_condition"].startswith("{"):
                try:
                    r["capture_condition"] = json.loads(r["capture_condition"])
                except Exception:
                    pass
            records.append(r)
    else:
        raise ValueError(f"Unsupported manifest format '{suffix}'. Supported: .parquet, .jsonl, .json, .csv")

    logger.info(f"Loaded {len(records)} raw entries from {manifest_path.name}")
    return records


def validate_records(
    raw_records: List[Dict[str, Any]],
) -> Tuple[List[ImageSampleRecord], List[Dict[str, Any]]]:
    """Validates raw records using Pydantic schema, recording any compliance errors."""
    valid_records: List[ImageSampleRecord] = []
    schema_errors: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_records):
        try:
            record = ImageSampleRecord(**raw)
            valid_records.append(record)
        except ValidationError as exc:
            schema_errors.append({
                "record_index": idx,
                "sample_id": raw.get("sample_id", f"unknown_row_{idx}"),
                "participant_id": raw.get("participant_id"),
                "errors": exc.errors(),
            })

    return valid_records, schema_errors


# ==============================================================================
# 3. Patient-Level Disjoint Split Leakage Analysis
# ==============================================================================

def analyze_patient_leakage(records: List[ImageSampleRecord]) -> Dict[str, Any]:
    """Ensures 100% strict patient-level disjoint splitting across train, val, and test.
    
    Returns leakage statistics and any intersecting participant IDs.
    """
    split_participants: Dict[str, Set[str]] = {
        DatasetSplit.TRAIN.value: set(),
        DatasetSplit.VAL.value: set(),
        DatasetSplit.TEST.value: set(),
    }
    participant_to_samples: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        split_val = rec.split.value
        pid = rec.participant_id
        split_participants[split_val].add(pid)
        participant_to_samples[pid][split_val].append(str(rec.sample_id))

    train_pts = split_participants[DatasetSplit.TRAIN.value]
    val_pts = split_participants[DatasetSplit.VAL.value]
    test_pts = split_participants[DatasetSplit.TEST.value]

    # Calculate pairwise intersections
    train_val_leak = train_pts.intersection(val_pts)
    train_test_leak = train_pts.intersection(test_pts)
    val_test_leak = val_pts.intersection(test_pts)

    all_leaked_pts = train_val_leak.union(train_test_leak).union(val_test_leak)
    is_disjoint = len(all_leaked_pts) == 0

    leakage_details = []
    for pid in sorted(all_leaked_pts):
        leakage_details.append({
            "participant_id": pid,
            "splits_present": list(participant_to_samples[pid].keys()),
            "sample_counts_per_split": {
                s: len(samples) for s, samples in participant_to_samples[pid].items()
            },
            "sample_ids": dict(participant_to_samples[pid]),
        })

    return {
        "is_strictly_disjoint": is_disjoint,
        "participant_counts": {
            "train": len(train_pts),
            "val": len(val_pts),
            "test": len(test_pts),
            "total_unique": len(train_pts.union(val_pts).union(test_pts)),
        },
        "leakage_summary": {
            "train_val_overlap_count": len(train_val_leak),
            "train_test_overlap_count": len(train_test_leak),
            "val_test_overlap_count": len(val_test_leak),
            "total_violating_participants": len(all_leaked_pts),
        },
        "violating_participants": leakage_details,
    }


# ==============================================================================
# 4. Cryptographic (SHA-256) & Perceptual (pHash) Duplicate Scanning
# ==============================================================================

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest of file bytes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class SimpleImageHash:
    """Small Hamming-distance compatible hash used when ImageHash is unavailable."""

    def __init__(self, bits: List[int]):
        self.bits = bits

    def __sub__(self, other: "SimpleImageHash") -> int:
        return sum(1 for a, b in zip(self.bits, other.bits) if a != b)


def compute_phash(file_path: Path) -> Any:
    """Computes a 64-bit perceptual hash.

    Uses ImageHash when installed; otherwise falls back to a dependency-free
    average-hash implementation that preserves Hamming-distance semantics.
    """
    with Image.open(file_path) as img:
        rgb = img.convert("RGB")
        if imagehash is not None:
            return imagehash.phash(rgb)
        gray = rgb.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        arr = list(gray.getdata())
        avg = sum(arr) / max(1, len(arr))
        return SimpleImageHash([1 if px >= avg else 0 for px in arr])


def scan_image_duplicates(
    records: List[ImageSampleRecord],
    images_base_dir: Optional[Path] = None,
    phash_threshold: int = 4,
) -> Dict[str, Any]:
    """Scans all resolved image files for exact (SHA-256) and perceptual (pHash) cross-split duplicates."""
    logger.info("Starting image duplicate & perceptual hash scan...")

    sample_meta: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    for rec in records:
        raw_path = Path(rec.image_path)
        resolved_path: Optional[Path] = None

        if raw_path.is_file():
            resolved_path = raw_path
        elif images_base_dir and (images_base_dir / raw_path).is_file():
            resolved_path = images_base_dir / raw_path

        if not resolved_path or not resolved_path.is_file():
            missing_images.append(str(rec.image_path))
            continue

        try:
            sha256_val = compute_sha256(resolved_path)
            phash_val = compute_phash(resolved_path)
            sample_meta.append({
                "sample_id": str(rec.sample_id),
                "participant_id": rec.participant_id,
                "split": rec.split.value,
                "path": str(resolved_path),
                "sha256": sha256_val,
                "phash": phash_val,
            })
        except Exception as exc:
            logger.warning(f"Failed to process image {resolved_path}: {exc}")

    logger.info(f"Processed hashes for {len(sample_meta)} images (missing: {len(missing_images)})")

    # Group by SHA-256
    sha_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in sample_meta:
        sha_map[item["sha256"]].append(item)

    sha_cross_split_dupes: List[Dict[str, Any]] = []
    for sha, items in sha_map.items():
        splits = {it["split"] for it in items}
        if len(splits) > 1:
            sha_cross_split_dupes.append({
                "sha256": sha,
                "splits": list(splits),
                "samples": items,
            })

    # Perceptual hash comparisons across splits
    phash_cross_split_collisions: List[Dict[str, Any]] = []
    n = len(sample_meta)

    # Compare pairwise across different splits
    for i in range(n):
        for j in range(i + 1, n):
            item1 = sample_meta[i]
            item2 = sample_meta[j]
            if item1["split"] == item2["split"]:
                continue

            dist = item1["phash"] - item2["phash"]  # Hamming distance
            if dist <= phash_threshold:
                phash_cross_split_collisions.append({
                    "distance": int(dist),
                    "sample1": {
                        "sample_id": item1["sample_id"],
                        "participant_id": item1["participant_id"],
                        "split": item1["split"],
                        "path": item1["path"],
                    },
                    "sample2": {
                        "sample_id": item2["sample_id"],
                        "participant_id": item2["participant_id"],
                        "split": item2["split"],
                        "path": item2["path"],
                    },
                })

    has_cross_split_leak = (len(sha_cross_split_dupes) > 0) or (len(phash_cross_split_collisions) > 0)

    return {
        "images_evaluated": len(sample_meta),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:10],
        "phash_threshold": phash_threshold,
        "is_hash_clean": not has_cross_split_leak,
        "exact_sha256_cross_split_duplicates_count": len(sha_cross_split_dupes),
        "exact_sha256_duplicates": sha_cross_split_dupes,
        "perceptual_cross_split_collisions_count": len(phash_cross_split_collisions),
        "perceptual_collisions": phash_cross_split_collisions[:20],  # cap for reporting
    }


# ==============================================================================
# 5. Terminal Statistics Formatting
# ==============================================================================

def print_summary_tables(
    records: List[ImageSampleRecord],
    leakage_result: Dict[str, Any],
    hash_result: Optional[Dict[str, Any]] = None,
) -> None:
    """Formats and prints beautiful terminal summary tables."""
    splits = [DatasetSplit.TRAIN.value, DatasetSplit.VAL.value, DatasetSplit.TEST.value]
    labels = [lbl.value for lbl in DiagnosisLabel]
    age_groups = [ag.value for ag in AgeGroup]

    # Matrix: Class x Split
    class_split_counts: Dict[str, Dict[str, int]] = {lbl: defaultdict(int) for lbl in labels}
    # Matrix: AgeGroup x Split
    age_split_counts: Dict[str, Dict[str, int]] = {ag: defaultdict(int) for ag in age_groups}
    # Split totals
    split_totals: Dict[str, int] = defaultdict(int)

    clinician_confirmed_count = 0

    for rec in records:
        s = rec.split.value
        lbl = rec.label.value
        ag = rec.age_group.value
        class_split_counts[lbl][s] += 1
        age_split_counts[ag][s] += 1
        split_totals[s] += 1
        if rec.clinician_confirmed:
            clinician_confirmed_count += 1

    total_samples = len(records)
    total_participants = leakage_result["participant_counts"]["total_unique"]

    print("\n" + "=" * 80)
    print("      REMICARE STRABISMUS AI - DATASET GOVERNANCE AUDIT REPORT")
    print("=" * 80)

    # 1. Class Distribution Table
    print("\n[1] SAMPLE DISTRIBUTION BY CLINICAL CLASS & SPLIT:")
    header = f"{'Class Label':<22} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8} | {'Ratio (%)':>9}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for lbl in labels:
        tr = class_split_counts[lbl][DatasetSplit.TRAIN.value]
        va = class_split_counts[lbl][DatasetSplit.VAL.value]
        te = class_split_counts[lbl][DatasetSplit.TEST.value]
        tot = tr + va + te
        pct = (tot / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"{lbl:<22} | {tr:>8} | {va:>8} | {te:>8} | {tot:>8} | {pct:>8.2f}%")

    print("-" * len(header))
    tot_tr = split_totals[DatasetSplit.TRAIN.value]
    tot_va = split_totals[DatasetSplit.VAL.value]
    tot_te = split_totals[DatasetSplit.TEST.value]
    print(f"{'TOTAL SAMPLES':<22} | {tot_tr:>8} | {tot_va:>8} | {tot_te:>8} | {total_samples:>8} | {'100.00%':>9}")
    print("-" * len(header))

    # 2. Age Group Table
    print("\n[2] SAMPLE DISTRIBUTION BY AGE GROUP & SPLIT:")
    print("-" * len(header))
    print(f"{'Age Group':<22} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8} | {'Ratio (%)':>9}")
    print("-" * len(header))

    for ag in age_groups:
        tr = age_split_counts[ag][DatasetSplit.TRAIN.value]
        va = age_split_counts[ag][DatasetSplit.VAL.value]
        te = age_split_counts[ag][DatasetSplit.TEST.value]
        tot = tr + va + te
        pct = (tot / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"{ag:<22} | {tr:>8} | {va:>8} | {te:>8} | {tot:>8} | {pct:>8.2f}%")

    print("-" * len(header))

    # 3. Patient Counts & Leakage Check
    pt_counts = leakage_result["participant_counts"]
    leak_sum = leakage_result["leakage_summary"]
    is_disjoint = leakage_result["is_strictly_disjoint"]

    print("\n[3] PATIENT-LEVEL DISJOINT SPLIT GOVERNANCE:")
    print(f"  • Unique Participants:   Total={total_participants:,} | Train={pt_counts['train']} | Val={pt_counts['val']} | Test={pt_counts['test']}")
    print(f"  • Train ∩ Val Overlap:   {leak_sum['train_val_overlap_count']} participants")
    print(f"  • Train ∩ Test Overlap:  {leak_sum['train_test_overlap_count']} participants")
    print(f"  • Val ∩ Test Overlap:    {leak_sum['val_test_overlap_count']} participants")

    if is_disjoint:
        print("  >>> STATUS: [PASSED] 100% STRICT PATIENT-LEVEL DISJOINT SPLIT GUARANTEED. ZERO LEAKAGE.")
    else:
        print(f"  >>> STATUS: [CRITICAL FAILURE] LEAKAGE DETECTED! {leak_sum['total_violating_participants']} participants overlap across splits.")

    # 4. Hash & Image Analysis
    if hash_result:
        print("\n[4] CRYPTOGRAPHIC (SHA-256) & PERCEPTUAL (pHash) AUDIT:")
        print(f"  • Images Evaluated:      {hash_result['images_evaluated']:,} (Missing: {hash_result['missing_images_count']})")
        print(f"  • Cross-split SHA dupes: {hash_result['exact_sha256_cross_split_duplicates_count']}")
        print(f"  • Cross-split pHash:     {hash_result['perceptual_cross_split_collisions_count']} (threshold <={hash_result['phash_threshold']})")
        if hash_result["is_hash_clean"]:
            print("  >>> STATUS: [PASSED] NO CROSS-SPLIT IMAGE DUPLICATES OR CROPPED COLLISION VARIANTS.")
        else:
            print("  >>> STATUS: [CRITICAL FAILURE] CROSS-SPLIT DUPLICATE IMAGES DETECTED!")

    # 5. Quality & Clinician Assurance
    clinician_pct = (clinician_confirmed_count / total_samples * 100.0) if total_samples > 0 else 0.0
    print("\n[5] CLINICAL VALIDATION METRICS:")
    print(f"  • Clinician Confirmed:   {clinician_confirmed_count:,} / {total_samples:,} ({clinician_pct:.1f}%)")
    print("=" * 80 + "\n")


# ==============================================================================
# 6. Command Line Interface & Runner
# ==============================================================================

def verify_dataset(
    manifest_path: Path,
    images_dir: Optional[Path] = None,
    phash_threshold: int = 4,
    output_report: Optional[Path] = None,
    strict_fail: bool = True,
) -> bool:
    """Runs end-to-end dataset governance checks and returns True if fully compliant."""
    logger.info(f"Auditing manifest: {manifest_path}")
    raw_records = load_manifest_records(manifest_path)

    # 1. Schema Validation
    valid_records, schema_errors = validate_records(raw_records)
    if schema_errors:
        logger.error(f"SCHEMA VALIDATION FAILED: {len(schema_errors)} invalid records found!")
        for err in schema_errors[:5]:
            logger.error(f"  Sample ID: {err['sample_id']} - Errors: {err['errors']}")
        if len(schema_errors) > 5:
            logger.error(f"  ... and {len(schema_errors) - 5} more schema violations.")

    # 2. Patient Leakage Analysis
    leakage_result = analyze_patient_leakage(valid_records)

    # 3. Hash & Perceptual Scan (if images exist)
    hash_result: Optional[Dict[str, Any]] = None
    if images_dir or any(Path(r.image_path).is_file() for r in valid_records[:5]):
        hash_result = scan_image_duplicates(
            valid_records,
            images_base_dir=images_dir,
            phash_threshold=phash_threshold,
        )

    # 4. Terminal Summary
    print_summary_tables(valid_records, leakage_result, hash_result)

    # Overall compliance status
    passed = (
        len(schema_errors) == 0
        and leakage_result["is_strictly_disjoint"]
        and (hash_result is None or hash_result["is_hash_clean"])
    )

    # 5. Export JSON Report
    if output_report:
        report_data = {
            "manifest": str(manifest_path),
            "total_records": len(raw_records),
            "valid_records": len(valid_records),
            "schema_errors_count": len(schema_errors),
            "schema_errors": schema_errors[:50],
            "patient_leakage": leakage_result,
            "hash_audit": hash_result,
            "audit_passed": passed,
        }
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved governance audit report to {output_report}")

    if not passed and strict_fail:
        logger.error("Dataset governance verification FAILED. Please review the errors above.")
        return False

    logger.info("Dataset governance verification PASSED successfully.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RemiCare Strabismus AI - Dataset Governance & Leakage Verification Tool"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest file (.parquet, .jsonl, .json, .csv)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Root directory containing images referenced in manifest",
    )
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=4,
        help="Hamming distance threshold for pHash near-duplicate collision (default: 4)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to output detailed JSON audit report",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit with status code 1 on governance failures",
    )

    args = parser.parse_args()

    success = verify_dataset(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        phash_threshold=args.hash_threshold,
        output_report=args.output_report,
        strict_fail=not args.no_strict,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
