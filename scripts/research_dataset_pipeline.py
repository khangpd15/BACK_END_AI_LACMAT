"""Research dataset preparation utilities for RemiCare Phase 5.

This module prepares manifests and split plans only. It never trains a model and
never treats model outputs as ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESEARCH_DATASET_VERSION = "remicare-hirschberg-cover-dataset-v0.1"
RESEARCH_FEATURE_VERSION = "research-geometry-v0.1"
TRAINING_DISABLED_REASON = "TRAINING_REQUIRES_APPROVED_PROTOCOL_DATASET_AND_DOCTOR_LABELS"

REQUIRED_LABEL_FIELDS = (
    "participant_id",
    "label_source",
    "exam_date",
    "horizontal_strabismus_type",
    "intermittent",
    "glasses_group",
    "pseudostrabismus",
    "exam_method",
)

ALLOWED_LABEL_SOURCES = {"doctor", "orthoptist"}
ALLOWED_DOMAINS_FOR_TRAINING = {"hirschberg_cover_protocol_v1"}
LEGACY_DOMAIN = "legacy_non_hirschberg"


@dataclass(frozen=True)
class ParticipantRecord:
    participant_id: str
    domain: str
    consent_research: bool
    consent_raw_video_or_landmarks: bool
    age_years: Optional[int]
    glasses_on: Optional[bool]
    distance_bucket: str
    hirschberg: Dict[str, Any]
    cover: Dict[str, Any]
    ground_truth: Dict[str, Any]


def stable_patient_split(participant_ids: Iterable[str], seed: int = 20261003) -> Dict[str, str]:
    """Assigns each participant to exactly one deterministic split.

    Split is by participant ID only; frames/sessions from the same participant
    must not cross train/validation/test.
    """
    assignments: Dict[str, str] = {}
    for participant_id in sorted({str(pid) for pid in participant_ids if str(pid).strip()}):
        digest = hashlib.sha256(f"{seed}:{participant_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < 0.70:
            split = "train"
        elif bucket < 0.85:
            split = "validation"
        else:
            split = "test"
        assignments[participant_id] = split
    return assignments


def validate_ground_truth(row: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for field in REQUIRED_LABEL_FIELDS:
        if row.get(field) in (None, ""):
            issues.append(f"MISSING_{field.upper()}")
    source = str(row.get("label_source", "")).strip().lower()
    if source and source not in ALLOWED_LABEL_SOURCES:
        issues.append("LABEL_SOURCE_NOT_DOCTOR_OR_ORTHOPTIST")
    if str(row.get("model_prediction", "")).strip():
        issues.append("MODEL_OUTPUT_MUST_NOT_BE_LABEL")
    return issues


def training_eligibility(record: ParticipantRecord) -> Dict[str, Any]:
    reasons: List[str] = []
    if record.domain not in ALLOWED_DOMAINS_FOR_TRAINING:
        reasons.append("DOMAIN_NOT_PROTOCOL_HIRSCHBERG_COVER")
    if record.domain == LEGACY_DOMAIN:
        reasons.append("LEGACY_NON_HIRSCHBERG_FORBIDDEN_FOR_THRESHOLD_OR_TRAINING")
    if not record.consent_research:
        reasons.append("MISSING_RESEARCH_CONSENT")
    reasons.extend(validate_ground_truth(record.ground_truth))
    if not record.hirschberg:
        reasons.append("MISSING_HIRSCHBERG")
    if not record.cover:
        reasons.append("MISSING_COVER")
    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_manifest_jsonl(path: Path) -> List[ParticipantRecord]:
    records: List[ParticipantRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            participant_id = str(raw.get("participant_id") or "").strip()
            if not participant_id:
                raise ValueError(f"Line {line_no}: participant_id is required")
            records.append(
                ParticipantRecord(
                    participant_id=participant_id,
                    domain=str(raw.get("domain") or "UNKNOWN"),
                    consent_research=_coerce_bool(raw.get("consent_research")),
                    consent_raw_video_or_landmarks=_coerce_bool(raw.get("consent_raw_video_or_landmarks")),
                    age_years=raw.get("age_years"),
                    glasses_on=raw.get("glasses_on"),
                    distance_bucket=str(raw.get("distance_bucket") or "UNKNOWN"),
                    hirschberg=raw.get("hirschberg") or {},
                    cover=raw.get("cover") or {},
                    ground_truth=raw.get("ground_truth") or {},
                )
            )
    return records


def read_ground_truth_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("participant_id") or "").strip(): row for row in rows if row.get("participant_id")}


def merge_ground_truth(records: List[ParticipantRecord], labels: Dict[str, Dict[str, Any]]) -> List[ParticipantRecord]:
    merged: List[ParticipantRecord] = []
    for record in records:
        label = labels.get(record.participant_id)
        if not label:
            merged.append(record)
            continue
        raw = asdict(record)
        raw["ground_truth"] = {**record.ground_truth, **label}
        merged.append(ParticipantRecord(**raw))
    return merged


def distance_experiment_report(records: List[ParticipantRecord]) -> Dict[str, Any]:
    by_bucket: Dict[str, List[ParticipantRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.distance_bucket].append(record)

    bucket_reports = {}
    for bucket, bucket_records in sorted(by_bucket.items()):
        focus_good = 0
        exact_reflex = 0
        delta_values = []
        for record in bucket_records:
            h_quality = record.hirschberg.get("quality") or {}
            h_measurements = record.hirschberg.get("measurements") or {}
            if h_quality.get("focusStatus") in {"GOOD", "PASS", "VALID"}:
                focus_good += 1
            reflex = h_quality.get("reflexCountPerEye") or {}
            if reflex.get("OD") == 1 and reflex.get("OS") == 1:
                exact_reflex += 1
            delta_h = h_measurements.get("delta_h")
            if isinstance(delta_h, (int, float)):
                delta_values.append(float(delta_h))

        bucket_reports[bucket] = {
            "participantCount": len({r.participant_id for r in bucket_records}),
            "sampleCount": len(bucket_records),
            "focusPassRatio": round(focus_good / len(bucket_records), 4) if bucket_records else 0,
            "exactOneReflexPerEyeRatio": round(exact_reflex / len(bucket_records), 4) if bucket_records else 0,
            "deltaHDistribution": {
                "count": len(delta_values),
                "min": min(delta_values) if delta_values else None,
                "max": max(delta_values) if delta_values else None,
                "mean": round(sum(delta_values) / len(delta_values), 6) if delta_values else None,
            },
        }
    return {
        "note": "Distance buckets are compared for researcher review only; this report does not choose a clinical standard distance.",
        "buckets": bucket_reports,
    }


def build_dataset_summary(records: List[ParticipantRecord], seed: int = 20261003) -> Dict[str, Any]:
    split = stable_patient_split((r.participant_id for r in records), seed=seed)
    participant_to_split = split
    split_counts = Counter(participant_to_split.values())
    eligibility = {r.participant_id: training_eligibility(r) for r in records}
    eligible_count = sum(1 for item in eligibility.values() if item["eligible"])

    return {
        "datasetVersion": RESEARCH_DATASET_VERSION,
        "featureVersion": RESEARCH_FEATURE_VERSION,
        "seed": seed,
        "recordCount": len(records),
        "participantCount": len(participant_to_split),
        "splitCountsByParticipant": dict(split_counts),
        "participantSplit": participant_to_split,
        "trainingEligibleParticipantCount": eligible_count,
        "trainingEligibility": eligibility,
        "domainCounts": dict(Counter(r.domain for r in records)),
        "distanceExperiment": distance_experiment_report(records),
        "trainingDisabledReason": TRAINING_DISABLED_REASON,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare RemiCare research dataset manifest summary without training.")
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    records = read_manifest_jsonl(args.manifest_jsonl)
    if args.ground_truth_csv:
        records = merge_ground_truth(records, read_ground_truth_csv(args.ground_truth_csv))

    summary = build_dataset_summary(records, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset preparation summary to {args.output}")
    print(TRAINING_DISABLED_REASON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
