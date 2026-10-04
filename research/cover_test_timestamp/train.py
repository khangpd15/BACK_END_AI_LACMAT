"""Audit local timestamps; fit only with documented source-specific permission."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

from research.cover_test_timestamp.features import (
    CONTRACT, FEATURE_NAMES, TARGET_RATES, downsample, extract_features, sampling_quality,
    split_clock_segments,
)


SOURCE = "KOREAN_EYE_TRACKER"
SEED = 20261005


def load_recordings(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "samples" not in payload:
            continue
        if payload.get("source") != SOURCE or payload.get("label") not in ("NORMAL", "STRABISMUS"):
            raise ValueError("Unexpected source or class label")
        records.append({"id": path.relative_to(root).as_posix(), "payload": payload,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not records:
        raise ValueError("No normalized Korean timestamp recordings found")
    return records


def audit(records: list[dict]) -> dict:
    quality = []
    invalid = []
    clock_resets = []
    for record in records:
        try:
            samples = record["payload"]["samples"]
            segments = split_clock_segments(samples)
            if len(segments) > 1:
                clock_resets.append({"recording_sha256": record["sha256"],
                    "clock_boundaries": len(segments) - 1,
                    "boundary_indices": np.cumsum([len(s) for s in segments])[:-1].tolist()})
            for segment_index, segment in enumerate(segments):
                try:
                    quality.append(sampling_quality(segment))
                except (ValueError, KeyError, TypeError) as exc:
                    invalid.append({"recording_sha256": record["sha256"],
                                    "segment_index": segment_index, "reason": str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append({"recording_sha256": record["sha256"], "reason": str(exc)})
    rates = [q["median_hz"] for q in quality]
    feasibility = {}
    try:
        x, labels, owners, rejected = build_rows(records)
        feasibility = {"usable_recordings": len(np.unique(owners)), "usable_rate_segment_variants": len(x),
                       "class_counts_recordings": dict(Counter(records[i]["payload"]["label"] for i in np.unique(owners))),
                       "rejected_variants_count": len(rejected), "rejected_variants": rejected,
                       "fitted_model": False}
    except ValueError as exc:
        feasibility = {"fitted_model": False, "reason": str(exc)}
    return {
        "status": "AUDIT_ONLY_NO_TRAINING",
        "recordings": len(records),
        "class_counts": dict(Counter(r["payload"]["label"] for r in records)),
        "samples": sum(len(r["payload"]["samples"]) for r in records),
        "fully_monotonic_recordings": len(records) - len(clock_resets) - sum("segment_index" not in r for r in invalid),
        "recordings_with_clock_boundaries": len(clock_resets),
        "valid_monotonic_segments": len(quality),
        "clock_boundaries": clock_resets,
        "invalid_segments_or_recordings": invalid,
        "observed_median_hz_min_median_max": [float(v) for v in
            (min(rates), np.median(rates), max(rates))] if rates else None,
        "total_gaps_over_100ms": sum(q["gap_over_100ms_count"] for q in quality),
        "segments_with_cover_phase_labels": sum(q["cover_phase_labels_present"] for q in quality),
        "verified_patient_count": None,
        "legacy_report_note": "Local 41 recordings match the previous test conversion count; raw training directory is empty.",
        "permission": "UNVERIFIED",
        "feature_feasibility": feasibility,
        "can_learn_cover_uncover_response": False,
        "source_url": "https://github.com/hyunwoongko/strabismus-recognition",
    }


def verify_permission(path: Path) -> dict:
    permission = json.loads(path.read_text(encoding="utf-8"))
    if (permission.get("source_domain") != SOURCE
            or permission.get("training_permitted") is not True
            or permission.get("consent_verified") is not True
            or not permission.get("verified_by")
            or not permission.get("evidence_file")):
        raise ValueError("Source-specific documented permission and consent are required")
    evidence = (path.parent / permission["evidence_file"]).resolve()
    if not evidence.is_file() or evidence.stat().st_size == 0:
        raise ValueError("Permission evidence file missing or empty")
    return {"source_domain": SOURCE, "verified_by": permission["verified_by"],
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "training_permitted": True, "consent_verified": True}


def build_rows(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows, labels, owners, rejected = [], [], [], []
    for i, record in enumerate(records):
        try:
            segments = split_clock_segments(record["payload"]["samples"])
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"recording_sha256": record["sha256"], "reason": str(exc)})
            continue
        for segment_index, segment in enumerate(segments):
            for hz in TARGET_RATES:
                try:
                    series = downsample(segment, hz)
                    rows.append(extract_features(series))
                    labels.append(record["payload"]["label"])
                    owners.append(i)
                except (ValueError, KeyError, TypeError) as exc:
                    rejected.append({"recording_sha256": record["sha256"], "segment_index": segment_index,
                                     "target_hz": hz, "reason": str(exc)})
    if not rows or set(labels) != {"NORMAL", "STRABISMUS"}:
        raise ValueError("Need usable trajectories from both classes")
    return np.asarray(rows), np.asarray(labels), np.asarray(owners), rejected


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.1,
                                                              max_iter=2000, random_state=SEED))


def verified_groups(records: list[dict], path: Path) -> np.ndarray:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("identity_basis") != "verified_source_patient_mapping" or not manifest.get("verified_by"):
        raise ValueError("Patient grouping requires a verified source mapping; filenames/hashes are not patient IDs")
    mapping = manifest.get("recordings", {})
    groups = [mapping.get(r["id"]) for r in records]
    if any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError("Missing verified patient identity for one or more recordings")
    group_labels = {}
    for record, group in zip(records, groups):
        label = record["payload"]["label"]
        if group in group_labels and group_labels[group] != label:
            raise ValueError("Conflicting labels within a patient group")
        group_labels[group] = label
    return np.asarray(groups)


def metrics(labels, predictions) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=["NORMAL", "STRABISMUS"]).ravel()
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}


def evaluate(records, x, y, owners, groups) -> dict:
    labels = np.array([r["payload"]["label"] for r in records])
    available = np.unique(owners)
    group_counts = [len(set(groups[available][labels[available] == c])) for c in ("NORMAL", "STRABISMUS")]
    folds = min(5, *group_counts)
    if folds < 2:
        raise ValueError("Need at least two verified patient groups per class for grouped evaluation")
    predictions, truth, evaluated_groups = [], [], []
    evaluated_hashes, splits = [], []
    for train, test in StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(
            available, labels[available], groups[available]):
        train_owners, test_owners = available[train], available[test]
        if set(groups[train_owners]) & set(groups[test_owners]):
            raise ValueError("Patient leakage in split")
        training_mask = np.isin(owners, train_owners)
        weights = 1 / np.bincount(owners)[owners[training_mask]]
        model = estimator().fit(x[training_mask], y[training_mask], logisticregression__sample_weight=weights)
        if set(model.classes_) != {"NORMAL", "STRABISMUS"}:
            raise ValueError("Training fold lacks a class")
        split = {"train_sha256": [records[i]["sha256"] for i in train_owners],
                 "test_sha256": [records[i]["sha256"] for i in test_owners]}
        splits.append(split)
        for i in test_owners:
            probabilities = model.predict_proba(x[owners == i]).mean(axis=0)
            predictions.append(model.classes_[int(np.argmax(probabilities))])
            truth.append(labels[i])
            evaluated_groups.append(groups[i])
            evaluated_hashes.append(records[i]["sha256"])
    truth, predictions, evaluated_groups = map(np.asarray, (truth, predictions, evaluated_groups))
    estimates = metrics(truth, predictions)
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(evaluated_groups)
    bootstrap = []
    for _ in range(1000):
        selected = rng.choice(unique_groups, len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(evaluated_groups == g) for g in selected])
        if len(set(truth[idx])) == 2:
            bootstrap.append(metrics(truth[idx], predictions[idx]))
    return {"scope": "Korean source only, grouped OOF, not webcam/Cover Test clinical accuracy",
            "recording_count": len(truth), "verified_patient_count": len(unique_groups),
            "class_counts": dict(Counter(truth.tolist())), "folds": folds,
            "metrics": {k: {"estimate": v, "ci95_patient_bootstrap":
                np.quantile([row[k] for row in bootstrap], [0.025, 0.975]).tolist()} for k, v in estimates.items()},
            "bootstrap_replicates_with_both_classes": len(bootstrap),
            "oof": [{"sha256": h, "label": t, "prediction": p} for h, t, p in
                    zip(evaluated_hashes, truth.tolist(), predictions.tolist())],
            "splits": splits}


def train(records, permission, output: Path, grouping: Path | None = None) -> dict:
    if permission.get("training_permitted") is not True or permission.get("consent_verified") is not True:
        raise ValueError("Training not permitted")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    x, y, owners, rejected = build_rows(records)
    evaluation = {"status": "NOT_EVALUATED_NO_VERIFIED_PATIENT_MAPPING", "accuracy_claim": None}
    if grouping:
        evaluation = evaluate(records, x, y, owners, verified_groups(records, grouping))
    weights = 1 / np.bincount(owners)[owners]
    model = estimator().fit(x, y, logisticregression__sample_weight=weights)
    summary = {"status": "RESEARCH_TRAINED", "seed": SEED, "recordings_total": len(records),
               "recordings_used": len(np.unique(owners)), "training_rows_rate_variants": len(x),
               "rate_variants_are_not_independent_patients": True, "rejected_variants": rejected,
               "evaluation": evaluation, "sklearn_version": sklearn.__version__,
               "training_class_counts": dict(Counter([records[i]["payload"]["label"] for i in np.unique(owners)]))}
    bundle = {"model": model, "name": "korean-binocular-timestamp-30-60-research",
              "version": "v0.1", "contract": CONTRACT, "feature_names": FEATURE_NAMES,
              "source_domain": SOURCE, "timestamp_unit": "seconds", "target_rates_hz": TARGET_RATES,
              "status": "RESEARCH_ONLY", "clinical_validation": False, "cover_response_trained": False,
              "permission": permission, "training_summary": summary,
              "trained_at": datetime.now(timezone.utc).isoformat(),
              "data_sha256": [r["sha256"] for r in records]}
    output.mkdir(parents=True)
    joblib.dump(bundle, output / "model.joblib")
    (output / "training_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/normalized/korean"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--permission", type=Path)
    parser.add_argument("--patient-mapping", type=Path)
    args = parser.parse_args()
    try:
        if args.train:
            if not args.permission:
                raise ValueError("--train requires --permission pointing to documented source-specific rights")
            permission = verify_permission(args.permission)
            records = load_recordings(args.data)
            result = train(records, permission, args.output, args.patient_mapping)
        else:
            if args.output.exists():
                raise FileExistsError("Refusing to overwrite an existing audit")
            result = audit(load_recordings(args.data))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
