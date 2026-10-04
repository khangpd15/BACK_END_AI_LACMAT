"""Legacy three-class geometry experiment, NOT patient-independent validation."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .grouping import cluster, collect

CLASSES = ("esotropia", "exotropia", "normal")
FEATURE_NAMES = ("OD_pupil_offset_axis", "OD_pupil_offset_perpendicular",
                 "OS_pupil_offset_axis", "OS_pupil_offset_perpendicular",
                 "binocular_axis_difference", "binocular_perpendicular_difference")


def legacy_features(row):
    points = {eye: {target: np.array([float(row[f"{eye}_{target}_x"]),
                                    float(row[f"{eye}_{target}_y"])])
                    for target in ("pupil", "reflex")} for eye in ("od", "os")}
    if not all(np.isfinite(v).all() for eye in points.values() for v in eye.values()):
        raise ValueError("Nonfinite legacy point")
    axis = points["os"]["pupil"] - points["od"]["pupil"]
    separation = float(np.linalg.norm(axis))
    if separation <= 1:
        raise ValueError("Degenerate two-pupil axis")
    horizontal = axis / separation
    vertical = np.array([-horizontal[1], horizontal[0]])
    od = (points["od"]["reflex"] - points["od"]["pupil"]) / separation
    os = (points["os"]["reflex"] - points["os"]["pupil"]) / separation
    od_h, od_v = float(od @ horizontal), float(od @ vertical)
    os_h, os_v = float(os @ horizontal), float(os @ vertical)
    return np.array([od_h, od_v, os_h, os_v, od_h - os_h, od_v - os_v])


def scores(y, predicted):
    return {"balanced_accuracy_3class": float(balanced_accuracy_score(y, predicted)),
            "macro_f1_3class": float(f1_score(y, predicted, labels=CLASSES, average="macro", zero_division=0)),
            "recall_per_class": dict(zip(CLASSES, recall_score(y, predicted, labels=CLASSES, average=None,
                                                               zero_division=0).tolist())),
            "confusion_rows_and_columns": list(CLASSES),
            "confusion": confusion_matrix(y, predicted, labels=CLASSES).tolist()}


def evaluate_oof(x, y, groups, seed=42):
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    if set(y) != set(CLASSES):
        raise ValueError("Exploratory cohort requires all three actual classes, no pseudo")
    group_counts = {label: len(set(groups[y == label])) for label in CLASSES}
    folds = min(5, min(group_counts.values()))
    if folds < 2:
        raise ValueError("Not enough provisional groups for cross-validation")
    predicted = np.empty(len(y), dtype=object)
    dummy_predicted = np.empty(len(y), dtype=object)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_details = []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, val) in enumerate(splitter.split(x, y, groups)):
        if set(groups[train]) & set(groups[val]):
            raise ValueError("Provisional group leakage")
        if set(y[train]) != set(CLASSES) or set(y[val]) != set(CLASSES):
            raise ValueError("Fold lacks a class; do not report incomplete three-class CV")
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", random_state=seed, max_iter=1000))
        model.fit(x[train], y[train])
        predicted[val] = model.predict(x[val])
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(x[train], y[train])
        dummy_predicted[val] = dummy.predict(x[val])
        fold_ids[val] = fold
        fold_details.append({"fold": fold, "train_n": len(train), "val_n": len(val),
                             "train_groups": len(set(groups[train])), "val_groups": len(set(groups[val])),
                             "train_counts": dict(Counter(y[train])), "val_counts": dict(Counter(y[val]))})
    if np.any(fold_ids < 0):
        raise ValueError("Missing OOF prediction")
    return {"model": scores(y, predicted), "dummy": scores(y, dummy_predicted),
            "folds": fold_details, "predictions": predicted.tolist(), "fold_ids": fold_ids.tolist()}


def run(root, csv_path, attestation_path, output_dir, seed=42):
    root, output = Path(root).resolve(), Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a new run directory; never overwrite")
    attestation = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    if not (attestation.get("approves_training_rights") is True and
            attestation.get("clinical_label_confirmation") == "reported_by_project_owner" and
            attestation.get("record_id")):
        raise ValueError("Supplied-data research permission and clinical confirmation required")
    inventory = collect(root)
    links_path = root / "datasets/pedseye_hirschberg_manifest_v0.1.jsonl"
    existing_links = {}
    if links_path.exists():
        for line in links_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                # Keep only links explicitly identified as provisional in the old manifest.
                if rec.get("participant_id_source") == "inferred_near_duplicate_cluster_not_real_patient_id":
                    existing_links[rec["file_path"]] = rec["participant_id"]
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    if len({r["relative_path"] for r in annotations}) != len(annotations):
        raise ValueError("Duplicate annotated image")
    records = {r["image_id"]: r for r in inventory}
    x, y = [], []
    for row in annotations:
        path = row["relative_path"]
        if path not in records or row["class_label"] != records[path]["label"]:
            raise ValueError("Annotation/inventory mapping mismatch")
        image = records[path]
        for eye in ("od", "os"):
            for target in ("pupil", "reflex"):
                px, py = float(row[f"{eye}_{target}_x"]), float(row[f"{eye}_{target}_y"])
                if not (0 <= px < image["width"] and 0 <= py < image["height"]):
                    raise ValueError("Legacy annotation outside stored crop")
        x.append(legacy_features(row))
        y.append(row["class_label"])
    results, primary_rows, primary_edges = {}, None, None
    # Fixed sensitivity analysis. The strict threshold is primary because broader
    # thresholds collapse annotated images into fewer, highly uneven groups.
    for threshold in (4, 8, 12):
        assigned, edges, summary = cluster(inventory, threshold, existing_links)
        mapping = {r["image_id"]: r["provisional_group_id"] for r in assigned}
        groups = [mapping[r["relative_path"]] for r in annotations]
        result = evaluate_oof(x, y, groups, seed)
        result["grouping"] = summary
        result["annotated_groups"] = len(set(groups))
        result["oof_records"] = [{"image_id": r["relative_path"], "provisional_group_id": group,
                                   "truth": r["class_label"], "prediction": prediction, "fold": fold}
                                  for r, group, prediction, fold in zip(annotations, groups,
                                       result.pop("predictions"), result.pop("fold_ids"))]
        results[str(threshold)] = result
        if threshold == 4:
            primary_rows, primary_edges = assigned, edges
    report = {"research_only": True, "clinical_evidence": False, "patient_independent": False,
              "split_unit": "provisional_similarity_group_not_patient", "primary_hash_threshold": 4,
              "coordinate_space": "stored_legacy_crop_not_source_native",
              "normalization": "two_pupil_distance_px_not_iris_diameter_not_mm",
              "labels": list(CLASSES), "pseudo_samples": 0, "annotation_n": len(annotations),
              "landmarks_clinician_confirmed": False, "permission_record_id": attestation["record_id"],
              "seed": seed, "feature_names": list(FEATURE_NAMES), "results_by_threshold": results,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "opencv": cv2.__version__},
              "annotation_csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
              "limitations": ["Hash groups can miss the same person in different photographs",
                              "False merges can reduce independent groups",
                              "All old data are development, not an untouched clinical test",
                              "Clinical disease confirmation does not validate point coordinates",
                              "No four-class or high-resolution sub-3px performance is established"]}
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("report.json", report), ("provisional_groups.json", primary_rows),
                            ("candidate_edges.json", primary_edges)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--csv", default="processed/hirschberg_manual_annotations.csv")
    parser.add_argument("--attestation", default="research/hirschberg_phase4/annotation/user_attestation_20261004.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(args.root, args.csv, args.attestation, args.output_dir, args.seed)
    print(json.dumps({"annotation_n": report["annotation_n"], "patient_independent": False,
                      "primary_threshold": report["primary_hash_threshold"],
                      "primary": report["results_by_threshold"][str(report["primary_hash_threshold"])]["model"]}, indent=2))


if __name__ == "__main__":
    main()
