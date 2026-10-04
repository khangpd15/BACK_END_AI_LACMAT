#!/usr/bin/env python3
"""Train Hirschberg research candidate v0.4 from the Phase A Pedseye manifest.

This script is research-only. It does not modify production endpoints and does
not replace the currently loaded Hirschberg service model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "remicare_matplotlib_cache"))

from app.services.hirschberg_ai_service import ONNX_PATH, extract_features_from_crop
from scripts.evaluate_and_train_pedseye_hirschberg import crops_for_service_contract, read_bgr

SEED = 20261004
MODEL_ID = "hirschberg-candidate-v0.4"
DATASET_VERSION = "pedseye_hirschberg_manifest_v0.1"
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "pedseye_hirschberg_manifest_v0.1.jsonl"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.4.joblib"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "reports" / "hirschberg_candidate_v0.4_eval.json"
DEFAULT_CARD_OUTPUT = PROJECT_ROOT / "reports" / "hirschberg_candidate_v0.4_model_card.md"
V01_MODEL_PATH = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.1.joblib"
ALL_EXPECTED_LABELS = ["normal", "esotropia", "exotropia", "pseudostrabismus", "poor_quality"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def validate_manifest_for_training(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    label_counts = Counter(str(row.get("label")) for row in rows)
    participant_source = Counter(str(row.get("participant_id_source")) for row in rows)
    split_by_pid: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path = PROJECT_ROOT / str(row.get("file_path", ""))
        if not image_path.is_file():
            errors.append(f"Missing file: {row.get('file_path')}")
        pid = str(row.get("participant_id") or "")
        if not pid:
            errors.append(f"Missing participant_id for {row.get('file_path')}")
        split_by_pid[pid].add(str(row.get("split")))

    leaked = {pid: sorted(splits) for pid, splits in split_by_pid.items() if len(splits) > 1}
    if leaked:
        errors.append(f"participant_id split leakage: {dict(list(leaked.items())[:5])}")

    missing_labels = [label for label in ALL_EXPECTED_LABELS if label_counts.get(label, 0) == 0]
    for label in missing_labels:
        warnings.append(f"Missing class in manifest: {label}")
    for label, count in sorted(label_counts.items()):
        if count < 30:
            warnings.append(f"Class {label} is very small for ML training: {count} images")
    if any("inferred" in source for source in participant_source):
        warnings.append("participant_id is inferred from near-duplicate clustering; true patient independence is unproven.")

    return {
        "errors": errors,
        "warnings": warnings,
        "label_counts": dict(label_counts),
        "participant_count": len(split_by_pid),
        "participant_id_source_counts": dict(participant_source),
        "participant_leakage": {"is_disjoint": not leaked, "violating_participants": leaked},
    }


def load_feature_names() -> List[str]:
    bundle = joblib.load(V01_MODEL_PATH)
    names = list(bundle.get("feature_names") or [])
    if len(names) != 73:
        raise ValueError(f"Expected 73 v0.1 feature names, got {len(names)}")
    return names


def extract_features(rows: Sequence[Dict[str, Any]], class_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[str] = []
    details: List[Dict[str, Any]] = []
    for row in rows:
        label = str(row["label"])
        if label not in class_to_idx:
            continue
        path = PROJECT_ROOT / row["file_path"]
        bgr = read_bgr(path)
        crop_features = [extract_features_from_crop(crop, session) for crop in crops_for_service_contract(bgr)]
        feature_vector = np.mean(np.array(crop_features, dtype=np.float32), axis=0)
        X.append(feature_vector)
        y.append(class_to_idx[label])
        groups.append(str(row["participant_id"]))
        details.append(
            {
                "image_id": row.get("image_id"),
                "file_path": row["file_path"],
                "label": label,
                "participant_id": row["participant_id"],
                "split": row.get("split"),
                "feature_count": int(feature_vector.shape[0]),
            }
        )
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), np.array(groups), details


def candidate_models() -> Dict[str, Pipeline]:
    return {
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=SEED)),
            ]
        ),
    }


def build_group_cv(y: np.ndarray, groups: np.ndarray) -> Tuple[Any, int, str]:
    group_label_counts = Counter()
    for group in sorted(set(groups)):
        labels = y[groups == group]
        group_label_counts[int(Counter(labels).most_common(1)[0][0])] += 1
    min_group_class = min(group_label_counts.values())
    n_splits = max(2, min(5, int(min_group_class)))
    if n_splits >= 2:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED), n_splits, "StratifiedGroupKFold"
    unique_groups = len(set(groups))
    n_splits = max(2, min(5, unique_groups))
    return GroupKFold(n_splits=n_splits), n_splits, "GroupKFold"


def predict_oof(estimator: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv: Any) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.zeros_like(y)
    probs = np.zeros((len(y), len(set(y))), dtype=np.float32)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = estimator
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        if hasattr(model, "predict_proba"):
            fold_probs = model.predict_proba(X[test_idx])
            # Class order can shrink only if a training fold is missing a class; keep defensive.
            if fold_probs.shape[1] == probs.shape[1]:
                probs[test_idx] = fold_probs
    return preds, probs


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    recalls = recall_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    normal_idx = classes.index("normal") if "normal" in classes else 0
    strab_true = y_true != normal_idx
    strab_pred = y_pred != normal_idx
    tp = int(np.sum(strab_true & strab_pred))
    tn = int(np.sum(~strab_true & ~strab_pred))
    fp = int(np.sum(~strab_true & strab_pred))
    fn = int(np.sum(strab_true & ~strab_pred))

    roc_auc = None
    if probs.size and probs.shape[1] == len(classes) and len(np.unique(strab_true)) == 2:
        strab_prob = np.sum(np.delete(probs, normal_idx, axis=1), axis=1)
        try:
            roc_auc = round(float(roc_auc_score(strab_true.astype(int), strab_prob)), 4)
        except ValueError:
            roc_auc = None

    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(range(len(classes))), average="macro", zero_division=0)), 4),
        "confusion_matrix_labels": list(classes),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": {classes[i]: round(float(recalls[i]), 4) for i in range(len(classes))},
        "per_class_precision": {classes[i]: round(float(precisions[i]), 4) for i in range(len(classes))},
        "binary_strabismus_vs_normal": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": round(float(tp / max(1, tp + fn)), 4),
            "specificity": round(float(tn / max(1, tn + fp)), 4),
            "ppv": round(float(tp / max(1, tp + fp)), 4),
            "npv": round(float(tn / max(1, tn + fn)), 4),
            "roc_auc": roc_auc,
        },
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], iterations: int = 2000) -> Dict[str, Any]:
    rng = random.Random(SEED)
    n = len(y_true)
    normal_idx = classes.index("normal")

    def one(indices: List[int]) -> Dict[str, float]:
        yt = y_true[indices]
        yp = y_pred[indices]
        values: Dict[str, float] = {}
        for i, name in enumerate(classes):
            tp = int(np.sum((yt == i) & (yp == i)))
            fn = int(np.sum((yt == i) & (yp != i)))
            tn = int(np.sum((yt != i) & (yp != i)))
            fp = int(np.sum((yt != i) & (yp == i)))
            values[f"recall_{name}"] = tp / max(1, tp + fn)
            values[f"specificity_{name}"] = tn / max(1, tn + fp)
        st = yt != normal_idx
        sp = yp != normal_idx
        tp = int(np.sum(st & sp))
        fn = int(np.sum(st & ~sp))
        tn = int(np.sum(~st & ~sp))
        fp = int(np.sum(~st & sp))
        values["binary_sensitivity"] = tp / max(1, tp + fn)
        values["binary_specificity"] = tn / max(1, tn + fp)
        return values

    base = one(list(range(n)))
    samples: Dict[str, List[float]] = {key: [] for key in base}
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        vals = one(idx)
        for key, value in vals.items():
            samples[key].append(value)
    out: Dict[str, Any] = {}
    for key, arr in samples.items():
        arr = sorted(arr)
        out[key] = {
            "point": round(base[key], 4),
            "ci95": [round(arr[int(0.025 * len(arr))], 4), round(arr[int(0.975 * len(arr)) - 1], 4)],
        }
    return out


def train_and_evaluate(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    present_classes = [label for label in ["normal", "esotropia", "exotropia"] if any(row["label"] == label for row in rows)]
    class_to_idx = {label: i for i, label in enumerate(present_classes)}
    X, y, groups, feature_details = extract_features(rows, class_to_idx)
    cv, n_splits, cv_name = build_group_cv(y, groups)

    comparisons: Dict[str, Any] = {}
    best_name = ""
    best_score = -1.0
    best_pred: np.ndarray | None = None
    best_probs: np.ndarray | None = None
    for name, estimator in candidate_models().items():
        pred, probs = predict_oof(estimator, X, y, groups, cv)
        metrics = metric_bundle(y, pred, probs, present_classes)
        binary = metrics["binary_strabismus_vs_normal"]
        score = metrics["balanced_accuracy"] + metrics["macro_f1"] + (2.0 * binary["sensitivity"])
        comparisons[name] = metrics
        if score > best_score:
            best_score = score
            best_name = name
            best_pred = pred
            best_probs = probs

    assert best_pred is not None and best_probs is not None
    final_model = candidate_models()[best_name]
    final_model.fit(X, y)
    selected_metrics = comparisons[best_name]
    selected_metrics["bootstrap_ci"] = bootstrap_ci(y, best_pred, present_classes)

    false_negatives = []
    normal_idx = present_classes.index("normal")
    for detail, yt, yp in zip(feature_details, y, best_pred):
        if present_classes[int(yt)] != "normal" and int(yp) == normal_idx:
            false_negatives.append({**detail, "predicted": present_classes[int(yp)]})

    train_summary = {
        "trained": True,
        "model_id": MODEL_ID,
        "selected_model": best_name,
        "cv": {"name": cv_name, "splits": n_splits, "group_source": "manifest participant_id"},
        "classes": present_classes,
        "class_to_idx": class_to_idx,
        "feature_count": int(X.shape[1]),
        "comparisons": comparisons,
        "selected_metrics": selected_metrics,
        "false_negative_review": false_negatives,
        "feature_extraction_rows": feature_details,
    }

    bundle = {
        "model_id": MODEL_ID,
        "status": "research_candidate",
        "model_type": best_name,
        "pipeline": final_model,
        "feature_names": load_feature_names(),
        "classes": present_classes,
        "class_to_idx": class_to_idx,
        "dataset_version": DATASET_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "metrics": selected_metrics,
        "is_production": False,
        "non_clinical_declaration": (
            "Research-only Hirschberg candidate trained from a small Pedseye-derived manifest. "
            "Not a clinical diagnosis, not a clinical probability, and not deployed."
        ),
        "limitations": [
            "Dataset has inferred pseudo participant IDs, not true patient IDs.",
            "Dataset is far below recommended sample counts.",
            "pseudostrabismus and poor_quality classes are missing.",
            "Pedseye folder labels are not independently verified in this repository.",
        ],
    }
    return train_summary, bundle


def model_card(report: Dict[str, Any]) -> str:
    training = report["training"]
    metrics = training["selected_metrics"]
    binary = metrics["binary_strabismus_vs_normal"]
    return f"""# {MODEL_ID} Model Card

Status: `research_candidate`

This model is for offline research comparison only. It is not deployed to
`/api/v1/strabismus/predict` and must not be presented as a clinical diagnosis
or clinical probability.

## Dataset

- Manifest: `{report['manifest_path']}`
- Dataset version: `{DATASET_VERSION}`
- Total samples used: `{report['manifest_validation']['total_trainable_rows']}`
- Class counts: `{report['manifest_validation']['label_counts']}`
- Participant IDs: inferred near-duplicate clusters, not true patient IDs.
- Missing classes: `{report['manifest_validation']['missing_expected_labels']}`

## Feature Contract

- Feature count: `{training['feature_count']}`
- Feature source: same 73-feature contract as `hirschberg_candidate_v0.1`.
- Crop aggregation: mean of service-compatible crop feature vectors.

## Selected Baseline

- Model family: `{training['selected_model']}`
- CV: `{training['cv']}`
- Balanced accuracy: `{metrics['balanced_accuracy']}`
- Macro-F1: `{metrics['macro_f1']}`
- Binary sensitivity: `{binary['sensitivity']}`
- Binary specificity: `{binary['specificity']}`
- PPV: `{binary['ppv']}`
- NPV: `{binary['npv']}`
- ROC-AUC: `{binary['roc_auc']}`

## Recommendation

Do not deploy. The dataset is too small, lacks pseudostrabismus and poor-quality
examples, and does not contain true participant IDs. Use this candidate only as a
research baseline for error review and future data collection.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Hirschberg research candidate v0.4 from manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--model-card-output", type=Path, default=DEFAULT_CARD_OUTPUT)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    validation = validate_manifest_for_training(rows)
    trainable_rows = [row for row in rows if row.get("label") in {"normal", "esotropia", "exotropia"}]
    validation["total_trainable_rows"] = len(trainable_rows)
    validation["missing_expected_labels"] = [
        label for label in ALL_EXPECTED_LABELS if validation["label_counts"].get(label, 0) == 0
    ]

    if validation["errors"]:
        report = {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED_MANIFEST_ERRORS",
            "manifest_path": str(args.manifest),
            "manifest_validation": validation,
            "training": {"trained": False},
        }
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "errors": validation["errors"]}, ensure_ascii=False, indent=2))
        return 1

    training, bundle = train_and_evaluate(trainable_rows)
    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_RESEARCH_CANDIDATE_WITH_WARNINGS",
        "manifest_path": str(args.manifest),
        "model_output": str(args.model_output),
        "model_card_output": str(args.model_card_output),
        "seed": SEED,
        "manifest_validation": validation,
        "training": training,
        "deployment_recommendation": {
            "deploy": False,
            "reason": "Dataset is too small, missing pseudostrabismus/poor_quality, and participant IDs are inferred.",
            "allowed_use": "offline research baseline only",
        },
        "governance": [
            "Production endpoint /api/v1/strabismus/predict was not modified.",
            "Existing artifacts v0.1/v0.2/v0.3 were not overwritten.",
            "Output is sàng lọc nghiên cứu only, not a diagnosis.",
        ],
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.model_card_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_card_output.write_text(model_card(report), encoding="utf-8")

    print(json.dumps(
        {
            "status": report["status"],
            "selected_model": training["selected_model"],
            "balanced_accuracy": training["selected_metrics"]["balanced_accuracy"],
            "macro_f1": training["selected_metrics"]["macro_f1"],
            "binary": training["selected_metrics"]["binary_strabismus_vs_normal"],
            "false_negative_count": len(training["false_negative_review"]),
            "model_output": str(args.model_output),
            "eval_output": str(args.eval_output),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
