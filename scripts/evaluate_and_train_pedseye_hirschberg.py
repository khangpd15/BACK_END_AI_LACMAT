#!/usr/bin/env python3
"""Evaluate Hirschberg research candidates on the Pedseye-labeled image folder.

This script is research-only. It does not modify the production endpoint and it
does not replace the currently loaded Hirschberg service artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "remicare_matplotlib_cache"))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.hirschberg_ai_service import (
    ONNX_PATH,
    extract_features_from_crop,
    predict_hirschberg,
)


DEFAULT_DATASET = PROJECT_ROOT / "dataset_by_pedseye_manifest"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "pedseye_hirschberg_external_eval.json"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.3_pedseye.joblib"
DEFAULT_CARD_OUTPUT = PROJECT_ROOT / "reports" / "hirschberg_candidate_v0.3_pedseye_model_card.md"
CLASSES = ["esotropia", "exotropia", "normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
MIN_SAMPLES_PER_CLASS = 10
SEED = 20261004


def infer_label(path: Path) -> str | None:
    folder = path.parent.name.lower()
    if "esotropia" in folder or "estropia" in path.name.lower():
        return "esotropia"
    if "exotropia" in folder:
        return "exotropia"
    if "normal" in folder:
        return "normal"
    return None


def iter_records(dataset_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = infer_label(path)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(path),
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "label": label,
            }
        )
    return rows


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def crops_for_service_contract(bgr: np.ndarray) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < 1.35 and min(w, h) <= 400:
        return [cv2.resize(bgr, (224, 224))]
    return [
        cv2.resize(bgr[:, : w // 2], (224, 224)),
        cv2.resize(bgr[:, w // 2 :], (224, 224)),
    ]


def predict_current_service(rows: Sequence[Dict[str, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    y_pred: List[str] = []
    details: List[Dict[str, Any]] = []
    for row in rows:
        path = Path(row["image_path"])
        try:
            result = predict_hirschberg(read_bgr(path))
            pred = str(result.get("predictedClass", "UNAVAILABLE")).lower()
            if pred not in CLASS_TO_IDX:
                pred = "unavailable"
            y_pred.append(pred)
            details.append({**row, "prediction": pred, "aiPrediction": result})
        except Exception as exc:
            y_pred.append("error")
            details.append({**row, "prediction": "error", "error": str(exc)})
    return y_pred, details


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    labels = CLASSES
    filtered_true = []
    filtered_pred = []
    unavailable = 0
    for truth, pred in zip(y_true, y_pred):
        if pred not in CLASS_TO_IDX:
            unavailable += 1
            continue
        filtered_true.append(truth)
        filtered_pred.append(pred)

    if not filtered_true:
        return {"evaluated_samples": 0, "unavailable_predictions": unavailable}

    cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)
    report = classification_report(filtered_true, filtered_pred, labels=labels, output_dict=True, zero_division=0)
    strabismus_truth = np.array([t != "normal" for t in filtered_true], dtype=bool)
    strabismus_pred = np.array([p != "normal" for p in filtered_pred], dtype=bool)
    tp = int(np.sum(strabismus_truth & strabismus_pred))
    tn = int(np.sum(~strabismus_truth & ~strabismus_pred))
    fp = int(np.sum(~strabismus_truth & strabismus_pred))
    fn = int(np.sum(strabismus_truth & ~strabismus_pred))

    return {
        "evaluated_samples": len(filtered_true),
        "unavailable_predictions": unavailable,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": round(float(balanced_accuracy_score(filtered_true, filtered_pred)), 4),
        "macro_f1": round(float(f1_score(filtered_true, filtered_pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": report,
        "binary_strabismus_vs_normal": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": round(float(tp / max(1, tp + fn)), 4),
            "specificity": round(float(tn / max(1, tn + fp)), 4),
        },
    }


def extract_dataset_features(rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    import onnxruntime as ort

    onnx_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    feature_details: List[Dict[str, Any]] = []
    for row in rows:
        path = Path(row["image_path"])
        bgr = read_bgr(path)
        crop_features = [extract_features_from_crop(crop, onnx_sess) for crop in crops_for_service_contract(bgr)]
        feature_vector = np.mean(np.array(crop_features, dtype=np.float32), axis=0)
        x_rows.append(feature_vector)
        y_rows.append(CLASS_TO_IDX[row["label"]])
        feature_details.append({**row, "feature_count": int(feature_vector.shape[0])})
    return np.vstack(x_rows).astype(np.float32), np.array(y_rows, dtype=np.int64), feature_details


def candidate_models() -> Dict[str, Pipeline]:
    return {
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
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


def train_candidate_if_allowed(
    rows: Sequence[Dict[str, str]],
    model_output: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    counts = Counter(row["label"] for row in rows)
    if any(counts.get(cls, 0) < MIN_SAMPLES_PER_CLASS for cls in CLASSES):
        return {
            "trained": False,
            "reason": f"Need at least {MIN_SAMPLES_PER_CLASS} images per class; found {dict(counts)}.",
        }, None

    X, y, feature_details = extract_dataset_features(rows)
    min_class_count = min(Counter(y).values())
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    comparisons: Dict[str, Any] = {}
    best_name = ""
    best_score = -1.0
    best_model: Pipeline | None = None
    for name, estimator in candidate_models().items():
        pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
        y_true_names = [CLASSES[i] for i in y]
        y_pred_names = [CLASSES[i] for i in pred]
        metrics = compute_metrics(y_true_names, y_pred_names)
        comparisons[name] = metrics
        binary = metrics.get("binary_strabismus_vs_normal", {})
        score = (
            2.0 * float(binary.get("sensitivity", 0.0))
            + float(metrics.get("macro_f1", 0.0))
            + float(metrics.get("balanced_accuracy", 0.0))
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_model = estimator

    assert best_model is not None
    best_model.fit(X, y)
    selected_metrics = comparisons[best_name]
    bundle = {
        "model_id": "hirschberg-candidate-v0.3-pedseye",
        "status": "research_candidate",
        "model_type": best_name,
        "pipeline": best_model,
        "feature_names": [f"hirschberg_v01_feature_{i:02d}" for i in range(X.shape[1])],
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "dataset_version": "dataset_by_pedseye_manifest-v0.1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "metrics": selected_metrics,
        "is_production": False,
        "non_clinical_declaration": (
            "Research-only candidate trained on a small Pedseye-labeled image folder. "
            "Labels are treated as external reference labels, not independent clinical ground truth. "
            "Not deployed and not for diagnosis."
        ),
        "limitations": [
            "Only 67 images; no participant_id manifest, so patient-level leakage risk is unknown.",
            "No pseudostrabismus class in this folder.",
            "Pedseye labels were not independently verified inside this repository.",
            "Evaluation is cross-validation on a small image folder, not prospective clinical validation.",
        ],
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    return {
        "trained": True,
        "model_output": str(model_output),
        "selected_model": best_name,
        "cv_splits": n_splits,
        "feature_count": int(X.shape[1]),
        "model_comparison": comparisons,
        "feature_extraction": feature_details,
    }, bundle


def write_model_card(path: Path, train_summary: Dict[str, Any], dataset_counts: Dict[str, int]) -> None:
    if not train_summary.get("trained"):
        return
    selected = train_summary["selected_model"]
    selected_metrics = train_summary["model_comparison"][selected]
    md = f"""# hirschberg-candidate-v0.3-pedseye Model Card

Status: `research_candidate`

This artifact is for offline research only. It is not deployed to `/api/v1/strabismus/predict`
and must not be presented as a clinical diagnosis or clinical probability.

## Dataset

- Dataset: `dataset_by_pedseye_manifest-v0.1`
- Image count: `{sum(dataset_counts.values())}`
- Class counts: `{dataset_counts}`
- Label source: folder labels from the user-provided Pedseye image set.
- Participant IDs: unavailable, so patient-level leakage cannot be ruled out.
- Pseudostrabismus class: absent.

## Feature Contract

- Feature count: `73`
- Extractor: `app.services.hirschberg_ai_service.extract_features_from_crop`
- Aggregation: mean of left/right heuristic crops for full-face images.

## Selected Model

- Model family: `{selected}`
- CV splits: `{train_summary['cv_splits']}`
- Balanced accuracy: `{selected_metrics.get('balanced_accuracy')}`
- Macro-F1: `{selected_metrics.get('macro_f1')}`
- Binary strabismus sensitivity: `{selected_metrics.get('binary_strabismus_vs_normal', {}).get('sensitivity')}`
- Binary strabismus specificity: `{selected_metrics.get('binary_strabismus_vs_normal', {}).get('specificity')}`

## Recommendation

Do not deploy this artifact as production. Use it to compare candidate behavior and to
decide which Pedseye cases need manual review, especially false negatives and normal
false positives.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and optionally train on Pedseye Hirschberg images.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--model-card-output", type=Path, default=DEFAULT_CARD_OUTPUT)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    rows = iter_records(args.dataset)
    y_true = [row["label"] for row in rows]
    counts = dict(Counter(y_true))

    y_pred, prediction_details = predict_current_service(rows)
    current_metrics = compute_metrics(y_true, y_pred)

    train_summary: Dict[str, Any]
    if args.skip_train:
        train_summary = {"trained": False, "reason": "--skip-train was set"}
        bundle = None
    else:
        train_summary, bundle = train_candidate_if_allowed(rows, args.model_output)
        write_model_card(args.model_card_output, train_summary, counts)

    false_negative_review = [
        item
        for item in prediction_details
        if item["label"] in {"esotropia", "exotropia"} and item.get("prediction") == "normal"
    ]

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_RESEARCH_EVALUATION",
        "dataset": {
            "path": str(args.dataset),
            "version": "dataset_by_pedseye_manifest-v0.1",
            "counts": counts,
            "total": len(rows),
            "label_source": "user-provided Pedseye folder labels",
            "participant_ids_available": False,
            "leakage_risk": "unknown because participant_id/group_id manifest is not available",
        },
        "current_service_model_eval": current_metrics,
        "current_service_prediction_details": prediction_details,
        "current_service_false_negative_review": false_negative_review,
        "training": train_summary,
        "governance": [
            "Research-only evaluation; no production endpoint changed.",
            "Pedseye folder labels are used as external reference labels, not independently verified clinical ground truth.",
            "No clinical probability or diagnosis claim is made.",
            "New model artifact, if trained, is versioned separately and not auto-loaded by the backend service.",
        ],
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "dataset_counts": counts,
        "current_service_balanced_accuracy": current_metrics.get("balanced_accuracy"),
        "current_service_macro_f1": current_metrics.get("macro_f1"),
        "false_negative_count": len(false_negative_review),
        "trained_new_candidate": train_summary.get("trained"),
        "selected_model": train_summary.get("selected_model"),
        "report_output": str(args.report_output),
        "model_output": train_summary.get("model_output"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
