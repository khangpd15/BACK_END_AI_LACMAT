#!/usr/bin/env python3
"""Train Hirschberg candidate v0.5 with a conservative abstention policy.

The current Pedseye-sized dataset is too small for a clinical classifier. This
script therefore trains a research candidate and tunes a confidence threshold
that can return INCONCLUSIVE instead of forcing a wrong 3-class label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.hirschberg_ai_service import ONNX_PATH, extract_features_from_crop  # noqa: E402

SEED = 20261004
MODEL_ID = "hirschberg-candidate-v0.5-safe"
DATASET_VERSION = "pedseye_hirschberg_manifest_v0.1"
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "pedseye_hirschberg_manifest_v0.1.jsonl"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.5_safe.joblib"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "reports" / "hirschberg_candidate_v0.5_safe_eval.json"
DEFAULT_CARD_OUTPUT = PROJECT_ROOT / "docs" / "research" / "hirschberg" / "hirschberg_candidate_v0.5_safe_model_card.md"
CLASSES = ["esotropia", "exotropia", "normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
ALL_EXPECTED_LABELS = ["normal", "esotropia", "exotropia", "pseudostrabismus", "poor_quality"]
V01_MODEL_PATH = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.1.joblib"


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
    for label in ALL_EXPECTED_LABELS:
        if label_counts.get(label, 0) == 0:
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


def build_group_cv(y: np.ndarray, groups: np.ndarray) -> Tuple[Any, int, str]:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

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


def load_feature_names() -> List[str]:
    bundle = joblib.load(V01_MODEL_PATH)
    names = list(bundle.get("feature_names") or [])
    if len(names) != 73:
        raise ValueError(f"Expected 73 v0.1 feature names, got {len(names)}")
    return names


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


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    recalls = recall_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    normal_idx = classes.index("normal")
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


def clone_estimator(estimator: Any) -> Any:
    from sklearn.base import clone

    return clone(estimator)


def pair_feature_names() -> List[str]:
    base = load_feature_names()
    return (
        [f"left_{name}" for name in base]
        + [f"right_{name}" for name in base]
        + [f"left_minus_right_{name}" for name in base]
        + [f"abs_left_minus_right_{name}" for name in base]
        + [f"mean_left_right_{name}" for name in base]
    )


def pair_feature_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, left - right, np.abs(left - right), (left + right) / 2.0]).astype(np.float32)


def extract_pair_features(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[str] = []
    details: List[Dict[str, Any]] = []
    for row in rows:
        label = str(row["label"])
        if label not in CLASS_TO_IDX:
            continue
        bgr = read_bgr(PROJECT_ROOT / row["file_path"])
        crops = crops_for_service_contract(bgr)
        crop_features = [np.array(extract_features_from_crop(crop, session), dtype=np.float32) for crop in crops]
        if len(crop_features) == 1:
            left = right = crop_features[0]
        else:
            left, right = crop_features[0], crop_features[1]
        feature_vector = pair_feature_vector(left, right)
        X.append(feature_vector)
        y.append(CLASS_TO_IDX[label])
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


def candidate_models_v05() -> Dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)),
            ]
        ),
        "random_forest_pair_balanced": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=800,
                        class_weight="balanced",
                        max_depth=4,
                        min_samples_leaf=2,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "extra_trees_pair_balanced": Pipeline(
            [
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        class_weight="balanced",
                        max_depth=4,
                        min_samples_leaf=2,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("model", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.04, random_state=SEED)),
            ]
        ),
    }


def predict_oof(estimator: Any, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv: Any) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.zeros_like(y)
    probs = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = clone_estimator(estimator)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        fold_probs = model.predict_proba(X[test_idx])
        for local_idx, cls_idx in enumerate(model.classes_):
            probs[test_idx, int(cls_idx)] = fold_probs[:, local_idx]
    return preds, probs


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    normal_idx = CLASS_TO_IDX["normal"]
    strab_true = y_true != normal_idx
    strab_pred = y_pred != normal_idx
    tp = int(np.sum(strab_true & strab_pred))
    tn = int(np.sum(~strab_true & ~strab_pred))
    fp = int(np.sum(~strab_true & strab_pred))
    fn = int(np.sum(strab_true & ~strab_pred))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(float(tp / max(1, tp + fn)), 4),
        "specificity": round(float(tn / max(1, tn + fp)), 4),
    }


def evaluate_abstention(y_true: np.ndarray, probs: np.ndarray, threshold: float, margin_threshold: float) -> Dict[str, Any]:
    sorted_probs = np.sort(probs, axis=1)
    confidence = np.max(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    pred = np.argmax(probs, axis=1)
    covered_mask = (confidence >= threshold) & (margin >= margin_threshold)
    covered_count = int(np.sum(covered_mask))
    abstained_count = int(len(y_true) - covered_count)
    wrong_confident = int(np.sum((pred != y_true) & covered_mask))
    correct_confident = int(np.sum((pred == y_true) & covered_mask))

    out: Dict[str, Any] = {
        "threshold": round(float(threshold), 4),
        "margin_threshold": round(float(margin_threshold), 4),
        "coverage": round(float(covered_count / max(1, len(y_true))), 4),
        "covered_count": covered_count,
        "abstained_count": abstained_count,
        "correct_confident": correct_confident,
        "wrong_confident": wrong_confident,
        "covered_accuracy": round(float(correct_confident / max(1, covered_count)), 4),
    }
    if covered_count:
        covered_true = y_true[covered_mask]
        covered_pred = pred[covered_mask]
        out["covered_confusion_matrix_labels"] = CLASSES
        from sklearn.metrics import confusion_matrix

        out["covered_confusion_matrix"] = confusion_matrix(covered_true, covered_pred, labels=list(range(len(CLASSES)))).tolist()
        out["covered_binary_strabismus_vs_normal"] = binary_counts(covered_true, covered_pred)
    else:
        out["covered_confusion_matrix_labels"] = CLASSES
        out["covered_confusion_matrix"] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        out["covered_binary_strabismus_vs_normal"] = None
    return out


def choose_policy(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for threshold in np.round(np.arange(0.34, 0.99, 0.02), 2):
        for margin_threshold in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            candidates.append(evaluate_abstention(y_true, probs, float(threshold), float(margin_threshold)))

    def rank_key(item: Dict[str, Any]) -> Tuple[int, float, float, float]:
        wrong_confident = int(item["wrong_confident"])
        wrong_rate = wrong_confident / max(1, item["covered_count"])
        return (-wrong_confident, -wrong_rate, item["covered_accuracy"], item["coverage"])

    chosen = dict(max(candidates, key=rank_key))
    chosen["selection_rule"] = "Minimize confident wrong predictions first; use coverage only as a tie-breaker."
    zero_wrong = [
        {key: value for key, value in item.items() if key != "grid_summary"}
        for item in candidates
        if item["wrong_confident"] == 0 and item["covered_count"] > 0
    ][:5]
    max_coverage = max(candidates, key=lambda item: item["coverage"])
    chosen["grid_summary"] = {
        "best_zero_wrong": zero_wrong,
        "max_coverage": {key: value for key, value in max_coverage.items() if key != "grid_summary"},
    }
    return chosen


def false_prediction_review(details: Sequence[Dict[str, Any]], y: np.ndarray, probs: np.ndarray, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    confidence = np.max(probs, axis=1)
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    pred = np.argmax(probs, axis=1)
    covered = (confidence >= policy["threshold"]) & (margin >= policy["margin_threshold"])
    rows = []
    for detail, true_idx, pred_idx, conf, mar, is_covered, prob in zip(details, y, pred, confidence, margin, covered, probs):
        if bool(is_covered) and int(true_idx) != int(pred_idx):
            rows.append(
                {
                    "image_id": detail.get("image_id"),
                    "file_path": detail.get("file_path"),
                    "label": CLASSES[int(true_idx)],
                    "predicted": CLASSES[int(pred_idx)],
                    "confidence": round(float(conf), 4),
                    "margin": round(float(mar), 4),
                    "probabilities": {CLASSES[i]: round(float(prob[i]), 4) for i in range(len(CLASSES))},
                }
            )
    return rows


def train(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    X, y, groups, details = extract_pair_features(rows)
    cv, n_splits, cv_name = build_group_cv(y, groups)

    comparisons: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e9
    best_probs: np.ndarray | None = None
    best_pred: np.ndarray | None = None
    for name, estimator in candidate_models_v05().items():
        pred, probs = predict_oof(estimator, X, y, groups, cv)
        metrics = metric_bundle(y, pred, probs, CLASSES)
        policy = choose_policy(y, probs)
        binary = metrics["binary_strabismus_vs_normal"]
        score = (
            2.5 * policy["covered_accuracy"]
            + 1.0 * policy["coverage"]
            + 1.0 * binary["sensitivity"]
            + 0.5 * binary["specificity"]
            - 2.5 * (policy["wrong_confident"] / max(1, policy["covered_count"]))
        )
        comparisons[name] = {
            "forced_3class_metrics": metrics,
            "abstention_policy_oof": policy,
            "selection_score": round(float(score), 4),
        }
        if score > best_score:
            best_score = score
            best_name = name
            best_probs = probs
            best_pred = pred

    if best_probs is None or best_pred is None:
        raise RuntimeError("No candidate model produced predictions.")

    final_model = clone_estimator(candidate_models_v05()[best_name])
    final_model.fit(X, y)
    selected_forced = metric_bundle(y, best_pred, best_probs, CLASSES)
    selected_policy = choose_policy(y, best_probs)
    confidence = np.max(best_probs, axis=1)
    margin = np.sort(best_probs, axis=1)[:, -1] - np.sort(best_probs, axis=1)[:, -2]

    training = {
        "trained": True,
        "model_id": MODEL_ID,
        "selected_model": best_name,
        "cv": {
            "name": cv_name,
            "splits": n_splits,
            "group_source": "manifest participant_id inferred from near-duplicate clusters",
        },
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "feature_count": int(X.shape[1]),
        "comparisons": comparisons,
        "selected_forced_3class_metrics": selected_forced,
        "selected_abstention_policy_oof": selected_policy,
        "confident_wrong_review": false_prediction_review(details, y, best_probs, selected_policy),
        "confidence_distribution": {
            "min": round(float(np.min(confidence)), 4),
            "p25": round(float(np.quantile(confidence, 0.25)), 4),
            "median": round(float(np.median(confidence)), 4),
            "p75": round(float(np.quantile(confidence, 0.75)), 4),
            "max": round(float(np.max(confidence)), 4),
            "margin_median": round(float(np.median(margin)), 4),
        },
        "feature_extraction_rows": details,
    }

    bundle = {
        "model_id": MODEL_ID,
        "status": "research_candidate",
        "model_type": best_name,
        "pipeline": final_model,
        "feature_contract": "hirschberg_pair_features_v0.5",
        "base_feature_count": len(load_feature_names()),
        "feature_names": pair_feature_names(),
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "dataset_version": DATASET_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "metrics": {
            **selected_forced,
            "abstention_policy_oof": selected_policy,
        },
        "decision_policy": {
            "type": "confidence_margin_abstention",
            "abstain_status": "INCONCLUSIVE",
            "threshold": selected_policy["threshold"],
            "margin_threshold": selected_policy["margin_threshold"],
            "source": "out_of_fold_tuning_on_small_pedseye_manifest",
        },
        "is_production": False,
        "non_clinical_declaration": (
            "Research-only Hirschberg screening candidate. Returns INCONCLUSIVE when model evidence is weak. "
            "Not a diagnosis, not a clinical probability, and not suitable for clearance."
        ),
        "literature_notes": [
            "Recent Hirschberg/computer-vision papers favor corneal reflex, iris/pupil, and eye-landmark features with participant-level validation.",
            "The public Roboflow 5-class strabismus dataset was identified, but not imported because this run has no Roboflow export/API key.",
        ],
        "limitations": [
            "Only 67 local Pedseye-labeled images were available.",
            "participant_id values are inferred from near-duplicate clustering, not real patient IDs.",
            "pseudostrabismus, poor_quality, hypertropia, and hypotropia classes are absent from local training.",
            "The model must be treated as a conservative screening aid with manual review.",
        ],
    }
    return training, bundle


def write_model_card(path: Path, report: Dict[str, Any]) -> None:
    training = report["training"]
    forced = training["selected_forced_3class_metrics"]
    policy = training["selected_abstention_policy_oof"]
    binary = forced["binary_strabismus_vs_normal"]
    text = f"""# {MODEL_ID} Model Card

Status: `research_candidate`

This candidate is wired for Hirschberg research screening only. It must not be
presented as diagnosis, clinical probability, or clearance.

## Why v0.5 Exists

`hirschberg_candidate_v0.3_pedseye` produced unsafe label flips on the local
Pedseye set. v0.5 keeps the same 73-feature service contract but adds a
confidence/margin abstention policy so weak cases return `INCONCLUSIVE`.

## Dataset

- Manifest: `{report['manifest_path']}`
- Dataset version: `{DATASET_VERSION}`
- Total trainable samples: `{report['manifest_validation']['total_trainable_rows']}`
- Class counts: `{report['manifest_validation']['label_counts']}`
- Participant IDs: inferred near-duplicate groups, not real patient IDs.
- Missing expected classes: `{report['manifest_validation']['missing_expected_labels']}`

## Feature Contract

- Feature count: `{training['feature_count']}`
- Extractor: `app.services.hirschberg_ai_service.extract_features_from_crop`
- Crop aggregation: mean feature vector across service-compatible eye crops.
- Classes: `{CLASSES}`

## Selected Model

- Model family: `{training['selected_model']}`
- CV: `{training['cv']}`
- Forced 3-class balanced accuracy: `{forced['balanced_accuracy']}`
- Forced 3-class macro-F1: `{forced['macro_f1']}`
- Forced binary sensitivity: `{binary['sensitivity']}`
- Forced binary specificity: `{binary['specificity']}`

## Conservative Decision Policy

- Policy: `confidence_margin_abstention`
- Confidence threshold: `{policy['threshold']}`
- Margin threshold: `{policy['margin_threshold']}`
- OOF coverage: `{policy['coverage']}`
- OOF covered accuracy: `{policy['covered_accuracy']}`
- OOF confident wrong predictions: `{policy['wrong_confident']}`
- OOF abstained count: `{policy['abstained_count']}`

## Research References Considered

- Corneal light-reflection/Hirschberg systems emphasize reflex/iris/pupil geometry and participant-level validation.
- Recent mobile-photo screening work supports feature-based Random Forest style classifiers over small interpretable geometry features.
- Public Roboflow 5-class strabismus data was identified as a future import candidate, but was not imported in this run.

## Deployment Recommendation

Deploy only as `research_candidate` with `INCONCLUSIVE` behavior enabled. Do not
use the output as a medical diagnosis. More clinician-confirmed images,
pseudostrabismus examples, poor-quality examples, and real participant IDs are
required before removing the abstention guard.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train conservative Hirschberg candidate v0.5.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--model-card-output", type=Path, default=DEFAULT_CARD_OUTPUT)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.manifest) if row.get("label") in set(CLASSES)]
    validation = validate_manifest_for_training(rows)
    validation["total_trainable_rows"] = len(rows)
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

    counts = Counter(row["label"] for row in rows)
    if min(counts.values()) < 10:
        report = {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED_TOO_FEW_SAMPLES",
            "manifest_path": str(args.manifest),
            "manifest_validation": validation,
            "training": {"trained": False, "class_counts": dict(counts)},
        }
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "class_counts": dict(counts)}, ensure_ascii=False, indent=2))
        return 1

    training, bundle = train(rows)
    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_RESEARCH_CANDIDATE_WITH_ABSTENTION",
        "manifest_path": str(args.manifest),
        "model_output": str(args.model_output),
        "model_card_output": str(args.model_card_output),
        "seed": SEED,
        "manifest_validation": validation,
        "training": training,
        "deployment_recommendation": {
            "deploy": True,
            "scope": "research_candidate_screening_only",
            "reason": "v0.5 reduces forced label flips by returning INCONCLUSIVE for weak evidence.",
            "must_keep_warning": True,
        },
        "governance": [
            "Production endpoint /api/v1/strabismus/predict remains unchanged.",
            "Existing v0.1 and v0.3 artifacts were not overwritten.",
            "Output is sàng lọc nghiên cứu only, not diagnosis.",
        ],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_model_card(args.model_card_output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_model": training["selected_model"],
                "forced_balanced_accuracy": training["selected_forced_3class_metrics"]["balanced_accuracy"],
                "forced_macro_f1": training["selected_forced_3class_metrics"]["macro_f1"],
                "abstention_policy": training["selected_abstention_policy_oof"],
                "model_output": str(args.model_output),
                "eval_output": str(args.eval_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
