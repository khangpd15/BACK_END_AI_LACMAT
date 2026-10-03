"""Phase 6B: Train exploratory Hirschberg research classifier.

Trained on folder-labeled 224x224 ocular crops from `hirschberg_folder_labels.jsonl`.
Strictly marked as `research_candidate` for exploratory research.
DOES NOT alter production endpoints or replace production models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
SCRIPTS_DIR = CURRENT_FILE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app.services.research_measurement_service import detect_pupil_in_roi, detect_reflexes_in_roi
from benchmark_hirschberg_detectors import estimate_iris_center_in_crop, resolve_image_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("remicare.train_hirschberg")

DEFAULT_SEED = 20261003
DEFAULT_MANIFEST = Path("D:/AI_Check_Lac/manifests/hirschberg_folder_labels.jsonl")
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app" / "models" / "research" / "hirschberg_candidate_v0.1.joblib"
DEFAULT_EVAL_OUTPUT = Path("D:/AI_Check_Lac/manifests/hirschberg_candidate_eval.json")
DEFAULT_FEATURE_CACHE = Path("D:/AI_Check_Lac/manifests/hirschberg_extracted_features.npz")

FEATURE_NAMES = [
    # 1. Iris geometry (3)
    "iris_cx_n",
    "iris_cy_n",
    "iris_d_n",
    # 2. Corneal reflex detection and metrics (5)
    "reflex_found",
    "reflex_cx_n",
    "reflex_cy_n",
    "reflex_area_n",
    "reflex_tier_val",
    # 3. Pupil detection and metrics (4)
    "pupil_found",
    "pupil_cx_n",
    "pupil_cy_n",
    "pupil_diameter_n",
    # 4. Hirschberg displacement vectors (5)
    "both_detected",
    "h_iris_x",
    "h_iris_y",
    "h_pupil_x",
    "h_pupil_y",
    # 5. Ocular dark centroid and displacement (3)
    "dark_centroid_x",
    "dark_centroid_y",
    "dark_cx_offset",
    # 6. Asymmetry (2)
    "lr_ratio",
    "lr_diff",
    # 7. Horizontal profile (16)
    *[f"x_profile_{i:02d}" for i in range(16)],
    # 8. Spatial grid 4x4 (32)
    *[f"cell_{r}_{c}_{stat}" for r in range(4) for c in range(4) for stat in ["mean", "std"]],
    # 9. Frozen ResNet18 bilateral ROI representation (3)
    "resnet_logit_0",
    "resnet_logit_1",
    "resnet_strabismus_prob",
]

CLASS_NAMES = ["esotropia", "exotropia", "normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def extract_features_from_image(
    bgr: np.ndarray,
    onnx_sess: ort.InferenceSession,
) -> List[float]:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Iris estimation
    cx, cy, diameter = estimate_iris_center_in_crop(rgb)
    iris_cx_n = float(cx / w)
    iris_cy_n = float(cy / h)
    iris_d_n = float(diameter / max(w, h))

    # 2. Reflex detection
    reflexes, r_status, r_tier = detect_reflexes_in_roi(rgb, cx, cy, diameter)
    r_found = 1.0 if len(reflexes) > 0 else 0.0
    if len(reflexes) > 0:
        best_r = min(reflexes, key=lambda c: (c["x"] - cx)**2 + (c["y"] - cy)**2)
        rcx_n = float(best_r["x"]) / w
        rcy_n = float(best_r["y"]) / h
        r_area_n = float(best_r.get("pixel_count", 1)) / (w * h)
    else:
        rcx_n, rcy_n, r_area_n = 0.5, 0.5, 0.0

    tier_map = {"strict_specular": 3.0, "medium_specular": 2.0, "adaptive_specular": 1.0}
    r_tier_val = tier_map.get(r_tier, 0.0)

    # 3. Pupil detection
    pupil_c, pupil_d, p_status = detect_pupil_in_roi(rgb, cx, cy, diameter)
    p_found = 1.0 if p_status == "DETECTED" and pupil_c is not None else 0.0
    if p_found > 0.5 and pupil_c is not None:
        pcx_n = float(pupil_c["x"]) / w
        pcy_n = float(pupil_c["y"]) / h
        pd_n = float(pupil_d) / max(w, h)
    else:
        pcx_n, pcy_n, pd_n = iris_cx_n, iris_cy_n, iris_d_n * 0.4

    # 4. Hirschberg vectors
    both_detected = 1.0 if (r_found > 0.5 and p_found > 0.5) else 0.0
    h_iris_x = (rcx_n - iris_cx_n) / max(0.01, iris_d_n) if r_found > 0.5 else 0.0
    h_iris_y = (rcy_n - iris_cy_n) / max(0.01, iris_d_n) if r_found > 0.5 else 0.0
    h_pupil_x = (rcx_n - pcx_n) / max(0.01, iris_d_n) if both_detected > 0.5 else 0.0
    h_pupil_y = (rcy_n - pcy_n) / max(0.01, iris_d_n) if both_detected > 0.5 else 0.0

    # 5. Ocular dark centroid
    dark = 255.0 - gray.astype(np.float32)
    tot_dark = float(np.sum(dark))
    if tot_dark > 1e-4:
        yy, xx = np.indices(gray.shape)
        dark_cx = float(np.sum(xx * dark) / (tot_dark * w))
        dark_cy = float(np.sum(yy * dark) / (tot_dark * h))
    else:
        dark_cx, dark_cy = 0.5, 0.5
    dark_cx_offset = dark_cx - 0.5

    # 6. Asymmetry
    col_means = np.mean(gray, axis=0) / 255.0
    left_mean = float(np.mean(col_means[: int(w * 0.4)]))
    right_mean = float(np.mean(col_means[int(w * 0.6) :]))
    lr_ratio = left_mean / max(0.01, right_mean)
    lr_diff = left_mean - right_mean

    # 7. Horizontal profile (16 bins)
    bin_size = w // 16
    x_profile = [float(np.mean(col_means[i * bin_size : (i + 1) * bin_size])) for i in range(16)]

    # 8. Spatial grid (4x4 = 16 cells x 2 stats = 32)
    spatial_feats = []
    gh, gw = h // 4, w // 4
    for gi in range(4):
        for gj in range(4):
            cell = gray[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw].astype(np.float32) / 255.0
            spatial_feats.extend([float(np.mean(cell)), float(np.std(cell))])

    # 9. ResNet logits
    rgb_norm = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    tensor = ((rgb_norm.transpose(2, 0, 1)[np.newaxis, ...] - mean) / std).astype(np.float32)
    logits = onnx_sess.run(None, {"input_frame": tensor})[0][0]
    l0, l1 = float(logits[0]), float(logits[1])
    exp_l = np.exp([l0, l1] - np.max([l0, l1]))
    prob_strab = float(exp_l[1] / np.sum(exp_l))

    feats = [
        iris_cx_n,
        iris_cy_n,
        iris_d_n,
        r_found,
        rcx_n,
        rcy_n,
        r_area_n,
        r_tier_val,
        p_found,
        pcx_n,
        pcy_n,
        pd_n,
        both_detected,
        h_iris_x,
        h_iris_y,
        h_pupil_x,
        h_pupil_y,
        dark_cx,
        dark_cy,
        dark_cx_offset,
        lr_ratio,
        lr_diff,
        *x_profile,
        *spatial_feats,
        l0,
        l1,
        prob_strab,
    ]
    return feats


def load_dataset(
    manifest_path: Path,
    cache_path: Path | None = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    if cache_path and cache_path.is_file() and not force_recompute:
        logger.info(f"Loading cached features from {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        return {
            "X_train": cached["X_train"],
            "y_train": cached["y_train"],
            "X_val": cached["X_val"],
            "y_val": cached["y_val"],
            "X_test": cached["X_test"],
            "y_test": cached["y_test"],
            "groups_train": cached["groups_train"],
            "groups_val": cached["groups_val"],
            "groups_test": cached["groups_test"],
            "feature_names": cached["feature_names"].tolist(),
        }

    logger.info(f"Extracting features from manifest {manifest_path}...")
    onnx_path = PROJECT_ROOT / "app" / "models" / "best_model.onnx"
    onnx_sess = ort.InferenceSession(str(onnx_path))

    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    splits: Dict[str, Dict[str, List[Any]]] = {
        "train": {"X": [], "y": [], "groups": []},
        "val": {"X": [], "y": [], "groups": []},
        "test": {"X": [], "y": [], "groups": []},
    }

    t0 = time.time()
    for idx, rec in enumerate(records, 1):
        split = rec.get("generated_split", "train")
        if split not in splits:
            split = "train"

        img_path = resolve_image_path(rec.get("absolute_path") or rec.get("relative_path"))
        if not img_path:
            logger.warning(f"Image missing: {rec.get('relative_path')}")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        feats = extract_features_from_image(bgr, onnx_sess)
        label_str = rec.get("class_label", "normal")
        label_idx = CLASS_TO_IDX.get(label_str, 2)
        group_id = rec.get("group_id", f"group_{idx}")

        splits[split]["X"].append(feats)
        splits[split]["y"].append(label_idx)
        splits[split]["groups"].append(group_id)

        if idx % 100 == 0 or idx == len(records):
            elapsed = time.time() - t0
            logger.info(f"Extracted {idx}/{len(records)} images in {elapsed:.1f}s")

    dataset = {
        "X_train": np.array(splits["train"]["X"], dtype=np.float32),
        "y_train": np.array(splits["train"]["y"], dtype=np.int64),
        "X_val": np.array(splits["val"]["X"], dtype=np.float32),
        "y_val": np.array(splits["val"]["y"], dtype=np.int64),
        "X_test": np.array(splits["test"]["X"], dtype=np.float32),
        "y_test": np.array(splits["test"]["y"], dtype=np.int64),
        "groups_train": np.array(splits["train"]["groups"]),
        "groups_val": np.array(splits["val"]["groups"]),
        "groups_test": np.array(splits["test"]["groups"]),
        "feature_names": FEATURE_NAMES,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
        logger.info(f"Saved feature cache to {cache_path}")

    return dataset


def train_and_evaluate(
    dataset: Dict[str, Any],
    seed: int = DEFAULT_SEED,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]

    candidates = {
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(
                random_state=seed,
                class_weight="balanced",
                max_iter=150,
                learning_rate=0.08,
                min_samples_leaf=10,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                random_state=seed,
                class_weight="balanced",
                n_estimators=150,
                max_depth=10,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=seed,
                class_weight="balanced",
                max_iter=1000,
                C=0.5,
            )),
        ]),
        "MLPClassifier": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=400,
                random_state=seed,
                early_stopping=True,
            )),
        ]),
    }

    validation_results = {}
    best_name = None
    best_val_score = -1.0
    best_pipeline = None

    logger.info("--- EVALUATING MODEL CANDIDATES ON VALIDATION SET ---")
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        val_preds = pipe.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_bal_acc = float(balanced_accuracy_score(y_val, val_preds))
        val_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

        validation_results[name] = {
            "val_accuracy": round(val_acc, 4),
            "val_balanced_accuracy": round(val_bal_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
        }
        logger.info(f"{name}: Acc={val_acc:.4f}, BalAcc={val_bal_acc:.4f}, MacroF1={val_macro_f1:.4f}")

        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_name = name
            best_pipeline = pipe

    logger.info(f"Best candidate selected: {best_name} with Val BalAcc = {best_val_score:.4f}")

    # Final evaluation on held-out TEST set
    test_preds = best_pipeline.predict(X_test)
    test_probs = best_pipeline.predict_proba(X_test)

    test_acc = float(accuracy_score(y_test, test_preds))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    cm = confusion_matrix(y_test, test_preds).tolist()
    clf_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    # Binary metrics: Normal (class 2) vs Strabismus (classes 0 or 1)
    y_test_binary = (y_test != 2).astype(int)  # 1 = Strabismus, 0 = Normal
    test_preds_binary = (test_preds != 2).astype(int)
    prob_strabismus = 1.0 - test_probs[:, 2]

    tn, fp, fn, tp = confusion_matrix(y_test_binary, test_preds_binary).ravel()
    binary_sensitivity = float(tp / max(1, (tp + fn)))
    binary_specificity = float(tn / max(1, (tn + fp)))
    binary_ppv = float(tp / max(1, (tp + fp)))
    binary_npv = float(tn / max(1, (tn + fn)))
    try:
        binary_roc_auc = float(roc_auc_score(y_test_binary, prob_strabismus))
    except Exception:
        binary_roc_auc = None

    test_metrics = {
        "model_name": best_name,
        "seed": seed,
        "test_accuracy": round(test_acc, 4),
        "test_balanced_accuracy": round(test_bal_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "confusion_matrix_3class": cm,
        "classification_report": clf_report,
        "binary_strabismus_vs_normal": {
            "sensitivity": round(binary_sensitivity, 4),
            "specificity": round(binary_specificity, 4),
            "ppv": round(binary_ppv, 4),
            "npv": round(binary_npv, 4),
            "roc_auc": round(binary_roc_auc, 4) if binary_roc_auc is not None else None,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }

    return best_pipeline, validation_results, test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6B Hirschberg Exploratory Classifier Trainer")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--cache-features", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    dataset = load_dataset(
        args.manifest,
        cache_path=args.cache_features,
        force_recompute=args.force_recompute,
    )

    best_pipeline, val_results, test_metrics = train_and_evaluate(dataset, seed=args.seed)

    # Save artifact
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_id": "hirschberg-candidate-v0.1",
        "status": "research_candidate",
        "model_type": test_metrics["model_name"],
        "pipeline": best_pipeline,
        "feature_names": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "dataset_version": "hirschberg-folder-labels-v0.1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "metrics": test_metrics,
        "is_production": False,
        "non_clinical_declaration": (
            "Exploratory research model trained on folder-labeled 224x224 crops. "
            "Not for clinical diagnosis. Not deployed to production."
        ),
    }
    joblib.dump(bundle, args.model_output)
    logger.info(f"Saved research candidate bundle to {args.model_output}")

    # Save eval report
    eval_report = {
        "phase": "Phase 6B Exploratory Hirschberg Classifier Evaluation",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_RESEARCH_CANDIDATE",
        "datasetVersion": "hirschberg-folder-labels-v0.1",
        "manifestPath": str(args.manifest),
        "artifactPath": str(args.model_output),
        "seed": args.seed,
        "modelFamily": test_metrics["model_name"],
        "validationComparison": val_results,
        "testMetrics": test_metrics,
        "governanceDeclarations": [
            "Status is strictly research_candidate, NOT deployed to production.",
            "Production endpoint /api/v1/strabismus/predict remains unchanged.",
            "Trained on 224x224 folder-labeled crops with unverified participant IDs.",
            "Not for clinical diagnosis or clinical angle determination.",
        ],
    }
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved evaluation report to {args.eval_output}")

    logger.info("=== PHASE 6B TRAINING AND EVALUATION COMPLETED SUCCESSFULLY ===")
    logger.info(f"Test Accuracy: {test_metrics['test_accuracy']}")
    logger.info(f"Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']}")
    logger.info(f"Test Macro F1: {test_metrics['test_macro_f1']}")
    logger.info(f"Binary Sensitivity: {test_metrics['binary_strabismus_vs_normal']['sensitivity']}")
    logger.info(f"Binary Specificity: {test_metrics['binary_strabismus_vs_normal']['specificity']}")
    logger.info(f"Binary ROC-AUC: {test_metrics['binary_strabismus_vs_normal']['roc_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
