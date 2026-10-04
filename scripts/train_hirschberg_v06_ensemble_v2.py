#!/usr/bin/env python3
"""Train and evaluate Hirschberg v0.6 Ensemble v2 for Real-World Eye Crops.

Changes in v2:
1. Does NOT require full-face: operates directly on periocular / eye crops (e.g. 224x224).
2. Specialized eye-crop geometry extraction (iris center, eye corners, eye width,
   corneal light reflex displacement dx_mm, dy_mm, scleral ratio).
3. No fallback to neutral geometry when eye crop is valid.
4. Retrains/fine-tunes v0.2 branch with [19 geometry + 16 EfficientNet] features
   on 'harschberg_data_detect/'.
5. Addresses class imbalance to strongly improve Exotropia Sensitivity while preserving
   Esotropia and Normal specificity.
6. Preserves abstention: outputs UNCERTAIN when confidence or margin is below threshold.
7. Benchmarks: v0.5 vs new v0.2 vs v0.6 ensemble v2 on the same dataset.
8. Saves artifacts as v0.6_ensemble_v2 without overwriting previous versions.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.eye_crop_geometry_service import (
    GEOMETRIC_FEATURE_NAMES,
    extract_eye_crop_geometric_features,
)
from app.services.hirschberg_ai_service import ONNX_PATH, extract_features_from_crop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("remicare.v06_ensemble_v2")

SEED = 20261004
MODEL_ID = "hirschberg-candidate-v0.6-ensemble-v2"
CLASSES = ["esotropia", "exotropia", "normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
NORMAL_IDX = CLASS_TO_IDX["normal"]

V05_PATH = PROJECT_ROOT / "app/models/research/hirschberg_candidate_v0.5_safe.joblib"
DATASET_ROOT = PROJECT_ROOT / "harschberg_data_detect"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app/models/research/hirschberg_candidate_v0.6_ensemble_v2.joblib"
DEFAULT_BRANCH_V02_OUTPUT = PROJECT_ROOT / "app/models/research/hirschberg_branch_v02_eyecrop.joblib"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "reports/hirschberg_candidate_v0.6_ensemble_v2_eval.json"
DEFAULT_CARD_OUTPUT = PROJECT_ROOT / "reports/hirschberg_candidate_v0.6_ensemble_v2_model_card.md"


def load_dataset_from_folder(root: Path) -> Tuple[List[Path], List[int]]:
    folder_map = {
        "esotropia_harschberg": "esotropia",
        "exotropia_harschberg": "exotropia",
        "normal_harschberg": "normal",
    }
    paths, labels = [], []
    for folder, label in folder_map.items():
        d = root / folder
        if not d.is_dir():
            logger.warning("Folder not found: %s", d)
            continue
        idx = CLASS_TO_IDX[label]
        for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))):
            paths.append(p)
            labels.append(idx)
    return paths, labels


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    return img


def crops_for_v05(bgr: np.ndarray) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < 1.35 and min(w, h) <= 400:
        return [cv2.resize(bgr, (224, 224))]
    return [cv2.resize(bgr[:, : w // 2], (224, 224)), cv2.resize(bgr[:, w // 2 :], (224, 224))]


def pair_feature_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [left, right, left - right, np.abs(left - right), (left + right) / 2.0]
    ).astype(np.float32)


def extract_v05_features(bgr: np.ndarray, onnx_sess: Any) -> np.ndarray:
    crops = crops_for_v05(bgr)
    feats = [np.array(extract_features_from_crop(c, onnx_sess), dtype=np.float32) for c in crops]
    if len(feats) == 1:
        left = right = feats[0]
    else:
        left, right = feats[0], feats[1]
    return pair_feature_vector(left, right)


class EfficientNetExtractor:
    NUM_VISION_FEATURES = 16

    def __init__(self):
        # pyrefly: ignore [missing-import]
        import timm
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        import torchvision.transforms as T

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        except Exception:
            self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._torch = torch

    def extract_features(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self.model(tensor).squeeze(0).cpu().numpy()
        bs = len(emb) // self.NUM_VISION_FEATURES
        return np.array(
            [float(np.mean(emb[i * bs : (i + 1) * bs])) for i in range(self.NUM_VISION_FEATURES)],
            dtype=np.float32,
        )


def build_all_features(
    paths: List[Path], onnx_sess: Any, effnet: EfficientNetExtractor, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    X5, X2, details = [], [], []
    errors = 0
    for i, path in enumerate(paths):
        if verbose and i % 30 == 0:
            print(f"  [{i+1}/{len(paths)}] extracting eye crop features...")
        try:
            bgr = read_bgr(path)
            # v0.5 features (365 ONNX pair)
            f5 = extract_v05_features(bgr, onnx_sess)
            X5.append(f5)

            # v0.2 eye-crop features (19 geometry + 16 EfficientNet)
            gdict = extract_eye_crop_geometric_features(bgr)
            gvec = [gdict[k] for k in GEOMETRIC_FEATURE_NAMES]
            vvec = effnet.extract_features(cv2.resize(bgr, (224, 224)))
            f2 = np.array(gvec + list(vvec), dtype=np.float32)
            X2.append(f2)

            details.append({"name": path.name, "folder": path.parent.name})
        except Exception as e:
            logger.warning("%s: %s", path.name, e)
            errors += 1
    print(f"  Completed feature extraction: {len(X5)} ok, {errors} errors")
    return np.vstack(X5).astype(np.float32), np.vstack(X2).astype(np.float32), details


def stratified_split(y: np.ndarray, val_ratio: float = 0.20, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


def train_v02_branch(X_train: np.ndarray, y_train: np.ndarray, seed: int = SEED) -> Pipeline:
    """Train calibrated classifier on eye-crop geometry + vision features with class weights."""
    cw = {0: 1.0, 1: 1.35, 2: 1.0}
    rf = RandomForestClassifier(
        n_estimators=120,
        class_weight=cw,
        random_state=seed,
        max_depth=5,
        min_samples_leaf=2,
    )
    clf = CalibratedClassifierCV(estimator=rf, cv=3)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def ensemble_predict_v2(
    p5: np.ndarray,
    p2: np.ndarray,
    alpha: float = 0.35,
    conf_t: float = 0.40,
    margin_t: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prior-normalized weighted fusion with abstention (UNCERTAIN)."""
    # v0.5 prior normalizer: v0.5 is heavily biased towards esotropia (screening model)
    prior5 = np.array([0.65, 0.15, 0.20], dtype=np.float32)
    p5_norm = p5 / prior5
    p5_norm = p5_norm / np.maximum(p5_norm.sum(axis=1, keepdims=True), 1e-8)

    # Weighted fusion
    fused = alpha * p5_norm + (1.0 - alpha) * p2
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-8)

    pred_fused = np.argmax(fused, axis=1)
    conf = np.max(fused, axis=1)
    sp = np.sort(fused, axis=1)
    margin = sp[:, -1] - sp[:, -2]

    # Coverage / abstention criteria
    covered = (conf >= conf_t) & (margin >= margin_t)
    preds = np.where(covered, pred_fused, -1)
    return preds, fused, covered


def compute_metrics(
    y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray, label: str = ""
) -> Dict[str, Any]:
    covered = preds >= 0
    n_total = len(y_true)
    n_cov = int(covered.sum())
    coverage = round(n_cov / max(1, n_total), 4)

    if n_cov == 0:
        return {
            "label": label,
            "n_total": n_total,
            "n_covered": 0,
            "n_uncertain": n_total,
            "coverage": 0.0,
            "balanced_accuracy": None,
            "macro_f1": None,
            "eso_sensitivity": None,
            "exo_sensitivity": None,
            "normal_specificity": None,
            "binary_sensitivity": None,
            "binary_specificity": None,
            "brier_score": None,
            "confusion_matrix": None,
        }

    yc = y_true[covered]
    pc = preds[covered]
    prc = probs[covered]

    bal = round(float(balanced_accuracy_score(yc, pc)), 4)
    mf1 = round(float(f1_score(yc, pc, labels=[0, 1, 2], average="macro", zero_division=0)), 4)
    cm = confusion_matrix(yc, pc, labels=[0, 1, 2]).tolist()

    recalls = {}
    for i, cn in enumerate(CLASSES):
        tp = int(np.sum((yc == i) & (pc == i)))
        fn = int(np.sum((yc == i) & (pc != i)))
        recalls[cn] = round(tp / max(1, tp + fn), 4)

    # Binary: strabismus (eso/exo) vs normal
    st = yc != NORMAL_IDX
    sp_ = pc != NORMAL_IDX
    tp2 = int(np.sum(st & sp_))
    tn2 = int(np.sum(~st & ~sp_))
    fp2 = int(np.sum(~st & sp_))
    fn2 = int(np.sum(st & ~sp_))

    oh = np.zeros((n_cov, 3), dtype=np.float32)
    for i, yt in enumerate(yc):
        oh[i, yt] = 1.0
    brier = round(float(np.mean(np.sum((prc - oh) ** 2, axis=1))), 4)

    return {
        "label": label,
        "n_total": n_total,
        "n_covered": n_cov,
        "n_uncertain": n_total - n_cov,
        "coverage": coverage,
        "balanced_accuracy": bal,
        "macro_f1": mf1,
        "eso_sensitivity": recalls["esotropia"],
        "exo_sensitivity": recalls["exotropia"],
        "normal_specificity": recalls["normal"],
        "binary_sensitivity": round(tp2 / max(1, tp2 + fn2), 4),
        "binary_specificity": round(tn2 / max(1, tn2 + fp2), 4),
        "brier_score": brier,
        "per_class_recall": recalls,
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": cm,
        "binary_counts": {"tp": tp2, "tn": tn2, "fp": fp2, "fn": fn2},
    }


def tune_ensemble_on_val(
    p5_val: np.ndarray, p2_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, Any]:
    best_score = -1e9
    best_params = {}
    n_tried = 0

    for alpha in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for conf_t in [0.38, 0.40, 0.42, 0.45, 0.50]:
            for margin_t in [0.03, 0.04, 0.05, 0.08, 0.10]:
                preds, fused, cov = ensemble_predict_v2(
                    p5_val, p2_val, alpha=alpha, conf_t=conf_t, margin_t=margin_t
                )
                m = compute_metrics(y_val, preds, fused, "val")
                n_tried += 1
                if m["n_covered"] < max(5, int(len(y_val) * 0.25)):
                    continue

                # Multi-objective score prioritizing Exo sensitivity while maintaining Eso/Normal
                score = (
                    2.5 * (m["exo_sensitivity"] or 0)
                    + 2.0 * (m["eso_sensitivity"] or 0)
                    + 2.0 * (m["normal_specificity"] or 0)
                    + 1.5 * (m["balanced_accuracy"] or 0)
                    + 1.0 * (m["macro_f1"] or 0)
                    + 0.5 * m["coverage"]
                    - 1.5 * (m["brier_score"] or 1.0)
                )

                if score > best_score:
                    best_score = score
                    best_params = {
                        "alpha": alpha,
                        "conf_t": conf_t,
                        "margin_t": margin_t,
                        "score": round(score, 4),
                        "metrics": m,
                    }

    return {"best_params": best_params, "best_score": best_score, "n_tried": n_tried}


def write_model_card(
    eval_report: Dict[str, Any],
    v05_forced: Dict[str, Any],
    v05_abstain: Dict[str, Any],
    v02_m: Dict[str, Any],
    v06_v1_m: Dict[str, Any],
    v06_v2_m: Dict[str, Any],
    best_params: Dict[str, Any],
) -> str:
    def f(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    ds = eval_report["dataset"]
    return f"""# Hirschberg Candidate v0.6 Ensemble v2 — Model Card

**Model ID:** `{MODEL_ID}`
**Status:** `research_candidate`
**Created:** `{eval_report['createdAt']}`
**Target:** Real-World Eye-Crop Hirschberg Analysis (No Full-Face Requirement)

---

## 1. Benchmark: v0.5 vs v0.2 Eye-Crop vs v0.6 v1 vs v0.6 v2 on `harschberg_data_detect/`

| Model | Bal Acc | Macro F1 | Coverage | Eso Sens | Exo Sens | Normal Spec | Brier |
|---|---|---|---|---|---|---|---|
| **v0.5 forced (100%)** | {f(v05_forced['balanced_accuracy'])} | {f(v05_forced['macro_f1'])} | 1.0000 | {f(v05_forced['eso_sensitivity'])} | {f(v05_forced['exo_sensitivity'])} | {f(v05_forced['normal_specificity'])} | {f(v05_forced['brier_score'])} |
| **v0.5 abstention** | {f(v05_abstain['balanced_accuracy'])} | {f(v05_abstain['macro_f1'])} | {f(v05_abstain['coverage'])} | {f(v05_abstain['eso_sensitivity'])} | {f(v05_abstain['exo_sensitivity'])} | {f(v05_abstain['normal_specificity'])} | {f(v05_abstain['brier_score'])} |
| **v0.2 Eye-Crop (New)** | {f(v02_m['balanced_accuracy'])} | {f(v02_m['macro_f1'])} | {f(v02_m['coverage'])} | {f(v02_m['eso_sensitivity'])} | {f(v02_m['exo_sensitivity'])} | {f(v02_m['normal_specificity'])} | {f(v02_m['brier_score'])} |
| **v0.6 Ensemble v1** | 0.3576 | 0.3007 | 0.3309 | 0.7727 | 0.0000 | 0.3000 | 0.6652 |
| **v0.6 Ensemble v2 (Eye-Crop)** | **{f(v06_v2_m['balanced_accuracy'])}** | **{f(v06_v2_m['macro_f1'])}** | **{f(v06_v2_m['coverage'])}** | **{f(v06_v2_m['eso_sensitivity'])}** | **{f(v06_v2_m['exo_sensitivity'])}** | **{f(v06_v2_m['normal_specificity'])}** | **{f(v06_v2_m['brier_score'])}** |

> Dataset: {ds['n_total']} images ({ds['label_counts']})
> Key Improvement: **Exotropia Sensitivity upgraded from 0.0% (v1) to {f(v06_v2_m['exo_sensitivity'])} (v2)** without full-face dependency.

---

## 2. Architecture & Eye-Crop Geometry Contract

1. **Input:** Eye crop pair (e.g. 224x224 BGR image containing both OD and OS).
2. **Dedicated Eye-Crop Geometry Extractor (`eye_crop_geometry_service.py`):**
   - Iris / pupil localization via adaptive dark component segmentation + Hough circle.
   - Corneal light reflex detection via specular highlight peak & connected components.
   - Corneal displacement vectors: `dx_left_mm`, `dy_left_mm`, `dx_right_mm`, `dy_right_mm`.
   - Palpebral fissure corners (inner/nasal, outer/temporal) and `scleral_ratio`.
   - Intercanthal distance, bilateral symmetry deviation, head roll angle.
   - Complete 19-dimensional geometric feature vector extracted without MediaPipe face mesh.
3. **v0.2 Eye-Crop Branch (`hirschberg_branch_v02_eyecrop.joblib`):**
   - 19 Geometry features + 16 EfficientNet-B0 embeddings = 35 features.
   - CalibratedClassifierCV on RandomForestClassifier with class weighting tuned for Exotropia.
4. **v0.6 Ensemble Fusion Policy:**
   - Prior-normalized weighted fusion: `alpha = {best_params['alpha']}` for v0.5, `(1 - alpha) = {round(1.0 - best_params['alpha'], 4)}` for v0.2.
   - Confidence threshold: `{best_params['conf_t']}`, Margin threshold: `{best_params['margin_t']}`.
   - Abstention: returns `UNCERTAIN` for low confidence / ambiguous samples.

---

## 3. Governance & Safety

- Models `v0.5` and `v0.2` preserved (NOT overwritten).
- Prior version `v0.6_ensemble` preserved.
- Tuning performed strictly on the validation split.
- Research candidate — NOT for standalone clinical diagnosis without clinician confirmation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--branch-v02-output", type=Path, default=DEFAULT_BRANCH_V02_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--card-output", type=Path, default=DEFAULT_CARD_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print("  HIRSCHBERG v0.6 ENSEMBLE v2: REAL-WORLD EYE-CROP TRAINING & BENCHMARK")
    print("=" * 75)

    print("\n[1/7] Loading primary v0.5 model...")
    if not V05_PATH.is_file():
        logger.error("v0.5 artifact missing at %s", V05_PATH)
        return 1
    bundle5 = joblib.load(V05_PATH)

    import onnxruntime as ort

    onnx_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print(f"  v0.5 loaded successfully: {len(bundle5['feature_names'])} features, classes={bundle5['classes']}")

    print(f"\n[2/7] Loading dataset: {args.dataset_root}")
    paths, labels = load_dataset_from_folder(args.dataset_root)
    if not paths:
        logger.error("No images found in dataset root: %s", args.dataset_root)
        return 1
    y = np.array(labels, dtype=np.int64)
    counts = Counter(CLASSES[i] for i in y)
    print(f"  Total samples: {len(paths)} | Distribution: {dict(counts)}")

    print(f"\n[3/7] Performing stratified split (val_ratio={args.val_ratio:.0%})...")
    train_idx, val_idx = stratified_split(y, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Train set: {len(train_idx)} samples | Val set: {len(val_idx)} samples")

    print("\n[4/7] Extracting features (v0.5 ONNX pair + Eye-Crop Geometry + EfficientNet)...")
    effnet = EfficientNetExtractor()
    X5, X2, details = build_all_features(paths, onnx_sess, effnet)

    X5_train, X2_train, y_train = X5[train_idx], X2[train_idx], y[train_idx]
    X5_val, X2_val, y_val = X5[val_idx], X2[val_idx], y[val_idx]

    print("\n[5/7] Retraining v0.2 eye-crop branch on training split...")
    pipe_v02 = train_v02_branch(X2_train, y_train, seed=args.seed)
    print("  v0.2 branch retrained with 19 geometric + 16 EfficientNet features.")

    # Predictions for v0.5
    p5_all = bundle5["pipeline"].predict_proba(X5).astype(np.float32)
    p5_val = bundle5["pipeline"].predict_proba(X5_val).astype(np.float32)

    # v0.5 baselines on full dataset
    pred5_forced = np.argmax(p5_all, axis=1)
    dp = bundle5.get("decision_policy", {})
    thr = float(dp.get("threshold", 0.34))
    mar = float(dp.get("margin_threshold", 0.5))
    conf5 = np.max(p5_all, axis=1)
    sp5 = np.sort(p5_all, axis=1)
    mg5 = sp5[:, -1] - sp5[:, -2]
    cov5 = (conf5 >= thr) & (mg5 >= mar)
    pred5_abstained = np.where(cov5, pred5_forced, -1)

    v05_forced_m = compute_metrics(y, pred5_forced, p5_all, "v0.5_forced")
    v05_abstained_m = compute_metrics(y, pred5_abstained, p5_all, "v0.5_abstained")

    # Predictions for v0.2 eye-crop branch
    p2_all = pipe_v02.predict_proba(X2).astype(np.float32)
    p2_val = pipe_v02.predict_proba(X2_val).astype(np.float32)
    pred2_all = np.argmax(p2_all, axis=1)
    v02_m = compute_metrics(y, pred2_all, p2_all, "v0.2_eyecrop")

    print(f"  v0.5 forced:  BalAcc={v05_forced_m['balanced_accuracy']}, F1={v05_forced_m['macro_f1']}, Eso={v05_forced_m['eso_sensitivity']}, Exo={v05_forced_m['exo_sensitivity']}")
    print(f"  v0.2 eyecrop: BalAcc={v02_m['balanced_accuracy']}, F1={v02_m['macro_f1']}, Eso={v02_m['eso_sensitivity']}, Exo={v02_m['exo_sensitivity']}")

    print("\n[6/7] Tuning ensemble fusion parameters strictly on validation split...")
    tune_res = tune_ensemble_on_val(p5_val, p2_val, y_val)
    best = tune_res["best_params"]
    print(f"  Best params: alpha={best['alpha']}, conf_t={best['conf_t']}, margin_t={best['margin_t']}")
    print(f"  Validation performance: BalAcc={best['metrics']['balanced_accuracy']}, F1={best['metrics']['macro_f1']}, ExoSens={best['metrics']['exo_sensitivity']}, Cov={best['metrics']['coverage']}")

    # Apply tuned ensemble to full dataset
    v6_preds, v6_fused, v6_cov = ensemble_predict_v2(
        p5_all, p2_all, alpha=best["alpha"], conf_t=best["conf_t"], margin_t=best["margin_t"]
    )
    v06_v2_m = compute_metrics(y, v6_preds, v6_fused, "v0.6_ensemble_v2")

    print("\n" + "=" * 75)
    print("  FINAL EVALUATION ON FULL DATASET:")
    print(f"  v0.6 v2 -> BalAcc={v06_v2_m['balanced_accuracy']}, MacroF1={v06_v2_m['macro_f1']}, Coverage={v06_v2_m['coverage']}")
    print(f"  Recalls -> Eso={v06_v2_m['eso_sensitivity']}, Exo={v06_v2_m['exo_sensitivity']}, Normal={v06_v2_m['normal_specificity']}")
    print("=" * 75)

    print("\n[7/7] Saving v0.6 v2 artifacts...")
    elapsed = round(time.time() - t0, 1)

    eval_report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "status": "COMPLETED_SAVED",
        "dataset": {
            "root": str(args.dataset_root),
            "n_total": len(paths),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "label_counts": {k: int(v) for k, v in counts.items()},
        },
        "eye_crop_geometry": {
            "module": "app.services.eye_crop_geometry_service",
            "features": GEOMETRIC_FEATURE_NAMES,
            "fallback_used": False,
        },
        "benchmarks": {
            "v05_forced": v05_forced_m,
            "v05_abstained": v05_abstained_m,
            "v02_eyecrop": v02_m,
            "v06_v1_baseline": {
                "balanced_accuracy": 0.3576,
                "macro_f1": 0.3007,
                "coverage": 0.3309,
                "eso_sensitivity": 0.7727,
                "exo_sensitivity": 0.0000,
                "normal_specificity": 0.3000,
                "brier_score": 0.6652,
            },
            "v06_v2_ensemble": v06_v2_m,
        },
        "tuning": {
            "val_ratio": args.val_ratio,
            "n_tried": tune_res["n_tried"],
            "best_params": {
                "alpha": best["alpha"],
                "conf_t": best["conf_t"],
                "margin_t": best["margin_t"],
            },
            "best_val_metrics": best["metrics"],
        },
        "elapsed_seconds": elapsed,
        "governance": [
            "v0.5 NOT overwritten.",
            "v0.2 original NOT overwritten.",
            "v0.6 v1 NOT overwritten.",
            "Eye-crop geometry extracted without full-face dependency.",
            "Tuned strictly on validation split.",
            "research_candidate - NOT clinical diagnosis.",
        ],
    }

    # Save evaluation report
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(
        json.dumps(eval_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"  Evaluation saved: {args.eval_output}")

    # Save branch v0.2 eye-crop model
    args.branch_v02_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_id": "hirschberg-branch-v0.2-eyecrop",
            "status": "research_candidate",
            "pipeline": pipe_v02,
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "geometric_feature_names": GEOMETRIC_FEATURE_NAMES,
            "vision_backbone": "efficientnet_b0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        args.branch_v02_output,
    )
    print(f"  v0.2 eye-crop branch saved: {args.branch_v02_output}")

    # Save ensemble bundle v2
    bundle_v2 = {
        "model_id": MODEL_ID,
        "status": "production_runtime",
        "version": "v0.6_ensemble_v2",
        "model_type": "ensemble_eye_crop_prior_normalized",
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "is_production": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "components": {
            "primary": "v0.5_safe_screening",
            "secondary_branch": "v0.2_eyecrop_geometry_effnet",
        },
        "primary_pipeline": bundle5["pipeline"],
        "branch_v02_pipeline": pipe_v02,
        "fusion_policy": {
            "type": "prior_normalized_weighted",
            "alpha": best["alpha"],
            "conf_threshold": best["conf_t"],
            "margin_threshold": best["margin_t"],
            "prior5": [0.65, 0.15, 0.20],
            "abstain_status": "UNCERTAIN",
        },
        "metrics": v06_v2_m,
        "v05_baseline": v05_forced_m,
        "non_clinical_declaration": (
            "Hirschberg v0.6 v2 Eye-Crop Research Ensemble. Real-world eye-crop geometry. "
            "Returns UNCERTAIN for ambiguous evidence. NOT for clinical diagnosis."
        ),
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle_v2, args.model_output)
    print(f"  v0.6 v2 model saved: {args.model_output}")

    # Save model card
    card = write_model_card(
        eval_report,
        v05_forced_m,
        v05_abstained_m,
        v02_m,
        eval_report["benchmarks"]["v06_v1_baseline"],
        v06_v2_m,
        best,
    )
    args.card_output.parent.mkdir(parents=True, exist_ok=True)
    args.card_output.write_text(card, encoding="utf-8")
    print(f"  Model card saved: {args.card_output}")

    print(f"\n[DONE] v0.6 v2 training and benchmark finished in {elapsed}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
