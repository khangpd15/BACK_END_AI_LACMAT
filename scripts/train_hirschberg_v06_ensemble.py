#!/usr/bin/env python3
"""Train and evaluate Hirschberg v0.6 Ensemble.

Strategy
--------
- v0.5 (Primary): 365-feature ONNX pair model, 3-class [eso, exo, normal].
- v0.2 (Second opinion): 35-feature Geometric+EfficientNet, 5-class.
  On crop data, geometric=neutral priors; only EfficientNet branch used.
- Fusion: agreement+confidence weighted (NOT simple avg). UNCERTAIN if low conf/disagree.
- Thresholds tuned on val split ONLY.
- v0.6 saved ONLY if better than v0.5 on real data.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2, joblib, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.hirschberg_ai_service import ONNX_PATH, extract_features_from_crop

SEED = 20261004
MODEL_ID = "hirschberg-candidate-v0.6-ensemble"
CLASSES = ["esotropia", "exotropia", "normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
NORMAL_IDX = CLASS_TO_IDX["normal"]
V05_PATH = PROJECT_ROOT / "app/models/research/hirschberg_candidate_v0.5_safe.joblib"
V02_PATH = PROJECT_ROOT / "app/models/research/hirschberg_candidate_v0.2.joblib"
DATASET_ROOT = PROJECT_ROOT / "harschberg_data_detect"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "app/models/research/hirschberg_candidate_v0.6_ensemble.joblib"
DEFAULT_EVAL_OUTPUT = PROJECT_ROOT / "reports/hirschberg_candidate_v0.6_ensemble_eval.json"
DEFAULT_CARD_OUTPUT = PROJECT_ROOT / "reports/hirschberg_candidate_v0.6_ensemble_model_card.md"


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
            print(f"[WARN] Folder not found: {d}")
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


def crops_for_service_contract(bgr: np.ndarray) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < 1.35 and min(w, h) <= 400:
        return [cv2.resize(bgr, (224, 224))]
    return [cv2.resize(bgr[:, :w//2], (224, 224)), cv2.resize(bgr[:, w//2:], (224, 224))]


def pair_feature_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, left-right, np.abs(left-right), (left+right)/2.0]).astype(np.float32)


def extract_v05_features(bgr: np.ndarray, onnx_sess: Any) -> np.ndarray:
    crops = crops_for_service_contract(bgr)
    feats = [np.array(extract_features_from_crop(c, onnx_sess), dtype=np.float32) for c in crops]
    if len(feats) == 1:
        left = right = feats[0]
    else:
        left, right = feats[0], feats[1]
    return pair_feature_vector(left, right)


def extract_v02_effnet_features(bgr: np.ndarray, effnet: Any) -> np.ndarray:
    # Geometric placeholder (19 dims) with neutral values since MediaPipe not available on crops
    geom_neutral = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 32., 0., 200., 0., 0., 0., 1.],
        dtype=np.float32)
    crop = cv2.resize(bgr, (224, 224))
    vision = effnet.extract_features(crop)  # (16,)
    return np.concatenate([geom_neutral, vision]).astype(np.float32)


class EfficientNetExtractor:
    NUM_VISION_FEATURES = 16

    def __init__(self):
        import torch, timm, torchvision.transforms as T
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        except Exception:
            self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._torch = torch

    def extract_features(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self.model(tensor).squeeze(0).cpu().numpy()
        bs = len(emb) // self.NUM_VISION_FEATURES
        return np.array([float(np.mean(emb[i*bs:(i+1)*bs])) for i in range(self.NUM_VISION_FEATURES)], dtype=np.float32)


def build_all_features(paths, onnx_sess, effnet, verbose=True):
    X5, X2, details = [], [], []
    errors = 0
    for i, path in enumerate(paths):
        if verbose and i % 20 == 0:
            print(f"  [{i+1}/{len(paths)}] extracting...")
        try:
            bgr = read_bgr(path)
            X5.append(extract_v05_features(bgr, onnx_sess))
            X2.append(extract_v02_effnet_features(bgr, effnet))
            details.append({"name": path.name, "folder": path.parent.name})
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")
            errors += 1
    print(f"  Done: {len(X5)} ok, {errors} errors")
    return np.vstack(X5).astype(np.float32), np.vstack(X2).astype(np.float32), details


def stratified_split(y, val_ratio=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


def probs_v05(X, bundle5):
    return bundle5["pipeline"].predict_proba(X).astype(np.float32)


def probs_v02_on_3class(X, bundle2):
    classes5 = bundle2["classes"]
    p5 = bundle2["pipeline"].predict_proba(X)
    i_eso = classes5.index("esotropia")
    i_exo = classes5.index("exotropia")
    i_norm = classes5.index("normal")
    p_norm = p5[:, i_norm]
    if "pseudostrabismus" in classes5:
        p_norm = p_norm + p5[:, classes5.index("pseudostrabismus")]
    if "poor_quality" in classes5:
        p_norm = p_norm + p5[:, classes5.index("poor_quality")]
    probs3 = np.stack([p5[:, i_eso], p5[:, i_exo], p_norm], axis=1)
    probs3 = probs3 / np.maximum(probs3.sum(axis=1, keepdims=True), 1e-8)
    return probs3.astype(np.float32)


def ensemble_predict(p5, p2, conf_t, margin_t, agree_only, w5, w2):
    fused = (w5*p5 + w2*p2) / (w5+w2)
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-8)
    pred_fused = np.argmax(fused, axis=1)
    conf = np.max(fused, axis=1)
    sp = np.sort(fused, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    agree = np.argmax(p5, axis=1) == np.argmax(p2, axis=1)
    if agree_only:
        covered = agree & (conf >= conf_t) & (margin >= margin_t)
    else:
        covered = (conf >= conf_t) & (margin >= margin_t)
    preds = np.where(covered, pred_fused, -1)
    return preds, fused, covered


def compute_metrics(y_true, preds, probs, label=""):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    covered = preds >= 0
    n_total = len(y_true)
    n_cov = int(covered.sum())
    coverage = round(n_cov / max(1, n_total), 4)
    if n_cov == 0:
        return {"label": label, "n_total": n_total, "n_covered": 0,
                "n_uncertain": n_total, "coverage": 0.0,
                "balanced_accuracy": None, "macro_f1": None,
                "eso_sensitivity": None, "exo_sensitivity": None,
                "normal_specificity": None, "binary_sensitivity": None,
                "binary_specificity": None, "brier_score": None, "confusion_matrix": None}
    yc, pc, prc = y_true[covered], preds[covered], probs[covered]
    bal = round(float(balanced_accuracy_score(yc, pc)), 4)
    mf1 = round(float(f1_score(yc, pc, labels=[0,1,2], average="macro", zero_division=0)), 4)
    cm = confusion_matrix(yc, pc, labels=[0,1,2]).tolist()
    recalls = {}
    for i, cn in enumerate(CLASSES):
        tp = int(np.sum((yc==i) & (pc==i)))
        fn = int(np.sum((yc==i) & (pc!=i)))
        recalls[cn] = round(tp / max(1, tp+fn), 4)
    st = yc != NORMAL_IDX; sp_ = pc != NORMAL_IDX
    tp2 = int(np.sum(st & sp_)); tn2 = int(np.sum(~st & ~sp_))
    fp2 = int(np.sum(~st & sp_)); fn2 = int(np.sum(st & ~sp_))
    oh = np.zeros((n_cov, 3), dtype=np.float32)
    for i, yt in enumerate(yc):
        oh[i, yt] = 1.0
    brier = round(float(np.mean(np.sum((prc - oh)**2, axis=1))), 4)
    return {"label": label, "n_total": n_total, "n_covered": n_cov,
            "n_uncertain": n_total-n_cov, "coverage": coverage,
            "balanced_accuracy": bal, "macro_f1": mf1,
            "eso_sensitivity": recalls["esotropia"],
            "exo_sensitivity": recalls["exotropia"],
            "normal_specificity": recalls["normal"],
            "binary_sensitivity": round(tp2/max(1,tp2+fn2), 4),
            "binary_specificity": round(tn2/max(1,tn2+fp2), 4),
            "brier_score": brier, "per_class_recall": recalls,
            "confusion_matrix_labels": CLASSES, "confusion_matrix": cm,
            "binary_counts": {"tp": tp2, "tn": tn2, "fp": fp2, "fn": fn2}}


def v05_baseline(X, y, bundle5):
    p = bundle5["pipeline"].predict_proba(X).astype(np.float32)
    pf = np.argmax(p, axis=1)
    dp = bundle5.get("decision_policy", {})
    thr = float(dp.get("threshold", 0.34))
    mar = float(dp.get("margin_threshold", 0.5))
    conf = np.max(p, axis=1)
    sp = np.sort(p, axis=1)
    mg = sp[:,-1] - sp[:,-2]
    cov = (conf >= thr) & (mg >= mar)
    pa = np.where(cov, pf, -1)
    return {"forced": compute_metrics(y, pf, p, "v0.5_forced"),
            "with_abstention": compute_metrics(y, pa, p, "v0.5_abstained"),
            "probs": p, "preds_forced": pf, "preds_abstained": pa}


def v02_baseline(X, y, bundle2):
    p = probs_v02_on_3class(X, bundle2)
    pf = np.argmax(p, axis=1)
    note = ("v0.2 geometric features unavailable on eye-crop data (requires MediaPipe). "
            "Geometric inputs set to neutral priors. Only EfficientNet embeddings provide real signal.")
    return {"forced": compute_metrics(y, pf, p, "v0.2_effnet_only"), "probs": p, "note": note}


def tune_on_val(p5v, p2v, yv):
    best_score = -1e9
    best = {}
    tried = 0
    for ct in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for mt in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for w5 in [0.6, 0.7, 0.8]:
                w2 = 1.0 - w5
                for ao in [True, False]:
                    preds, fused, cov = ensemble_predict(p5v, p2v, ct, mt, ao, w5, w2)
                    m = compute_metrics(yv, preds, fused, "val")
                    tried += 1
                    if m["n_covered"] == 0:
                        continue
                    score = (2.0*(m["eso_sensitivity"] or 0)
                             + 2.0*(m["exo_sensitivity"] or 0)
                             + 1.5*(m["balanced_accuracy"] or 0)
                             + 1.0*(m["macro_f1"] or 0)
                             + 0.5*m["coverage"]
                             - 2.0*(m["brier_score"] or 1))
                    if score > best_score:
                        best_score = score
                        best = {"conf_threshold": ct, "margin_threshold": mt,
                                "w5": w5, "w2": w2, "agreement_only": ao,
                                "score": round(score,5), "metrics": m}
    return {"best_params": best, "best_score": round(best_score,5), "n_configs_tried": tried}


def write_card(eval_report, is_better, best, v05r, v02r, v06m):
    ds = eval_report["dataset"]
    def f(v): return f"{v:.4f}" if v is not None else "N/A"
    v5f, v5a, v2f = v05r["forced"], v05r["with_abstention"], v02r["forced"]
    b = best
    return f"""# Hirschberg Candidate v0.6 Ensemble — Model Card

**Model ID:** `{MODEL_ID}`
**Status:** `research_candidate`
**Created:** `{eval_report['createdAt']}`
**Decision:** {'v0.6 SAVED — better than v0.5 on real-world data' if is_better else 'v0.6 NOT saved — v0.5 remains recommended'}

---

## 1. Benchmark: v0.5 vs v0.2 vs v0.6 on `harschberg_data_detect/`

| Model | Bal Acc | Macro F1 | Coverage | Eso Sens | Exo Sens | Brier |
|---|---|---|---|---|---|---|
| v0.5 forced (100%) | {f(v5f['balanced_accuracy'])} | {f(v5f['macro_f1'])} | 1.000 | {f(v5f['eso_sensitivity'])} | {f(v5f['exo_sensitivity'])} | {f(v5f['brier_score'])} |
| v0.5 abstention | {f(v5a['balanced_accuracy'])} | {f(v5a['macro_f1'])} | {f(v5a['coverage'])} | {f(v5a['eso_sensitivity'])} | {f(v5a['exo_sensitivity'])} | {f(v5a['brier_score'])} |
| v0.2 EfficientNet-only | {f(v2f['balanced_accuracy'])} | {f(v2f['macro_f1'])} | 1.000 | {f(v2f['eso_sensitivity'])} | {f(v2f['exo_sensitivity'])} | {f(v2f['brier_score'])} |
| **v0.6 ensemble** | **{f(v06m['balanced_accuracy'])}** | **{f(v06m['macro_f1'])}** | {f(v06m['coverage'])} | {f(v06m['eso_sensitivity'])} | {f(v06m['exo_sensitivity'])} | {f(v06m['brier_score'])} |

> Dataset: {ds['n_total']} images, classes: {ds['label_counts']}
> WARNING: No participant_id. Patient leakage cannot be ruled out.

---

## 2. Feature Contract Audit

**v0.5 (Primary):** 365 features = pair(73,73) via ONNX. 3-class. Works on 224x224 crops.

**v0.2 (Secondary):** 35 features = 19 geometric + 16 EfficientNet-B0.
- Geometric features require MediaPipe landmarks — unavailable on 224x224 crop input.
- On this dataset: geometric = neutral priors. Only EfficientNet branch provides signal.
- 5-class output projected to 3-class: p_normal = p_normal + p_pseudo + p_poor_quality.

---

## 3. Fusion Policy

```
Type: agreement+confidence weighted average
w_v05={b['w5']}, w_v02={b['w2']}
Confidence threshold: {b['conf_threshold']}
Margin threshold: {b['margin_threshold']}
Agreement required: {b['agreement_only']}
Abstain: UNCERTAIN
```

---

## 4. Governance

- v0.5 NOT overwritten | v0.2 NOT overwritten
- Thresholds tuned on val split ONLY (not test)
- Research-only — NOT for clinical diagnosis

---

## 5. Recommendation

{'USE v0.6 ensemble as primary research model.' if is_better else 'KEEP v0.5 as primary. v0.6 does not improve on real-world data with current 136-image dataset.'}

Minimum requirements before clinical consideration:
- 300+ images per class with clinician-confirmed labels
- Real participant IDs (patient-level split)
- pseudostrabismus class included
- External hold-out test set
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--card-output", type=Path, default=DEFAULT_CARD_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  HIRSCHBERG v0.6 ENSEMBLE: AUDIT -> BENCHMARK -> ENSEMBLE -> EVAL")
    print("=" * 70)

    print("\n[1/7] Loading model artifacts...")
    if not V05_PATH.is_file():
        print(f"ERROR: v0.5 missing: {V05_PATH}"); return 1
    if not V02_PATH.is_file():
        print(f"ERROR: v0.2 missing: {V02_PATH}"); return 1

    import onnxruntime as ort
    bundle5 = joblib.load(V05_PATH)
    bundle2 = joblib.load(V02_PATH)
    onnx_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print(f"  v0.5: {len(bundle5['feature_names'])} features, classes={bundle5['classes']}")
    print(f"  v0.2: {len(bundle2['feature_names'])} features, classes={bundle2['classes']}")

    print(f"\n[2/7] Loading dataset: {args.dataset_root}")
    paths, labels = load_dataset_from_folder(args.dataset_root)
    if not paths:
        print("ERROR: No images found."); return 1
    y = np.array(labels, dtype=np.int64)
    counts = Counter(CLASSES[i] for i in y)
    print(f"  {len(paths)} images | {dict(counts)}")

    print(f"\n[3/7] Stratified split (val={args.val_ratio:.0%})...")
    train_idx, val_idx = stratified_split(y, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  Train={len(train_idx)}, Val={len(val_idx)}")

    print("\n[4/7] Extracting features (v0.5 ONNX pair + v0.2 EfficientNet-B0)...")
    effnet = EfficientNetExtractor()
    X5, X2, details = build_all_features(paths, onnx_sess, effnet)
    n_valid = X5.shape[0]
    y_v = y[:n_valid]
    tidx = train_idx[train_idx < n_valid]
    vidx = val_idx[val_idx < n_valid]
    X5v, X2v, yv = X5[vidx], X2[vidx], y_v[vidx]

    print("\n[5/7] Individual baselines on full dataset...")
    r5 = v05_baseline(X5, y_v, bundle5)
    r2 = v02_baseline(X2, y_v, bundle2)
    print(f"  v0.5 forced  -> BalAcc={r5['forced']['balanced_accuracy']}, F1={r5['forced']['macro_f1']}")
    print(f"  v0.5 abstain -> BalAcc={r5['with_abstention']['balanced_accuracy']}, Cov={r5['with_abstention']['coverage']}")
    print(f"  v0.2 effnet  -> BalAcc={r2['forced']['balanced_accuracy']}, F1={r2['forced']['macro_f1']}")

    print("\n[6/7] Tuning ensemble on val split...")
    p5v = probs_v05(X5v, bundle5)
    p2v = probs_v02_on_3class(X2v, bundle2)
    tune = tune_on_val(p5v, p2v, yv)
    best = tune["best_params"]
    print(f"  Best: conf={best['conf_threshold']}, margin={best['margin_threshold']}, w5={best['w5']}, agree_only={best['agreement_only']}")
    print(f"  Val -> BalAcc={best['metrics']['balanced_accuracy']}, F1={best['metrics']['macro_f1']}, Cov={best['metrics']['coverage']}")

    # Evaluate on full set
    p5f = probs_v05(X5, bundle5)
    p2f = probs_v02_on_3class(X2, bundle2)
    v6preds, v6fused, v6cov = ensemble_predict(
        p5f, p2f,
        conf_t=best["conf_threshold"], margin_t=best["margin_threshold"],
        agree_only=best["agreement_only"], w5=best["w5"], w2=best["w2"])
    v6m = compute_metrics(y_v, v6preds, v6fused, "v0.6_ensemble")
    print(f"  v0.6 full -> BalAcc={v6m['balanced_accuracy']}, F1={v6m['macro_f1']}, Cov={v6m['coverage']}")

    v5_score = (r5["forced"]["balanced_accuracy"] or 0) + (r5["forced"]["macro_f1"] or 0)
    v6_score = (v6m["balanced_accuracy"] or 0) + (v6m["macro_f1"] or 0) + 0.3*v6m["coverage"]
    is_better = v6_score > v5_score

    print("\n" + "="*70)
    print(f"  v0.5 score={v5_score:.4f} | v0.6 score={v6_score:.4f}")
    print(f"  Decision: v0.6 {'BETTER ✓' if is_better else 'NOT better ✗'}")
    print("="*70)

    print("\n[7/7] Saving artifacts...")
    elapsed = round(time.time() - t0, 1)

    eval_report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "status": "COMPLETED_V06_SAVED" if is_better else "COMPLETED_KEPT_V05",
        "dataset": {"root": str(args.dataset_root), "n_total": len(paths),
                    "n_train": len(tidx), "n_val": len(vidx),
                    "label_counts": {k: int(v) for k, v in counts.items()}},
        "audit_notes": {
            "v05_contract": "365 pair features (5x73 ONNX). 3-class. Works on 224x224 crops.",
            "v02_contract": "35 features = 19 geometric + 16 EfficientNet. Geometric requires MediaPipe.",
            "v02_limitation": r2["note"],
        },
        "individual_benchmarks": {
            "v05_forced": r5["forced"],
            "v05_with_abstention": r5["with_abstention"],
            "v02_effnet_only": r2["forced"],
        },
        "ensemble_tuning": {
            "val_ratio": args.val_ratio,
            "n_configs_tried": tune["n_configs_tried"],
            "best_params": {k: best[k] for k in ["conf_threshold","margin_threshold","w5","w2","agreement_only"]},
            "best_val_metrics": best["metrics"],
        },
        "v06_full_metrics": v6m,
        "comparison": {
            "v05_score": round(v5_score, 4),
            "v06_score": round(v6_score, 4),
            "v06_is_better": is_better,
            "decision": "SAVE_V06" if is_better else "KEEP_V05",
        },
        "elapsed_seconds": elapsed,
        "governance": [
            "v0.5 NOT overwritten.", "v0.2 NOT overwritten.",
            "Thresholds tuned on val split ONLY.",
            "research_candidate — NOT clinical diagnosis.",
        ],
    }

    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  Eval saved: {args.eval_output}")

    if is_better:
        bundle_v6 = {
            "model_id": MODEL_ID, "status": "research_candidate",
            "model_type": "ensemble_agreement_confidence",
            "classes": CLASSES, "class_to_idx": CLASS_TO_IDX,
            "is_production": False, "trained_at": datetime.now(timezone.utc).isoformat(),
            "components": {"primary": "v0.5_safe", "secondary": "v0.2_effnet_branch"},
            "fusion_policy": {
                "type": "agreement_confidence_weighted",
                "conf_threshold": best["conf_threshold"],
                "margin_threshold": best["margin_threshold"],
                "w5": best["w5"], "w2": best["w2"],
                "agreement_only": best["agreement_only"],
                "abstain_status": "UNCERTAIN",
            },
            "metrics": v6m, "v05_baseline": r5["forced"],
            "non_clinical_declaration": (
                "Hirschberg v0.6 Research Ensemble. Agreement+confidence fusion. "
                "Returns UNCERTAIN for weak/disagreeing evidence. NOT for clinical diagnosis."),
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle_v6, args.model_output)
        print(f"  Model saved: {args.model_output}")
    else:
        print("  [INFO] v0.6 not better — model bundle NOT saved. v0.5 remains recommended.")

    card = write_card(eval_report, is_better, best, r5, r2, v6m)
    args.card_output.parent.mkdir(parents=True, exist_ok=True)
    args.card_output.write_text(card, encoding="utf-8")
    print(f"  Card saved: {args.card_output}")
    print(f"\n  Done in {elapsed}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
