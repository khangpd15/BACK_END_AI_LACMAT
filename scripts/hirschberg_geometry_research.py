"""Research-only Hirschberg geometry evaluation.

This script does not train a CNN and does not touch production APIs. It uses
simple detector-derived geometry features to test whether left/right
asymmetry separates strabismus from normal on the accepted 136-image crop set.

Outputs:
- processed/hirschberg_geometry_features.csv
- reports/hirschberg_geometry_research_report.md
- reports/hirschberg_geometry_research_results.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "harschberg_data_detect"
PROCESSED_DIR = PROJECT_ROOT / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_CSV = PROCESSED_DIR / "hirschberg_geometry_features.csv"
RESULT_JSON = REPORTS_DIR / "hirschberg_geometry_research_results.json"
REPORT_MD = REPORTS_DIR / "hirschberg_geometry_research_report.md"
DEFAULT_MANUAL_CSV = PROCESSED_DIR / "hirschberg_manual_annotations.csv"
SEED = 20261003
CLASS_DIRS = {
    "esotropia": "esotropia_harschberg",
    "exotropia": "exotropia_harschberg",
    "normal": "normal_harschberg",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def iter_images(dataset_root: Path) -> Iterable[ImageRecord]:
    for label, folder_name in CLASS_DIRS.items():
        folder = dataset_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                yield ImageRecord(path=path, label=label)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (small > float(np.mean(small))).astype(np.uint8).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


class UnionFind:
    def __init__(self, keys: Sequence[str]):
        self.parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_near_duplicate_groups(records: Sequence[ImageRecord], threshold: int = 8) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
    rels: List[str] = []
    hashes: Dict[str, str] = {}
    ahashes: Dict[str, np.ndarray] = {}
    for rec in records:
        rel = str(rec.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        image = cv2.imread(str(rec.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        rels.append(rel)
        hashes[rel] = file_sha256(rec.path)
        ahashes[rel] = average_hash(image)

    uf = UnionFind(rels)
    near_pairs: List[Dict[str, object]] = []
    by_sha: Dict[str, List[str]] = defaultdict(list)
    for rel, digest in hashes.items():
        by_sha[digest].append(rel)
    for paths in by_sha.values():
        for path in paths[1:]:
            uf.union(paths[0], path)

    for i, rel_a in enumerate(rels):
        for rel_b in rels[i + 1:]:
            dist = hamming(ahashes[rel_a], ahashes[rel_b])
            if dist <= threshold:
                uf.union(rel_a, rel_b)
                near_pairs.append({"path1": rel_a, "path2": rel_b, "ahash_hamming": dist})

    root_to_gid: Dict[str, str] = {}
    groups: Dict[str, str] = {}
    for rel in rels:
        root = uf.find(rel)
        if root not in root_to_gid:
            root_to_gid[root] = f"group_{len(root_to_gid):04d}"
        groups[rel] = root_to_gid[root]
    return groups, near_pairs


def estimate_iris_center(eye_bgr: np.ndarray) -> Tuple[float, float, float, str]:
    h, w = eye_bgr.shape[:2]
    default = (w / 2.0, h / 2.0, min(w, h) * 0.32)
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)

    # The pupil/iris is the dominant dark compact component in these accepted
    # crops. Prefer that over Hough circles because eyelid edges often pull the
    # circle fit away from the manual pupil center.
    blurred_dark = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_threshold = min(95.0, float(np.percentile(blurred_dark, 12)))
    dark_mask = (blurred_dark <= dark_threshold).astype(np.uint8) * 255
    central_mask = np.zeros_like(dark_mask)
    cv2.rectangle(
        central_mask,
        (int(w * 0.03), int(h * 0.12)),
        (int(w * 0.97), int(h * 0.88)),
        255,
        -1,
    )
    dark_mask = cv2.bitwise_and(dark_mask, central_mask)
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)
    dark_candidates = []
    crop_center = np.array([w / 2.0, h / 2.0])
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        box_w = int(stats[i, cv2.CC_STAT_WIDTH])
        box_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 150 or area > 7000 or box_w < 10 or box_h < 10:
            continue
        aspect = max(box_w, box_h) / max(1, min(box_w, box_h))
        if aspect > 3.5:
            continue
        x, y = float(centroids[i][0]), float(centroids[i][1])
        dist = float(np.linalg.norm(np.array([x, y]) - crop_center))
        score = dist - min(area, 3000) * 0.015
        radius = max(8.0, min(max(box_w, box_h) * 0.5, math.sqrt(area / math.pi) * 1.25))
        dark_candidates.append((score, x, y, radius))
    if dark_candidates:
        _, x, y, radius = min(dark_candidates, key=lambda item: item[0])
        return x, y, radius, "dark_component"

    eq = cv2.equalizeHist(gray)
    blurred = cv2.medianBlur(eq, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, w // 3),
        param1=45,
        param2=18,
        minRadius=max(8, min(w, h) // 10),
        maxRadius=max(12, min(w, h) // 2),
    )
    if circles is not None and len(circles[0]) > 0:
        center = np.array([w / 2.0, h / 2.0])
        best = min(circles[0], key=lambda c: float(np.linalg.norm(np.array([c[0], c[1]]) - center)))
        if np.linalg.norm(np.array([best[0], best[1]]) - center) < max(w, h) * 0.35:
            return float(best[0]), float(best[1]), float(best[2]), "hough"

    # Fallback: dark centroid in the central eye crop.
    inv = 255.0 - gray.astype(np.float32)
    yy, xx = np.indices(gray.shape)
    total = float(np.sum(inv))
    if total > 1e-6:
        cx = float(np.sum(xx * inv) / total)
        cy = float(np.sum(yy * inv) / total)
        return cx, cy, default[2], "dark_centroid"
    return (*default, "default")


def detect_reflex(eye_bgr: np.ndarray, iris: Tuple[float, float, float]) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    cx, cy, radius = iris
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    broad_search_radius = max(16.0, radius * 1.60)
    strict_search_radius = max(12.0, radius * 1.30)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(broad_search_radius)), 255, -1)
    masked_values = gray[mask > 0]
    max_val = float(np.max(masked_values)) if masked_values.size else 0.0
    if max_val < 180.0:
        return None, None, None, "low_peak_brightness"
    thresh = max(180, int(max_val - 32))
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask)
    num_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = []
    rejected_large_or_long = 0
    rejected_too_far = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        x, y = float(centroids[i][0]), float(centroids[i][1])
        dist = float(math.hypot(x - cx, y - cy))
        if area > 60 or (aspect > 2.8 and area >= 4):
            rejected_large_or_long += 1
            continue
        if area < 1:
            continue
        if dist > strict_search_radius:
            rejected_too_far += 1
            continue
        component_mask = component_labels == i
        peak = float(np.max(gray[component_mask])) if np.any(component_mask) else max_val
        score = dist + area * 0.12 + max(0.0, aspect - 1.0) * 1.5 - (peak - thresh) * 0.03
        candidates.append((score, dist, x, y, float(area), aspect))
    if not candidates:
        if rejected_large_or_long:
            return None, None, None, "large_or_elongated_glare"
        if rejected_too_far:
            return None, None, None, "reflex_candidate_too_far"
        return None, None, None, "not_found"
    candidates = sorted(candidates, key=lambda item: item[0])
    if len(candidates) > 1:
        best = candidates[0]
        clustered = [
            cand for cand in candidates[1:]
            if math.hypot(cand[2] - best[2], cand[3] - best[3]) <= 8.0
            and cand[0] - best[0] <= 3.0
        ]
        if clustered:
            return None, None, None, "clustered_reflex_candidates"
        status = "detected_with_secondary_candidates"
    else:
        status = "detected"
    _, _, x, y, area, _ = candidates[0]
    return x, y, area, status


def extract_detector_features(rec: ImageRecord, group_id: str) -> Dict[str, object]:
    image = cv2.imread(str(rec.path))
    rel = str(rec.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    base: Dict[str, object] = {
        "relative_path": rel,
        "class_label": rec.label,
        "binary_label": "strabismus" if rec.label in {"esotropia", "exotropia"} else "normal",
        "group_id": group_id,
    }
    if image is None:
        base["quality_ok"] = False
        base["quality_reason"] = "cannot_read"
        return base

    h, w = image.shape[:2]
    od_crop = image[:, : w // 2]
    os_crop = image[:, w // 2:]
    od_iris = estimate_iris_center(od_crop)
    os_iris = estimate_iris_center(os_crop)
    od_reflex = detect_reflex(od_crop, od_iris[:3])
    os_reflex = detect_reflex(os_crop, os_iris[:3])

    def offset(reflex, iris) -> Optional[float]:
        if reflex[0] is None:
            return None
        return float((reflex[0] - iris[0]) / max(1e-6, iris[2]))

    od_offset = offset(od_reflex, od_iris)
    os_offset = offset(os_reflex, os_iris)
    both_reflex = od_offset is not None and os_offset is not None
    if both_reflex:
        signed_asymmetry = float(os_offset - od_offset)
        abs_asymmetry = abs(signed_asymmetry)
        mean_abs_offset = float((abs(od_offset) + abs(os_offset)) / 2.0)
        max_abs_offset = float(max(abs(od_offset), abs(os_offset)))
    else:
        signed_asymmetry = 0.0
        abs_asymmetry = 0.0
        mean_abs_offset = 0.0
        max_abs_offset = 0.0

    if both_reflex:
        quality_reason = "ok"
    else:
        quality_reason = f"missing_reflex_od={od_reflex[3]}_os={os_reflex[3]}"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base.update({
        "width": w,
        "height": h,
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "od_iris_x": od_iris[0],
        "od_iris_y": od_iris[1],
        "od_iris_radius": od_iris[2],
        "od_iris_method": od_iris[3],
        "os_iris_x": os_iris[0] + (w // 2),
        "os_iris_y": os_iris[1],
        "os_iris_radius": os_iris[2],
        "os_iris_method": os_iris[3],
        "od_reflex_x": od_reflex[0],
        "od_reflex_y": od_reflex[1],
        "od_reflex_area": od_reflex[2],
        "od_reflex_status": od_reflex[3],
        "os_reflex_x": None if os_reflex[0] is None else os_reflex[0] + (w // 2),
        "os_reflex_y": os_reflex[1],
        "os_reflex_area": os_reflex[2],
        "os_reflex_status": os_reflex[3],
        "od_offset_norm": od_offset,
        "os_offset_norm": os_offset,
        "signed_asymmetry_lr": signed_asymmetry,
        "abs_asymmetry_lr": abs_asymmetry,
        "mean_abs_offset": mean_abs_offset,
        "max_abs_offset": max_abs_offset,
        "both_reflex_detected": int(both_reflex),
        "quality_ok": bool(both_reflex),
        "quality_reason": quality_reason,
    })
    return base


def load_manual_annotations(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = row.get("relative_path", "")
            if not rel:
                continue
            try:
                rows[rel] = {key: float(row[key]) for key in [
                    "od_pupil_x", "od_pupil_y", "od_reflex_x", "od_reflex_y",
                    "os_pupil_x", "os_pupil_y", "os_reflex_x", "os_reflex_y",
                ]}
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def attach_manual_error(rows: List[Dict[str, object]], manual: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    errors = []
    for row in rows:
        rel = str(row["relative_path"])
        ann = manual.get(rel)
        if not ann:
            continue
        for eye in ("od", "os"):
            det_x = row.get(f"{eye}_reflex_x")
            det_y = row.get(f"{eye}_reflex_y")
            if det_x is None or det_y is None:
                continue
            err = math.hypot(float(det_x) - ann[f"{eye}_reflex_x"], float(det_y) - ann[f"{eye}_reflex_y"])
            row[f"{eye}_reflex_manual_error_px"] = err
            errors.append(err)
    if not errors:
        return {"manual_count": len(manual), "detector_reflex_error_px": None}
    arr = np.array(errors, dtype=float)
    return {
        "manual_count": len(manual),
        "detector_reflex_error_px": {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        },
    }


def as_matrix(rows: Sequence[Dict[str, object]], labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_names = [
        "signed_asymmetry_lr",
        "abs_asymmetry_lr",
        "mean_abs_offset",
        "max_abs_offset",
        "od_offset_norm",
        "os_offset_norm",
        "brightness_mean",
        "brightness_std",
    ]
    usable = [row for row in rows if row.get("quality_ok") is True and row.get("class_label") in labels]
    X = np.array([[float(row.get(name) or 0.0) for name in feature_names] for row in usable], dtype=float)
    y = np.array([labels.index(str(row["class_label"])) for row in usable], dtype=int)
    groups = np.array([str(row["group_id"]) for row in usable])
    return X, y, groups, feature_names


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, object]:
    out: Dict[str, object] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, metric: str, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values = []
    for _ in range(n_boot):
        sample_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sample_groups])
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        elif metric == "macro_f1":
            values.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    arr = np.array(values, dtype=float)
    return {"low": float(np.percentile(arr, 2.5)), "high": float(np.percentile(arr, 97.5))}


def repeated_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Pipeline,
    repeats: int,
    splits: int,
    seed: int,
) -> Dict[str, object]:
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    group_all = []
    fold_count = 0
    for repeat in range(repeats):
        cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
        for train_idx, test_idx in cv.split(X, y, groups):
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
            group_all.extend(groups[test_idx].tolist())
            if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
                y_score_all.extend(model.predict_proba(X[test_idx])[:, 1].tolist())
            fold_count += 1
    y_true_arr = np.array(y_true_all, dtype=int)
    y_pred_arr = np.array(y_pred_all, dtype=int)
    y_score_arr = np.array(y_score_all, dtype=float) if y_score_all else None
    group_arr = np.array(group_all)
    metrics = metric_summary(y_true_arr, y_pred_arr, y_score_arr)
    metrics["fold_count"] = fold_count
    metrics["bootstrap_ci"] = {
        "balanced_accuracy": bootstrap_ci(y_true_arr, y_pred_arr, group_arr, "balanced_accuracy", 1000, seed),
        "macro_f1": bootstrap_ci(y_true_arr, y_pred_arr, group_arr, "macro_f1", 1000, seed + 1),
    }
    return {
        "metrics": metrics,
        "y_true": y_true_arr.tolist(),
        "y_pred": y_pred_arr.tolist(),
        "groups": group_arr.tolist(),
    }


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Pipeline,
    observed_bal_acc: float,
    repeats: int,
    splits: int,
    n_perm: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_to_label = {g: int(Counter(y[groups == g]).most_common(1)[0][0]) for g in unique_groups}
    values = []
    for _ in range(n_perm):
        shuffled_labels = rng.permutation([group_to_label[g] for g in unique_groups])
        shuffled_map = dict(zip(unique_groups, shuffled_labels))
        y_perm = np.array([shuffled_map[g] for g in groups], dtype=int)
        try:
            res = repeated_group_cv(X, y_perm, groups, estimator, repeats=repeats, splits=splits, seed=int(rng.integers(1, 1_000_000)))
            values.append(float(res["metrics"]["balanced_accuracy"]))
        except ValueError:
            continue
    if not values:
        return {"n_permutations": 0, "p_value": None, "balanced_accuracy_values": []}
    arr = np.array(values, dtype=float)
    p_value = float((np.sum(arr >= observed_bal_acc) + 1) / (len(arr) + 1))
    return {
        "n_permutations": int(len(arr)),
        "p_value": p_value,
        "balanced_accuracy_mean": float(np.mean(arr)),
        "balanced_accuracy_p95": float(np.percentile(arr, 95)),
        "balanced_accuracy_values": arr.tolist(),
    }


def write_feature_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(results: Dict[str, object], path: Path) -> None:
    binary = results["binary_logistic"]["metrics"]
    perm = results["binary_logistic"]["permutation_test"]
    lines = [
        "# Hirschberg Geometry Research Report",
        "",
        "Status: research candidate analysis only. No CNN training. No production API integration.",
        "",
        "## Environment",
        "",
        f"- Python env: `.venv312`",
        f"- PyTorch added: no",
        f"- MediaPipe: {results['environment']['mediapipe']}",
        f"- OpenCV: {results['environment']['opencv']}",
        f"- scikit-learn: {results['environment']['sklearn']}",
        f"- Repeated CV config: repeats={results['run_config']['repeats']}, splits={results['run_config']['splits']}",
        f"- Permutations: {results['run_config']['permutations']}",
        "",
        "## Dataset",
        "",
        f"- Accepted images: {results['dataset']['total_images']}",
        f"- Quality usable by current detector: {results['dataset']['quality_usable_count']}",
        f"- Class counts: `{results['dataset']['class_counts']}`",
        f"- Quality counts: `{results['dataset']['quality_counts']}`",
        f"- OD reflex status counts: `{results['dataset']['reflex_status_counts']['od']}`",
        f"- OS reflex status counts: `{results['dataset']['reflex_status_counts']['os']}`",
        f"- Near-duplicate group count: {results['dataset']['group_count']}",
        f"- Near-duplicate pairs at aHash <= 8: {len(results['dataset']['near_duplicate_pairs'])}",
        "",
        "## Detector Rules",
        "",
        "- Pupil/iris center: dark connected component in the central eye crop, with Hough fallback.",
        "- Reflex search: broad iris-centered region, then reject candidates outside the stricter pupil/iris radius.",
        "- Hard rejects: large/elongated bright components, clustered competing glints, low peak brightness, and far candidates.",
        "- Candidate choice: compact, bright, iris-proximal component; secondary candidates are flagged in status.",
        "",
        "## Feature Contract",
        "",
        "Primary diagnostic feature is binocular asymmetry:",
        "",
        "`signed_asymmetry_lr = os_offset_norm - od_offset_norm`",
        "",
        "where each offset is `(reflex_x - iris_center_x) / iris_radius`.",
        "",
        "## Binary Strabismus vs Normal",
        "",
        f"- Model: Logistic Regression, repeated StratifiedGroupKFold",
        f"- Balanced accuracy: {binary['balanced_accuracy']:.4f}",
        f"- Macro-F1: {binary['macro_f1']:.4f}",
        f"- ROC-AUC: {binary.get('roc_auc')}",
        f"- Balanced accuracy 95% bootstrap CI: {binary['bootstrap_ci']['balanced_accuracy']}",
        f"- Permutation p-value: {perm['p_value']}",
        f"- Permutation balanced accuracy p95: {perm.get('balanced_accuracy_p95')}",
        f"- Confusion matrix: `{binary['confusion_matrix']}`",
        "",
        "## Eso vs Exo Direction",
        "",
    ]
    if results.get("stopped_near_random"):
        lines.extend([
            "Skipped by stop rule because the binary geometry signal did not exceed the permutation baseline.",
            "",
        ])
    direction = results.get("direction_rule")
    if direction:
        lines.extend([
            "Rule: `signed_asymmetry_lr > 0 => esotropia`, otherwise exotropia.",
            f"- Balanced accuracy: {direction['balanced_accuracy']:.4f}",
            f"- Macro-F1: {direction['macro_f1']:.4f}",
            f"- Confusion matrix: `{direction['confusion_matrix']}`",
            "",
        ])
    manual = results.get("manual_detector_error", {})
    lines.extend([
        "## Manual Annotation / Detector Error",
        "",
        f"- Manual annotation CSV: `{results['manual_csv']}`",
        f"- Manual annotated images loaded: {manual.get('manual_count', 0)}",
        f"- Detector reflex error px: `{manual.get('detector_reflex_error_px')}`",
        "",
        "Run `python scripts/hirschberg_manual_annotator.py` to create annotations, then rerun this script with `--manual-csv processed/hirschberg_manual_annotations.csv`.",
        "",
        "## Stop Rule",
        "",
    ])
    if results.get("stopped_near_random"):
        causes = results.get("near_random_analysis") or {}
        lines.extend([
            "The binary result is not convincingly above the permutation baseline. Stop here and inspect label quality, detector error, and geometry sign conventions before trying any other model.",
            "",
            "Likely causes to inspect:",
            "",
            f"- Detector: {causes.get('possible_detector_issue')}",
            f"- Labels: {causes.get('possible_label_issue')}",
            f"- Geometry rule: {causes.get('possible_geometry_issue')}",
            f"- Domain: {causes.get('possible_domain_issue')}",
            "",
        ])
    else:
        lines.extend([
            "The binary result is above the permutation baseline by this exploratory test, but it remains research-only because labels, participant groups, and real phone-photo domain data are not yet adequate.",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only Hirschberg geometry evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()

    records = list(iter_images(args.dataset_root))
    group_map, near_pairs = build_near_duplicate_groups(records)
    rows = []
    for rec in records:
        rel = str(rec.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rows.append(extract_detector_features(rec, group_map[rel]))

    manual = load_manual_annotations(args.manual_csv)
    manual_error = attach_manual_error(rows, manual)
    write_feature_csv(rows, FEATURE_CSV)

    usable_rows = [row for row in rows if row.get("quality_ok") is True]
    class_counts = dict(Counter(row["class_label"] for row in rows))
    quality_counts = dict(Counter(row["quality_reason"] for row in rows))
    reflex_status_counts = {
        "od": dict(Counter(str(row.get("od_reflex_status")) for row in rows)),
        "os": dict(Counter(str(row.get("os_reflex_status")) for row in rows)),
    }

    binary_rows = []
    for row in usable_rows:
        copied = dict(row)
        copied["class_label"] = "strabismus" if copied["class_label"] in {"esotropia", "exotropia"} else "normal"
        binary_rows.append(copied)

    X_bin, y_bin, groups_bin, feature_names = as_matrix(binary_rows, ["normal", "strabismus"])
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])
    binary_res = repeated_group_cv(X_bin, y_bin, groups_bin, estimator, args.repeats, args.splits, SEED)
    binary_res["permutation_test"] = permutation_test(
        X_bin,
        y_bin,
        groups_bin,
        estimator,
        float(binary_res["metrics"]["balanced_accuracy"]),
        repeats=max(2, min(3, args.repeats)),
        splits=args.splits,
        n_perm=args.permutations,
        seed=SEED + 100,
    )

    perm_p = binary_res["permutation_test"]["p_value"]
    perm_p95 = binary_res["permutation_test"].get("balanced_accuracy_p95")
    observed_bal_acc = float(binary_res["metrics"]["balanced_accuracy"])
    stopped_near_random = (
        perm_p is None
        or float(perm_p) > 0.05
        or (perm_p95 is not None and observed_bal_acc <= float(perm_p95))
    )

    binary_rf = None
    direction_rows = [] if stopped_near_random else [
        row for row in usable_rows if row.get("class_label") in {"esotropia", "exotropia"}
    ]
    direction_metrics = None
    if direction_rows:
        y_true = np.array([0 if row["class_label"] == "exotropia" else 1 for row in direction_rows], dtype=int)
        y_pred = np.array([1 if float(row["signed_asymmetry_lr"]) > 0 else 0 for row in direction_rows], dtype=int)
        direction_metrics = metric_summary(y_true, y_pred)

    mpl_cache = PROCESSED_DIR / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import mediapipe as mp
    import sklearn
    results: Dict[str, object] = {
        "environment": {
            "python_env": ".venv312",
            "pytorch_added": False,
            "mediapipe": mp.__version__,
            "opencv": cv2.__version__,
            "sklearn": sklearn.__version__,
        },
        "run_config": {
            "repeats": args.repeats,
            "splits": args.splits,
            "permutations": args.permutations,
            "seed": SEED,
        },
        "dataset": {
            "total_images": len(rows),
            "class_counts": class_counts,
            "quality_counts": quality_counts,
            "reflex_status_counts": reflex_status_counts,
            "quality_usable_count": len(usable_rows),
            "group_count": len(set(group_map.values())),
            "near_duplicate_pairs": near_pairs,
        },
        "feature_names": feature_names,
        "manual_csv": str(args.manual_csv.relative_to(PROJECT_ROOT)).replace("\\", "/") if args.manual_csv.is_absolute() else str(args.manual_csv),
        "manual_detector_error": manual_error,
        "binary_logistic": binary_res,
        "binary_random_forest": binary_rf,
        "direction_rule": direction_metrics,
        "stopped_near_random": stopped_near_random,
        "stop_reason": (
            "Binary geometry signal did not exceed permutation baseline; stopped before RandomForest and eso/exo direction."
            if stopped_near_random else None
        ),
        "near_random_analysis": {
            "possible_detector_issue": "Tightened detector reduces large reflex outliers but still excludes many eyes with glare, clustered glints, low brightness, or far candidates.",
            "possible_label_issue": "Folder labels are not backed by an auditable clinician manifest in this repo.",
            "possible_geometry_issue": "The assumed sign convention signed_asymmetry_lr = os - od may not match all crops/gaze directions.",
            "possible_domain_issue": "Accepted images are legacy 224x224 crops, not production phone full-face captures.",
        } if stopped_near_random else None,
        "governance": [
            "research_candidate_only",
            "no_cnn_training",
            "no_production_api_integration",
            "source_images_not_modified",
        ],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, REPORT_MD)
    print(json.dumps({
        "feature_csv": str(FEATURE_CSV),
        "result_json": str(RESULT_JSON),
        "report_md": str(REPORT_MD),
        "binary_balanced_accuracy": binary_res["metrics"]["balanced_accuracy"],
        "permutation_p_value": binary_res["permutation_test"]["p_value"],
        "quality_usable_count": len(usable_rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
