"""Manual/oracle Hirschberg geometry analysis.

Research-only. This script does not train a deployable model and does not touch
production APIs. It reads manual annotations, compares detector reflex points
against them, exports detector failure visualizations, and evaluates whether
manual geometry alone separates folder labels.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FAILURE_DIR = REPORTS_DIR / "detector_failures"
PLOT_DIR = REPORTS_DIR / "oracle_plots"
MANUAL_CSV = PROCESSED_DIR / "hirschberg_manual_annotations.csv"
DETECTOR_FEATURE_CSV = PROCESSED_DIR / "hirschberg_geometry_features.csv"
RESULT_JSON = REPORTS_DIR / "hirschberg_oracle_results.json"
REPORT_MD = REPORTS_DIR / "hirschberg_oracle_report.md"
SEED = 20261004

os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED_DIR / "matplotlib_cache"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class ManualRow:
    relative_path: str
    class_label: str
    od_pupil: Tuple[float, float]
    od_reflex: Tuple[float, float]
    os_pupil: Tuple[float, float]
    os_reflex: Tuple[float, float]


def read_manual(path: Path) -> List[ManualRow]:
    rows: List[ManualRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ManualRow(
                    relative_path=row["relative_path"],
                    class_label=row["class_label"],
                    od_pupil=(float(row["od_pupil_x"]), float(row["od_pupil_y"])),
                    od_reflex=(float(row["od_reflex_x"]), float(row["od_reflex_y"])),
                    os_pupil=(float(row["os_pupil_x"]), float(row["os_pupil_y"])),
                    os_reflex=(float(row["os_reflex_x"]), float(row["os_reflex_y"])),
                )
            )
    return rows


def read_detector_features(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}


def point(row: Dict[str, str], x_key: str, y_key: str) -> Optional[Tuple[float, float]]:
    try:
        if row.get(x_key, "") == "" or row.get(y_key, "") == "":
            return None
        return float(row[x_key]), float(row[y_key])
    except ValueError:
        return None


def oracle_features(row: ManualRow) -> Dict[str, object]:
    od_p = np.array(row.od_pupil, dtype=float)
    os_p = np.array(row.os_pupil, dtype=float)
    od_r = np.array(row.od_reflex, dtype=float)
    os_r = np.array(row.os_reflex, dtype=float)
    axis = os_p - od_p
    ipd = float(np.linalg.norm(axis))
    if ipd <= 1e-6:
        unit = np.array([1.0, 0.0])
    else:
        unit = axis / ipd

    # Positive means nasal direction for both eyes. OD nasal is toward OS
    # (+unit); OS nasal is toward OD (-unit), hence the mirrored sign.
    od_offset = float(np.dot(od_r - od_p, unit) / max(ipd, 1e-6))
    os_offset = float(-np.dot(os_r - os_p, unit) / max(ipd, 1e-6))
    signed_asymmetry = float(os_offset - od_offset)
    abs_asymmetry = abs(signed_asymmetry)
    mean_offset = float((od_offset + os_offset) / 2.0)
    mean_abs_offset = float((abs(od_offset) + abs(os_offset)) / 2.0)
    return {
        "relative_path": row.relative_path,
        "class_label": row.class_label,
        "binary_label": "strabismus" if row.class_label in {"esotropia", "exotropia"} else "normal",
        "ipd_px": ipd,
        "od_offset_nasal_ipd": od_offset,
        "os_offset_nasal_ipd": os_offset,
        "signed_asymmetry_nasal_ipd": signed_asymmetry,
        "abs_asymmetry_nasal_ipd": abs_asymmetry,
        "mean_offset_nasal_ipd": mean_offset,
        "mean_abs_offset_nasal_ipd": mean_abs_offset,
    }


def save_oracle_feature_csv(features: Sequence[Dict[str, object]]) -> Path:
    path = PROCESSED_DIR / "hirschberg_oracle_features.csv"
    fieldnames = list(features[0].keys()) if features else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
    return path


def detector_errors(manual_rows: Sequence[ManualRow], detector_rows: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    errors: List[Dict[str, object]] = []
    for row in manual_rows:
        det = detector_rows.get(row.relative_path)
        if not det:
            continue
        for eye, manual_pt, x_key, y_key in [
            ("od", row.od_reflex, "od_reflex_x", "od_reflex_y"),
            ("os", row.os_reflex, "os_reflex_x", "os_reflex_y"),
        ]:
            det_pt = point(det, x_key, y_key)
            if det_pt is None:
                continue
            err = math.hypot(det_pt[0] - manual_pt[0], det_pt[1] - manual_pt[1])
            errors.append({
                "relative_path": row.relative_path,
                "class_label": row.class_label,
                "eye": eye,
                "manual_x": manual_pt[0],
                "manual_y": manual_pt[1],
                "detector_x": det_pt[0],
                "detector_y": det_pt[1],
                "error_px": err,
            })
    return errors


def classify_failure(image_bgr: np.ndarray, failure: Dict[str, object]) -> str:
    det = np.array([float(failure["detector_x"]), float(failure["detector_y"])])
    manual = np.array([float(failure["manual_x"]), float(failure["manual_y"])])
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    half = w // 2
    x1, x2 = (0, half) if failure["eye"] == "od" else (half, w)
    eye_gray = gray[:, x1:x2]
    local_det_x = int(round(det[0] - x1))
    local_manual_x = int(round(manual[0] - x1))
    _, binary = cv2.threshold(eye_gray, 185, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        if area >= 1:
            comps.append((area, bw, bh, cx, cy))
    near_det = [
        c for c in comps
        if math.hypot(c[3] - local_det_x, c[4] - det[1]) <= 8.0
    ]
    near_manual = [
        c for c in comps
        if math.hypot(c[3] - local_manual_x, c[4] - manual[1]) <= 8.0
    ]
    big_or_long = any(area > 45 or max(bw, bh) / max(1, min(bw, bh)) > 3.0 for area, bw, bh, _, _ in near_det)
    multiple = len([c for c in comps if c[0] >= 2]) >= 3
    vertical_delta = float(det[1] - manual[1])
    if big_or_long:
        return "large_or_elongated_glare_environment_or_eyelid"
    if multiple:
        return "double_or_multiple_reflex_candidates"
    if abs(vertical_delta) > 25:
        return "eyelid_or_skin_bright_spot"
    if near_manual and near_det:
        return "selected_wrong_nearby_specular_spot"
    return "wrong_bright_spot_or_detector_center_error"


def draw_failure(image_bgr: np.ndarray, failure: Dict[str, object], cause: str, output_path: Path) -> None:
    img = image_bgr.copy()
    mx, my = int(round(float(failure["manual_x"]))), int(round(float(failure["manual_y"])))
    dx, dy = int(round(float(failure["detector_x"]))), int(round(float(failure["detector_y"])))
    cv2.circle(img, (mx, my), 7, (0, 255, 0), 2)
    cv2.circle(img, (dx, dy), 7, (0, 0, 255), 2)
    cv2.line(img, (mx, my), (dx, dy), (0, 255, 255), 1)
    text = f"{failure['class_label']} {failure['eye']} err={failure['error_px']:.1f}px {cause}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text[:80], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def export_detector_failures(manual_rows: Sequence[ManualRow], failures: Sequence[Dict[str, object]], p90: float) -> Tuple[List[Dict[str, object]], Path]:
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in FAILURE_DIR.glob("*.jpg"):
        old_path.unlink()
    old_csv = FAILURE_DIR / "detector_failures_p90.csv"
    if old_csv.exists():
        old_csv.unlink()

    rows_by_path = {row.relative_path: row for row in manual_rows}
    selected = [dict(f) for f in failures if float(f["error_px"]) >= p90]
    for idx, failure in enumerate(selected, 1):
        image_path = PROJECT_ROOT / str(failure["relative_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            failure["cause"] = "cannot_read_image"
            continue
        cause = classify_failure(image, failure)
        failure["cause"] = cause
        safe_name = f"{idx:02d}_{failure['class_label']}_{failure['eye']}_{Path(str(failure['relative_path'])).stem}_{failure['error_px']:.1f}px.jpg"
        out = FAILURE_DIR / safe_name
        draw_failure(image, failure, cause, out)
        failure["failure_image"] = str(out.relative_to(PROJECT_ROOT)).replace("\\", "/")

    csv_path = FAILURE_DIR / "detector_failures_p90.csv"
    fieldnames = sorted({k for f in selected for k in f.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected, csv_path


def plot_feature_distributions(features: Sequence[Dict[str, object]]) -> List[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "od_offset_nasal_ipd",
        "os_offset_nasal_ipd",
        "signed_asymmetry_nasal_ipd",
        "abs_asymmetry_nasal_ipd",
        "mean_offset_nasal_ipd",
        "mean_abs_offset_nasal_ipd",
    ]
    labels = ["normal", "esotropia", "exotropia"]
    saved = []
    for feat in feature_names:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        data = [[float(row[feat]) for row in features if row["class_label"] == label] for label in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        rng = np.random.default_rng(SEED)
        for i, vals in enumerate(data, 1):
            if not vals:
                continue
            x = rng.normal(i, 0.035, size=len(vals))
            ax.scatter(x, vals, alpha=0.75, s=24)
        ax.axhline(0, color="#666666", linewidth=1)
        ax.set_title(feat)
        ax.set_ylabel("normalized by pupil distance")
        fig.tight_layout()
        out = PLOT_DIR / f"{feat}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        saved.append(str(out.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return saved


def group_ids_from_detector_rows(features: Sequence[Dict[str, object]], detector_rows: Dict[str, Dict[str, str]]) -> np.ndarray:
    groups = []
    for row in features:
        det = detector_rows.get(str(row["relative_path"]), {})
        groups.append(det.get("group_id") or str(row["relative_path"]))
    return np.array(groups)


def threshold_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, float, int]:
    candidates = np.unique(x_train)
    if len(candidates) == 0:
        return np.zeros_like(x_test, dtype=int), 0.0, 1
    thresholds = []
    sorted_vals = np.sort(candidates)
    thresholds.append(float(sorted_vals[0] - 1e-9))
    thresholds.extend(float((a + b) / 2.0) for a, b in zip(sorted_vals[:-1], sorted_vals[1:]))
    thresholds.append(float(sorted_vals[-1] + 1e-9))
    best_score = -1.0
    best_threshold = thresholds[0]
    best_direction = 1
    pos = y_train == 1
    neg = y_train == 0
    n_pos = max(1, int(np.sum(pos)))
    n_neg = max(1, int(np.sum(neg)))
    for threshold in thresholds:
        for direction in (1, -1):
            pred = (x_train >= threshold).astype(int) if direction == 1 else (x_train <= threshold).astype(int)
            tp = int(np.sum((pred == 1) & pos))
            tn = int(np.sum((pred == 0) & neg))
            score = 0.5 * ((tp / n_pos) + (tn / n_neg))
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_direction = direction
    pred_test = (x_test >= best_threshold).astype(int) if best_direction == 1 else (x_test <= best_threshold).astype(int)
    return pred_test, best_threshold, best_direction


def repeated_threshold_cv(
    features: Sequence[Dict[str, object]],
    groups: np.ndarray,
    feature_name: str,
    repeats: int,
    splits: int,
    seed: int,
    y_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    x = np.array([float(row[feature_name]) for row in features], dtype=float)
    y = np.array([0 if row["binary_label"] == "normal" else 1 for row in features], dtype=int)
    if y_override is not None:
        y = y_override.astype(int)
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    groups_all: List[str] = []
    thresholds: List[float] = []
    directions: List[int] = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + rep)
        for train_idx, test_idx in cv.split(x.reshape(-1, 1), y, groups):
            pred, threshold, direction = threshold_fit_predict(x[train_idx], y[train_idx], x[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
            groups_all.extend(groups[test_idx].tolist())
            thresholds.append(threshold)
            directions.append(direction)
    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)
    out = {
        "feature": feature_name,
        "n_predictions": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold_median": float(np.median(thresholds)),
        "direction_counts": dict(Counter(directions)),
    }
    try:
        out["roc_auc_feature"] = float(roc_auc_score(y_true, [float(row[feature_name]) for row in features for _ in range(0)]))
    except Exception:
        pass
    out["bootstrap_ci"] = bootstrap_ci(y_true, y_pred, np.array(groups_all), seed=seed + 500)
    return out


def build_cv_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, repeats: int, splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    cv_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + rep)
        cv_splits.extend(list(cv.split(x.reshape(-1, 1), y, groups)))
    return cv_splits


def threshold_cv_with_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    for train_idx, test_idx in cv_splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        pred, _, _ = threshold_fit_predict(x[train_idx], y[train_idx], x[test_idx])
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
    if not y_true_all or len(np.unique(y_true_all)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(np.array(y_true_all), np.array(y_pred_all)))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, seed: int, n_boot: int = 1000) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    bal, f1s = [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        bal.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {
        "balanced_accuracy": {
            "low": float(np.percentile(bal, 2.5)),
            "high": float(np.percentile(bal, 97.5)),
        },
        "macro_f1": {
            "low": float(np.percentile(f1s, 2.5)),
            "high": float(np.percentile(f1s, 97.5)),
        },
    }


def permutation_test(
    features: Sequence[Dict[str, object]],
    groups: np.ndarray,
    feature_name: str,
    observed: float,
    repeats: int,
    splits: int,
    n_perm: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    x = np.array([float(row[feature_name]) for row in features], dtype=float)
    y = np.array([0 if row["binary_label"] == "normal" else 1 for row in features], dtype=int)
    unique_groups = np.unique(groups)
    group_labels = {g: int(Counter(y[groups == g]).most_common(1)[0][0]) for g in unique_groups}
    cv_splits = build_cv_splits(x, y, groups, repeats=max(2, min(3, repeats)), splits=splits, seed=seed + 1)
    vals = []
    for i in range(n_perm):
        shuffled = rng.permutation([group_labels[g] for g in unique_groups])
        shuffled_map = dict(zip(unique_groups, shuffled))
        y_perm = np.array([shuffled_map[g] for g in groups], dtype=int)
        score = threshold_cv_with_splits(x, y_perm, groups, cv_splits)
        if math.isfinite(score):
            vals.append(score)
    arr = np.array(vals, dtype=float)
    return {
        "n_permutations": int(len(arr)),
        "p_value": float((np.sum(arr >= observed) + 1) / (len(arr) + 1)),
        "balanced_accuracy_mean": float(np.mean(arr)),
        "balanced_accuracy_p95": float(np.percentile(arr, 95)),
    }


def create_grid(rows: Sequence[Dict[str, object]], output_path: Path, title_key: str) -> str:
    thumbs = []
    cell_w, cell_h = 190, 170
    for row in rows:
        image = cv2.imread(str(PROJECT_ROOT / str(row["relative_path"])))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = min((cell_w - 8) / w, (cell_h - 30) / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
        y0 = 4
        x0 = (cell_w - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        text = f"{row['class_label']} {float(row[title_key]):.3f}"
        cv2.putText(canvas, text[:28], (4, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        thumbs.append(canvas)
    if not thumbs:
        return ""
    cols = 5
    grid_rows = math.ceil(len(thumbs) / cols)
    grid = np.full((grid_rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * cell_h
        x = (i % cols) * cell_w
        grid[y:y + cell_h, x:x + cell_w] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def hypothesis_grids(features: Sequence[Dict[str, object]]) -> Dict[str, object]:
    normal = [row for row in features if row["class_label"] == "normal"]
    strab = [row for row in features if row["binary_label"] == "strabismus"]
    normal_sorted = sorted(normal, key=lambda r: float(r["abs_asymmetry_nasal_ipd"]), reverse=True)
    strab_sorted = sorted(strab, key=lambda r: float(r["abs_asymmetry_nasal_ipd"]))
    out = {
        "normal_high_asymmetry": normal_sorted[:10],
        "strabismus_low_asymmetry": strab_sorted[:10],
    }
    return {
        "normal_high_asymmetry_count": len(out["normal_high_asymmetry"]),
        "strabismus_low_asymmetry_count": len(out["strabismus_low_asymmetry"]),
        "normal_high_asymmetry_grid": create_grid(
            out["normal_high_asymmetry"],
            REPORTS_DIR / "normal_high_oracle_asymmetry_grid.jpg",
            "abs_asymmetry_nasal_ipd",
        ),
        "strabismus_low_asymmetry_grid": create_grid(
            out["strabismus_low_asymmetry"],
            REPORTS_DIR / "strabismus_low_oracle_asymmetry_grid.jpg",
            "abs_asymmetry_nasal_ipd",
        ),
        "environment_reflection_suspect_count": 0,
        "environment_reflection_note": "Needs human review of exported grids/failure images; automatic count is not reliable enough to label flash vs ambient reflection.",
    }


def write_report(results: Dict[str, object]) -> None:
    oracle = results["oracle_cv"]
    perm = results["oracle_permutation"]
    det = results["detector_failure_summary"]
    lines = [
        "# Hirschberg Oracle Manual Geometry Analysis",
        "",
        "Status: research-only; no CNN; no production API integration; no deployable model artifact.",
        "",
        "## Detector Failure Analysis",
        "",
        f"- Manual images: {results['manual_count']}",
        f"- Reflex point comparisons: {det['error_count']}",
        f"- P90 threshold: {det['p90_error_px']:.2f}px",
        f"- P90+ failures exported: {det['p90_failure_count']}",
        f"- Failure folder: `reports/detector_failures/`",
        f"- Cause counts: `{det['cause_counts']}`",
        "",
        "Recommended exclusions before using detector features:",
        "",
        "- Reject images when detected reflex is on a large/elongated bright component.",
        "- Reject or flag images with multiple bright candidates near the iris.",
        "- Require detector/manual-like consistency checks: compact area, aspect ratio, and distance from pupil/iris center.",
        "- Keep manual/oracle analysis separate until detector P90 error is much lower.",
        "",
        "## Oracle Geometry",
        "",
        "Offsets use manual pupil and reflex points. The horizontal axis is the line connecting the two pupils. Values are normalized by interpupillary distance. OD and OS are mirrored so positive is nasal for both eyes.",
        "",
        f"- Feature evaluated: `{oracle['feature']}`",
        f"- Balanced accuracy: {oracle['balanced_accuracy']:.4f}",
        f"- Macro-F1: {oracle['macro_f1']:.4f}",
        f"- Bootstrap CI: `{oracle['bootstrap_ci']}`",
        f"- Permutation p-value: {perm['p_value']}",
        f"- Permutation p95: {perm['balanced_accuracy_p95']:.4f}",
        f"- Confusion matrix: `{oracle['confusion_matrix']}`",
        "",
        "## Detector vs Oracle",
        "",
        f"- Detector balanced accuracy from previous report: {results['detector_previous']['balanced_accuracy']:.4f}",
        f"- Detector permutation p-value: {results['detector_previous']['permutation_p_value']}",
        f"- Oracle balanced accuracy: {oracle['balanced_accuracy']:.4f}",
        "",
        "## Label / Image Hypothesis Checks",
        "",
        f"- Normal high-asymmetry grid: `{results['hypothesis_grids']['normal_high_asymmetry_grid']}`",
        f"- Strabismus low-asymmetry grid: `{results['hypothesis_grids']['strabismus_low_asymmetry_grid']}`",
        f"- Ambient-reflection automatic count: {results['hypothesis_grids']['environment_reflection_suspect_count']}",
        f"- Note: {results['hypothesis_grids']['environment_reflection_note']}",
        "",
        "## Conclusion",
        "",
    ]
    if perm["p_value"] > 0.05:
        lines.extend([
            "Oracle geometry is also not convincingly above random. This points away from detector-only error as the main cause. The likely blockers are label quality/provenance, geometry sign/rule mismatch for these crops, and the dataset itself.",
            "",
            "Stop here. Do not try additional models on this dataset. Collect a new protocol dataset with clinician labels, participant IDs, phone flash full-face/crop pipeline consistency, and explicit notes about ambient reflections/double reflexes.",
            "",
        ])
    else:
        lines.extend([
            "Oracle geometry is above random while detector geometry was near random. This would indicate detector error is the primary bottleneck before any modeling.",
            "",
        ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual/oracle Hirschberg analysis.")
    parser.add_argument("--manual-csv", type=Path, default=MANUAL_CSV)
    parser.add_argument("--detector-csv", type=Path, default=DETECTOR_FEATURE_CSV)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    (PROCESSED_DIR / "matplotlib_cache").mkdir(exist_ok=True)

    manual_rows = read_manual(args.manual_csv)
    detector_rows = read_detector_features(args.detector_csv)
    features = [oracle_features(row) for row in manual_rows]
    oracle_csv = save_oracle_feature_csv(features)
    plot_paths = plot_feature_distributions(features)

    errors = detector_errors(manual_rows, detector_rows)
    err_arr = np.array([float(e["error_px"]) for e in errors], dtype=float)
    p90 = float(np.percentile(err_arr, 90)) if len(err_arr) else float("nan")
    failures, failure_csv = export_detector_failures(manual_rows, errors, p90)
    cause_counts = dict(Counter(str(f.get("cause")) for f in failures))

    groups = group_ids_from_detector_rows(features, detector_rows)
    oracle_cv = repeated_threshold_cv(
        features,
        groups,
        "abs_asymmetry_nasal_ipd",
        repeats=args.repeats,
        splits=args.splits,
        seed=SEED,
    )
    perm = permutation_test(
        features,
        groups,
        "abs_asymmetry_nasal_ipd",
        observed=float(oracle_cv["balanced_accuracy"]),
        repeats=args.repeats,
        splits=args.splits,
        n_perm=args.permutations,
        seed=SEED + 99,
    )
    hyp = hypothesis_grids(features)
    hyp["environment_reflection_suspect_count"] = int(
        cause_counts.get("large_or_elongated_glare_environment_or_eyelid", 0)
    )
    hyp["environment_reflection_note"] = (
        "Count is based on P90 detector-failure overlays classified as large/elongated glare; "
        "human review is still required to separate ambient reflection from eyelid/skin glare."
    )

    detector_previous = {"balanced_accuracy": None, "permutation_p_value": None}
    prev_path = REPORTS_DIR / "hirschberg_geometry_research_results.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        detector_previous = {
            "balanced_accuracy": prev["binary_logistic"]["metrics"]["balanced_accuracy"],
            "permutation_p_value": prev["binary_logistic"]["permutation_test"]["p_value"],
        }

    results = {
        "manual_count": len(manual_rows),
        "oracle_feature_csv": str(oracle_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "plots": plot_paths,
        "detector_failure_summary": {
            "error_count": len(errors),
            "mean_error_px": float(np.mean(err_arr)) if len(err_arr) else None,
            "median_error_px": float(np.median(err_arr)) if len(err_arr) else None,
            "p90_error_px": p90,
            "p90_failure_count": len(failures),
            "cause_counts": cause_counts,
            "failure_csv": str(failure_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "oracle_cv": oracle_cv,
        "oracle_permutation": perm,
        "hypothesis_grids": hyp,
        "detector_previous": detector_previous,
        "governance": [
            "research_only",
            "manual_oracle_geometry",
            "no_cnn",
            "no_production_api_integration",
            "no_source_data_modification",
        ],
    }
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps({
        "report": str(REPORT_MD),
        "result_json": str(RESULT_JSON),
        "failure_dir": str(FAILURE_DIR),
        "oracle_balanced_accuracy": oracle_cv["balanced_accuracy"],
        "oracle_permutation_p_value": perm["p_value"],
        "p90_failure_count": len(failures),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
