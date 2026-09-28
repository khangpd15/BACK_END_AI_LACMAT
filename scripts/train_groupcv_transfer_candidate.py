"""Train a leakage-resistant Korean-to-RemiCare transfer candidate.

The candidate is selected with nested participant-group cross-validation and
probability metrics. It is never copied over the production model. No threshold
tuning or probability calibration is performed because RemiCare has no labeled
validation set.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.shared_feature_contract import ALL_SHARED_FEATURES  # noqa: E402


TRAIN_CSV = ROOT / "data" / "processed" / "korean_shared_train.csv"
TEST_CSV = ROOT / "data" / "processed" / "korean_shared_test.csv"
REMICARE_SAMPLE = ROOT / "data" / "processed" / "remicare_shared_sample.json"
CURRENT_B2 = ROOT / "app" / "models" / "remicare_b2_rescaled_model.joblib"
OUTPUT_DIR = ROOT / "outputs" / "retraining" / "groupcv_candidate"
MODEL_DIR = ROOT / "models" / "candidates"
MODEL_PATH = MODEL_DIR / "korean_groupcv_transfer_candidate.joblib"

HORIZONTAL = [
    "meanDeltaX",
    "medianDeltaX",
    "stdDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "rangeDeltaX",
    "meanAbsDeltaX",
]
VIEWPORT_POSITIONS = ["meanLeftX", "meanLeftY", "meanRightX", "meanRightY"]
LOWER_SHIFT = [
    "leftValidRatio",
    "rightValidRatio",
    "bothValidRatio",
    "meanDeltaY",
    "medianDeltaY",
    "stdDeltaY",
    "minDeltaY",
    "maxDeltaY",
    "rangeDeltaY",
    "meanAbsDeltaY",
    "stdLeftX",
    "stdLeftY",
    "stdRightX",
    "stdRightY",
]
IPD_SCALE_FACTOR = 0.33009 / 0.13190

FEATURE_CONFIGS: dict[str, list[str]] = {
    "B2_30_RESCALED": list(ALL_SHARED_FEATURES),
    "B2_NO_VIEWPORT_26": [f for f in ALL_SHARED_FEATURES if f not in VIEWPORT_POSITIONS],
    "LOWER_SHIFT_14": LOWER_SHIFT,
}
REGULARIZATION_C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CLASS_WEIGHTS: list[str | None] = [None, "balanced"]


def participant_id(sample_id: str) -> str:
    """Apply the repository's Korean participant-name extraction rule."""
    match = re.search(r"[\uac00-\ud7a3]+", sample_id)
    if not match:
        return sample_id
    name = match.group(0)
    for suffix in ["정상", "좌내", "우내", "좌외", "우외", "정", "외", "내"]:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[: -len(suffix)]
    # Some filenames separate a caregiver marker with an underscore, e.g.
    # 유대규_엄마. Treat the caregiver and child as different participants.
    for caregiver_marker in ["어머니", "엄마"]:
        if caregiver_marker in sample_id and not name.endswith(caregiver_marker):
            name += caregiver_marker
            break
    return name


def apply_b2_training_transform(matrix: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Map Korean horizontal coordinates down to the RemiCare coordinate scale."""
    transformed = matrix.copy()
    for name in HORIZONTAL:
        if name in feature_names:
            transformed[:, feature_names.index(name)] /= IPD_SCALE_FACTOR
    return transformed


def prepare_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    matrix = np.nan_to_num(
        frame[feature_names].to_numpy(dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return apply_b2_training_transform(matrix, feature_names)


def make_model(c_value: float, class_weight: str | None) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=3000,
            random_state=42,
        ),
    )


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probability[mask]))
        value += float(np.sum(mask)) / total * abs(observed - predicted)
    return value


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1]).tolist()
    return {
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "ece_10_bins": expected_calibration_error(y_true, probability),
        "accuracy_at_0_5": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, prediction)),
        "precision_at_0_5": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix_at_0_5": cm,
    }


def candidate_key(config: dict[str, Any]) -> str:
    weight = config["class_weight"] or "none"
    return f'{config["feature_config"]}|C={config["C"]}|weight={weight}'


def all_candidates() -> list[dict[str, Any]]:
    return [
        {"feature_config": feature_config, "C": c_value, "class_weight": class_weight}
        for feature_config in FEATURE_CONFIGS
        for c_value in REGULARIZATION_C
        for class_weight in CLASS_WEIGHTS
    ]


def evaluate_inner(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    row_indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> float:
    names = FEATURE_CONFIGS[config["feature_config"]]
    matrix = prepare_matrix(frame.iloc[row_indices], names)
    subset_y = y[row_indices]
    subset_groups = groups[row_indices]
    unique_groups = len(np.unique(subset_groups))
    n_splits = min(4, unique_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.full(len(row_indices), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(matrix, subset_y, subset_groups):
        model = make_model(config["C"], config["class_weight"])
        model.fit(matrix[train_idx], subset_y[train_idx])
        probabilities[valid_idx] = model.predict_proba(matrix[valid_idx])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Inner group-CV did not produce a probability for every row")
    return float(brier_score_loss(subset_y, probabilities))


def select_on_indices(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    row_indices: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = []
    for config in all_candidates():
        brier = evaluate_inner(frame, y, groups, row_indices, config, seed)
        scores.append({**config, "mean_inner_brier": brier, "key": candidate_key(config)})
    scores.sort(key=lambda item: (item["mean_inner_brier"], item["feature_config"], item["C"], str(item["class_weight"])))
    return scores[0], scores


def to_native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(to_native(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    original_train = pd.read_csv(TRAIN_CSV)
    original_test = pd.read_csv(TEST_CSV)
    frame = pd.concat([original_train, original_test], ignore_index=True)
    original_row_count = len(frame)
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["sampleId"].map(participant_id).to_numpy(dtype=str)

    group_labels = pd.DataFrame({"group": groups, "label": y}).groupby("group")["label"].nunique()
    mixed_label_groups = group_labels[group_labels > 1].index.tolist()
    excluded_conflicting_rows: list[dict[str, Any]] = []
    if mixed_label_groups:
        conflict_mask = np.isin(groups, mixed_label_groups)
        excluded_conflicting_rows = [
            {
                "sampleId": frame.iloc[index]["sampleId"],
                "participantId": groups[index],
                "label": int(y[index]),
            }
            for index in np.flatnonzero(conflict_mask)
        ]
        frame = frame.loc[~conflict_mask].reset_index(drop=True)
        y = frame["label"].to_numpy(dtype=int)
        groups = frame["sampleId"].map(participant_id).to_numpy(dtype=str)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_probability = np.full(len(frame), np.nan, dtype=float)
    outer_folds: list[dict[str, Any]] = []
    all_indices = np.arange(len(frame))

    # Use a placeholder feature matrix for split generation; grouping and labels
    # alone determine StratifiedGroupKFold membership.
    placeholder = np.zeros((len(frame), 1), dtype=float)
    for fold_index, (train_idx, valid_idx) in enumerate(outer.split(placeholder, y, groups), start=1):
        selected, _ = select_on_indices(frame, y, groups, train_idx, seed=100 + fold_index)
        names = FEATURE_CONFIGS[selected["feature_config"]]
        train_x = prepare_matrix(frame.iloc[train_idx], names)
        valid_x = prepare_matrix(frame.iloc[valid_idx], names)
        model = make_model(selected["C"], selected["class_weight"])
        model.fit(train_x, y[train_idx])
        fold_probability = model.predict_proba(valid_x)[:, 1]
        outer_probability[valid_idx] = fold_probability
        outer_folds.append({
            "fold": fold_index,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx),
            "train_participants": len(np.unique(groups[train_idx])),
            "validation_participants": len(np.unique(groups[valid_idx])),
            "participant_overlap": sorted(set(groups[train_idx]).intersection(groups[valid_idx])),
            "selected_config": selected,
            "metrics": probability_metrics(y[valid_idx], fold_probability),
        })

    if np.isnan(outer_probability).any():
        raise RuntimeError("Outer nested group-CV did not produce a probability for every row")
    nested_metrics = probability_metrics(y, outer_probability)

    # Like-for-like reference: evaluate the legacy B2 LR configuration
    # (30 rescaled features, C=1, balanced weighting) on the exact same outer
    # participant folds. The production artifact itself was not refit or changed.
    legacy_probability = np.full(len(frame), np.nan, dtype=float)
    legacy_names = FEATURE_CONFIGS["B2_30_RESCALED"]
    legacy_x = prepare_matrix(frame, legacy_names)
    legacy_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in legacy_outer.split(placeholder, y, groups):
        legacy_model = make_model(1.0, "balanced")
        legacy_model.fit(legacy_x[train_idx], y[train_idx])
        legacy_probability[valid_idx] = legacy_model.predict_proba(legacy_x[valid_idx])[:, 1]
    legacy_groupcv_metrics = probability_metrics(y, legacy_probability)

    final_config, final_search = select_on_indices(frame, y, groups, all_indices, seed=4242)
    final_names = FEATURE_CONFIGS[final_config["feature_config"]]
    final_x = prepare_matrix(frame, final_names)
    final_model = make_model(final_config["C"], final_config["class_weight"])
    final_model.fit(final_x, y)

    remicare_payload = json.loads(REMICARE_SAMPLE.read_text(encoding="utf-8"))
    remicare_features = remicare_payload["features"]
    # B2 training transforms Korean rows down to RemiCare coordinate scale;
    # RemiCare input is therefore passed as-is.
    remicare_x = np.asarray([[remicare_features[name] for name in final_names]], dtype=float)
    remicare_probability = final_model.predict_proba(remicare_x)[0]
    remicare_score = float(final_model.decision_function(remicare_x)[0])

    current_artifact = joblib.load(CURRENT_B2)
    current_names = current_artifact["feature_names"]
    current_x = np.asarray([[remicare_features[name] for name in current_names]], dtype=float)
    current_probability = current_artifact["model"].predict_proba(current_x)[0]

    artifact = {
        "model": final_model,
        "name": "Logistic Regression (nested participant-group CV candidate)",
        "version": "korean-groupcv-transfer-candidate-v1.0.0",
        "status": "CANDIDATE_NOT_FOR_PRODUCTION",
        "feature_names": final_names,
        "feature_count": len(final_names),
        "feature_config": final_config["feature_config"],
        "ipd_scale_factor": IPD_SCALE_FACTOR,
        "training_transform": "Korean horizontal features divided by ipd_scale_factor when present; RemiCare inference as-is.",
        "selected_hyperparameters": {
            "C": final_config["C"],
            "class_weight": final_config["class_weight"],
        },
        "training_rows": len(frame),
        "source_rows_before_conflict_exclusion": original_row_count,
        "training_participants": len(np.unique(groups)),
        "label_frequencies": {
            "NORMAL": int(np.sum(y == 0)),
            "STRABISMUS": int(np.sum(y == 1)),
        },
        "nested_participant_group_cv_metrics": nested_metrics,
        "calibration_applied": False,
        "remicare_clinical_labels": 0,
        "excluded_conflicting_participants": mixed_label_groups,
        "excluded_conflicting_rows": excluded_conflicting_rows,
        "clinical_meaning": None,
        "notice": "Research transfer candidate only. Not a diagnosis and not a validated RemiCare clinical probability.",
    }
    joblib.dump(artifact, MODEL_PATH)

    oof_rows = [
        {
            "sampleId": frame.iloc[index]["sampleId"],
            "participantId": groups[index],
            "label": int(y[index]),
            "nested_oof_p_strabismus": float(outer_probability[index]),
        }
        for index in range(len(frame))
    ]
    results = {
        "method": "5-fold nested StratifiedGroupKFold by participant",
        "source_rows_before_conflict_exclusion": original_row_count,
        "total_rows": len(frame),
        "unique_participants": len(np.unique(groups)),
        "excluded_conflicting_participants": mixed_label_groups,
        "excluded_conflicting_rows": excluded_conflicting_rows,
        "participant_overlap_in_each_outer_fold": [fold["participant_overlap"] for fold in outer_folds],
        "nested_oof_metrics": nested_metrics,
        "legacy_b2_config_same_outer_groupcv_metrics": legacy_groupcv_metrics,
        "outer_folds": outer_folds,
        "final_selected_config": final_config,
        "full_search_ranked_by_brier": final_search,
        "calibration_applied": False,
        "threshold_tuned": False,
    }
    inference = {
        "sample_id": remicare_payload["sampleId"],
        "ground_truth": None,
        "domain_shift": "WARNING",
        "current_b2": {
            "p_normal": float(current_probability[0]),
            "p_strabismus": float(current_probability[1]),
        },
        "groupcv_candidate": {
            "decision_score": remicare_score,
            "p_normal": float(remicare_probability[0]),
            "p_strabismus": float(remicare_probability[1]),
        },
        "interpretation": "Technical model output only; probability accuracy cannot be measured without a RemiCare clinical label.",
    }

    write_json(OUTPUT_DIR / "nested_group_cv_results.json", results)
    write_json(OUTPUT_DIR / "nested_oof_predictions.json", oof_rows)
    write_json(OUTPUT_DIR / "remicare_candidate_inference.json", inference)

    class_weight_label = final_config["class_weight"] or "None"
    report = f"""# Participant-Group Retraining Report

## Outcome

A new **candidate-only** Logistic Regression model was trained and saved at `models/candidates/korean_groupcv_transfer_candidate.joblib`. The production B2 model was not modified.

Selected configuration:

- Feature set: `{final_config['feature_config']}` ({len(final_names)} features)
- `C`: `{final_config['C']}`
- `class_weight`: `{class_weight_label}`
- Korean training rows: `{len(frame)}`
- Rows excluded for contradictory labels: `{len(excluded_conflicting_rows)}` from participants `{mixed_label_groups}`
- Unique participants: `{len(np.unique(groups))}`
- Participant overlap inside every outer train/validation fold: `0`
- Calibration applied: `False`
- Threshold tuning: `False`

## Nested unseen-participant probability metrics

- Brier score (lower is better): `{nested_metrics['brier_score']:.8f}`
- Log loss (lower is better): `{nested_metrics['log_loss']:.8f}`
- ROC AUC: `{nested_metrics['roc_auc']:.8f}`
- ECE, 10 bins: `{nested_metrics['ece_10_bins']:.8f}`
- Accuracy at 0.5: `{nested_metrics['accuracy_at_0_5']:.8f}`
- Balanced accuracy at 0.5: `{nested_metrics['balanced_accuracy_at_0_5']:.8f}`
- F1 at 0.5: `{nested_metrics['f1_at_0_5']:.8f}`
- Confusion matrix: `{nested_metrics['confusion_matrix_at_0_5']}`

Legacy B2 configuration on the same outer participant folds:

- Brier score: `{legacy_groupcv_metrics['brier_score']:.8f}`
- Log loss: `{legacy_groupcv_metrics['log_loss']:.8f}`
- ROC AUC: `{legacy_groupcv_metrics['roc_auc']:.8f}`
- ECE, 10 bins: `{legacy_groupcv_metrics['ece_10_bins']:.8f}`

The nested candidate improves Korean unseen-participant Brier by `{legacy_groupcv_metrics['brier_score'] - nested_metrics['brier_score']:.8f}` relative to the legacy configuration on the same folds. Its log loss changes by `{nested_metrics['log_loss'] - legacy_groupcv_metrics['log_loss']:+.8f}` (positive means worse), so the candidate is not uniformly better across probability metrics.

These are Korean source-domain, unseen-participant estimates. They are not RemiCare validation metrics.

## Same RemiCare sample

- Current B2 `P(STRABISMUS)`: `{float(current_probability[1]):.12f}`
- Candidate `P(STRABISMUS)`: `{float(remicare_probability[1]):.12f}`
- RemiCare ground truth: `NONE`
- Domain shift: `WARNING`

The candidate probability being numerically lower or higher does not make it more clinically accurate. There is no RemiCare label against which to measure its error.

## Deployment decision

**DO NOT replace the production model yet.** The candidate removes participant leakage from validation and optimizes uncalibrated Korean probability quality, but target-domain probability accuracy remains unknown. Promotion requires an independent, participant-level RemiCare validation set with ophthalmologist-confirmed labels and a pre-specified evaluation protocol.

No clinical conclusion can be drawn from either probability.
"""
    (OUTPUT_DIR / "RETRAINING_REPORT.md").write_text(report, encoding="utf-8")

    print(f"Candidate model: {MODEL_PATH}")
    print(f"Selected config: {candidate_key(final_config)}")
    print(f"Nested participant-group Brier: {nested_metrics['brier_score']:.8f}")
    print(f"Nested participant-group log loss: {nested_metrics['log_loss']:.8f}")
    print(f"Current B2 RemiCare P(STRABISMUS): {float(current_probability[1]):.12f}")
    print(f"Candidate RemiCare P(STRABISMUS): {float(remicare_probability[1]):.12f}")
    print("Production model modified: NO")


if __name__ == "__main__":
    main()
