"""Train a Korean-source transfer candidate on timestamp-downsampled 15 FPS data.

The model is a research comparison model. It does not use the user's self-report
or RemiCare samples as labels, does not tune a threshold, and does not calibrate
probabilities without an independent labeled RemiCare validation set.
"""

from __future__ import annotations

import json
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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.korean_adapter import parse_korean_csv  # noqa: E402
from app.services.shared_feature_contract import ALL_SHARED_FEATURES, extract_shared_features_vector  # noqa: E402


TARGET_HZ = 15.0
IPD_SCALE_FACTOR = 0.33009 / 0.13190
PHASE_OFFSETS_SECONDS = [0.0, 1.0 / 60.0, 2.0 / 60.0, 3.0 / 60.0]
HORIZONTAL = [
    "meanDeltaX", "medianDeltaX", "stdDeltaX", "minDeltaX",
    "maxDeltaX", "rangeDeltaX", "meanAbsDeltaX",
]
VIEWPORT_POSITIONS = ["meanLeftX", "meanLeftY", "meanRightX", "meanRightY"]
FEATURE_CONFIGS = {
    "15FPS_B2_30": list(ALL_SHARED_FEATURES),
    "15FPS_NO_VIEWPORT_26": [name for name in ALL_SHARED_FEATURES if name not in VIEWPORT_POSITIONS],
    "15FPS_STABILITY_14": [
        "leftValidRatio", "rightValidRatio", "bothValidRatio",
        "meanDeltaY", "medianDeltaY", "stdDeltaY", "minDeltaY",
        "maxDeltaY", "rangeDeltaY", "meanAbsDeltaY",
        "stdLeftX", "stdLeftY", "stdRightX", "stdRightY",
    ],
}
REGULARIZATION_C = [0.03, 0.1, 0.3, 1.0, 3.0]
CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

OUTPUT_DIR = ROOT / "outputs" / "retraining" / "15fps_candidate"
FEATURE_CACHE = ROOT / "data" / "processed" / "korean_15fps_augmented_features.csv"
APP_MODEL_PATH = ROOT / "app" / "models" / "remicare_15fps_candidate.joblib"
CANDIDATE_MODEL_PATH = ROOT / "models" / "candidates" / "remicare_15fps_candidate.joblib"


def participant_id(sample_id: str) -> str:
    match = re.search(r"[\uac00-\ud7a3]+", sample_id)
    if not match:
        return sample_id
    name = match.group(0)
    for marker in ["어머니", "엄마"]:
        if marker in sample_id and not name.endswith(marker):
            name += marker
            break
    return name


def collect_tasks() -> list[tuple[Path, str]]:
    tasks: list[tuple[Path, str]] = []
    locations = [
        (ROOT / "data" / "korean_repo" / "data" / "normal", "NORMAL"),
        (ROOT / "data" / "korean_repo" / "data" / "esotropia", "STRABISMUS"),
        (ROOT / "data" / "korean_repo" / "data" / "exotropia", "STRABISMUS"),
        (ROOT / "data" / "korean_repo" / "data" / "hypertropia", "STRABISMUS"),
        (ROOT / "data" / "test_normal", "NORMAL"),
        (ROOT / "data" / "test_strabismus", "STRABISMUS"),
    ]
    for directory, label in locations:
        tasks.extend((path, label) for path in sorted(directory.rglob("*.csv")))
    return tasks


def timestamp_downsample(
    samples: list[dict[str, Any]],
    target_hz: float,
    phase_offset_seconds: float,
) -> list[dict[str, Any]]:
    """Choose the source sample nearest each target timestamp."""
    if len(samples) < 2:
        return list(samples)
    interval = 1.0 / target_hz
    start = float(samples[0]["t"]) + phase_offset_seconds
    end = float(samples[-1]["t"])
    target_times = np.arange(start, end + interval * 0.25, interval)
    source_times = np.asarray([float(sample["t"]) for sample in samples], dtype=float)
    selected_indices: list[int] = []
    for target in target_times:
        right = int(np.searchsorted(source_times, target, side="left"))
        choices = [index for index in (right - 1, right) if 0 <= index < len(samples)]
        if not choices:
            continue
        best = min(choices, key=lambda index: abs(source_times[index] - target))
        if not selected_indices or best != selected_indices[-1]:
            selected_indices.append(best)
    return [samples[index] for index in selected_indices]


def build_feature_cache() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path, label in collect_tasks():
        record, parse_errors = parse_korean_csv(str(path), label=label)
        if record is None or parse_errors:
            errors.extend(parse_errors)
            continue
        for phase_index, offset in enumerate(PHASE_OFFSETS_SECONDS):
            downsampled = timestamp_downsample(record.samples, TARGET_HZ, offset)
            features = extract_shared_features_vector(downsampled)
            if any(features[name] is None for name in ALL_SHARED_FEATURES):
                errors.append(f"Missing shared feature after downsampling: {path} offset={offset}")
                continue
            rows.append({
                "sampleId": record.sampleId,
                "participantId": participant_id(record.sampleId),
                "label": 1 if label == "STRABISMUS" else 0,
                "phaseOffsetIndex": phase_index,
                "phaseOffsetSeconds": offset,
                "sourceEstimatedHz": record.sampling["estimatedHz"],
                "downsampledCount": len(downsampled),
                **features,
            })
    frame = pd.DataFrame(rows)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURE_CACHE, index=False, encoding="utf-8")
    return frame, errors


def prepare_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    matrix = np.nan_to_num(frame[names].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    for horizontal_name in HORIZONTAL:
        if horizontal_name in names:
            matrix[:, names.index(horizontal_name)] /= IPD_SCALE_FACTOR
    return matrix


def make_model(c_value: float, class_weight: str | None) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight=class_weight, max_iter=3000, random_state=42),
    )


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, prediction)),
        "f1_at_0_5": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix_at_0_5": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def configs() -> list[dict[str, Any]]:
    return [
        {"feature_config": feature_config, "C": c_value, "class_weight": class_weight}
        for feature_config in FEATURE_CONFIGS
        for c_value in REGULARIZATION_C
        for class_weight in CLASS_WEIGHTS
    ]


def inner_brier(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> float:
    names = FEATURE_CONFIGS[config["feature_config"]]
    x = prepare_matrix(frame.iloc[indices], names)
    subset_y = y[indices]
    subset_groups = groups[indices]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    probability = np.full(len(indices), np.nan)
    for train_index, valid_index in splitter.split(x, subset_y, subset_groups):
        model = make_model(config["C"], config["class_weight"])
        model.fit(x[train_index], subset_y[train_index])
        probability[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Incomplete inner-CV predictions")
    return float(brier_score_loss(subset_y, probability))


def select_config(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranking = []
    for config in configs():
        ranking.append({**config, "inner_brier": inner_brier(frame, y, groups, indices, config, seed)})
    ranking.sort(key=lambda row: (row["inner_brier"], row["feature_config"], row["C"], str(row["class_weight"])))
    return ranking[0], ranking


def native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame, extraction_errors = build_feature_cache()
    mixed = frame.groupby("participantId")["label"].nunique()
    conflicting_participants = mixed[mixed > 1].index.tolist()
    excluded_rows = frame[frame["participantId"].isin(conflicting_participants)][
        ["sampleId", "participantId", "label"]
    ].drop_duplicates().to_dict("records")
    frame = frame[~frame["participantId"].isin(conflicting_participants)].reset_index(drop=True)

    y = frame["label"].to_numpy(dtype=int)
    groups = frame["participantId"].to_numpy(dtype=str)
    placeholder = np.zeros((len(frame), 1))
    all_indices = np.arange(len(frame))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1515)
    oof_probability = np.full(len(frame), np.nan)
    fold_reports = []
    for fold, (train_index, valid_index) in enumerate(outer.split(placeholder, y, groups), start=1):
        selected, _ = select_config(frame, y, groups, train_index, 1500 + fold)
        names = FEATURE_CONFIGS[selected["feature_config"]]
        train_x = prepare_matrix(frame.iloc[train_index], names)
        valid_x = prepare_matrix(frame.iloc[valid_index], names)
        model = make_model(selected["C"], selected["class_weight"])
        model.fit(train_x, y[train_index])
        oof_probability[valid_index] = model.predict_proba(valid_x)[:, 1]
        fold_reports.append({
            "fold": fold,
            "selected_config": selected,
            "train_participants": len(np.unique(groups[train_index])),
            "validation_participants": len(np.unique(groups[valid_index])),
            "participant_overlap": sorted(set(groups[train_index]).intersection(groups[valid_index])),
        })
    if np.isnan(oof_probability).any():
        raise RuntimeError("Incomplete outer-CV predictions")

    legacy_names = FEATURE_CONFIGS["15FPS_B2_30"]
    legacy_x = prepare_matrix(frame, legacy_names)
    legacy_probability = np.full(len(frame), np.nan)
    legacy_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1515)
    for train_index, valid_index in legacy_outer.split(placeholder, y, groups):
        legacy_model = make_model(1.0, "balanced")
        legacy_model.fit(legacy_x[train_index], y[train_index])
        legacy_probability[valid_index] = legacy_model.predict_proba(legacy_x[valid_index])[:, 1]
    if np.isnan(legacy_probability).any():
        raise RuntimeError("Incomplete legacy B2 outer-CV predictions")

    selected, ranking = select_config(frame, y, groups, all_indices, 151515)
    names = FEATURE_CONFIGS[selected["feature_config"]]
    final_x = prepare_matrix(frame, names)
    final_model = make_model(selected["C"], selected["class_weight"])
    final_model.fit(final_x, y)

    artifact = {
        "model": final_model,
        "name": "Korean 15 FPS Logistic Regression candidate",
        "version": "remicare-transfer-15fps-candidate-v1.0.0",
        "status": "RESEARCH_COMPARISON_ONLY",
        "feature_names": names,
        "feature_count": len(names),
        "feature_config": selected["feature_config"],
        "expected_sampling_hz": TARGET_HZ,
        "source_sampling_hz": "approximately 60Hz",
        "downsampling_method": "nearest source sample to uniform 15Hz timestamps",
        "training_phase_offsets_seconds": PHASE_OFFSETS_SECONDS,
        "ipd_scale_factor": IPD_SCALE_FACTOR,
        "training_transform": "Korean horizontal disparity divided by ipd_scale_factor; RemiCare input as-is.",
        "selected_hyperparameters": {"C": selected["C"], "class_weight": selected["class_weight"]},
        "nested_participant_group_cv_metrics": metrics(y, oof_probability),
        "training_rows_augmented": len(frame),
        "unique_source_sessions": int(frame["sampleId"].nunique()),
        "training_participants": int(frame["participantId"].nunique()),
        "calibration_applied": False,
        "threshold_tuned": False,
        "remicare_labels_used": 0,
        "clinical_meaning": None,
        "notice": "Research-only 15 FPS transfer output. Not a diagnosis and not a validated clinical probability.",
    }
    joblib.dump(artifact, APP_MODEL_PATH)
    joblib.dump(artifact, CANDIDATE_MODEL_PATH)

    session_stability = []
    for sample_id, sample_rows in frame.assign(oofProbability=oof_probability).groupby("sampleId"):
        session_stability.append({
            "sampleId": sample_id,
            "participantId": sample_rows.iloc[0]["participantId"],
            "label": int(sample_rows.iloc[0]["label"]),
            "mean_oof_probability": float(sample_rows["oofProbability"].mean()),
            "std_across_phase_offsets": float(sample_rows["oofProbability"].std(ddof=0)),
        })

    results = {
        "target_hz": TARGET_HZ,
        "source_files_discovered": len(collect_tasks()),
        "augmented_rows_after_exclusion": len(frame),
        "unique_sessions_after_exclusion": int(frame["sampleId"].nunique()),
        "unique_participants_after_exclusion": int(frame["participantId"].nunique()),
        "conflicting_participants_excluded": conflicting_participants,
        "excluded_source_rows": excluded_rows,
        "extraction_error_count": len(extraction_errors),
        "extraction_errors": extraction_errors,
        "selected_config": selected,
        "nested_oof_metrics": metrics(y, oof_probability),
        "legacy_b2_configuration_same_15fps_folds": metrics(y, legacy_probability),
        "folds": fold_reports,
        "ranked_search": ranking,
        "mean_phase_offset_probability_std": float(np.mean([row["std_across_phase_offsets"] for row in session_stability])),
        "max_phase_offset_probability_std": float(np.max([row["std_across_phase_offsets"] for row in session_stability])),
        "calibration_applied": False,
        "threshold_tuned": False,
    }
    write_json(OUTPUT_DIR / "training_results.json", results)
    write_json(OUTPUT_DIR / "session_phase_stability.json", session_stability)

    report = f"""# Korean 15 FPS Transfer Candidate

## Training outcome

- Source files: `{len(collect_tasks())}` Korean recordings
- Target training rate: `{TARGET_HZ:.0f} FPS`
- Phase-offset views per recording: `{len(PHASE_OFFSETS_SECONDS)}`
- Augmented rows after conflict exclusion: `{len(frame)}`
- Unique sessions: `{frame['sampleId'].nunique()}`
- Unique participants: `{frame['participantId'].nunique()}`
- Selected feature configuration: `{selected['feature_config']}` ({len(names)} features)
- Logistic `C`: `{selected['C']}`
- Class weight: `{selected['class_weight']}`
- Participant overlap in every validation fold: `0`
- Calibration: `NONE`
- Threshold tuning: `NONE`
- RemiCare labels used: `0`

## Participant-group validation

- Brier score: `{metrics(y, oof_probability)['brier_score']:.8f}`
- Log loss: `{metrics(y, oof_probability)['log_loss']:.8f}`
- ROC AUC: `{metrics(y, oof_probability)['roc_auc']:.8f}`
- Balanced accuracy at 0.5: `{metrics(y, oof_probability)['balanced_accuracy_at_0_5']:.8f}`
- F1 at 0.5: `{metrics(y, oof_probability)['f1_at_0_5']:.8f}`
- Mean probability std across four 15 FPS phase offsets: `{results['mean_phase_offset_probability_std']:.8f}`

Legacy B2 configuration retrained/evaluated on the same 15 FPS participant folds:

- Brier score: `{metrics(y, legacy_probability)['brier_score']:.8f}`
- Log loss: `{metrics(y, legacy_probability)['log_loss']:.8f}`
- ROC AUC: `{metrics(y, legacy_probability)['roc_auc']:.8f}`
- Balanced accuracy at 0.5: `{metrics(y, legacy_probability)['balanced_accuracy_at_0_5']:.8f}`

The selected 15 FPS candidate changes Brier by `{metrics(y, oof_probability)['brier_score'] - metrics(y, legacy_probability)['brier_score']:+.8f}` (negative is better) on these source-domain folds.

## Safety conclusion

This candidate improves sampling-rate alignment and is suitable for side-by-side technical comparison. It cannot be trained to guarantee `NORMAL > 80%` for a known user, because doing so would use the desired answer as a target and invalidate the experiment. It remains a Korean-source model under webcam domain shift, not a clinical diagnostic model.
"""
    (OUTPUT_DIR / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")

    print(f"15 FPS candidate saved: {APP_MODEL_PATH}")
    print(f"Selected: {selected}")
    print(f"Brier: {results['nested_oof_metrics']['brier_score']:.8f}")
    print(f"Log loss: {results['nested_oof_metrics']['log_loss']:.8f}")
    print(f"Extraction errors: {len(extraction_errors)}")
    print("Production Korean B2 replaced: NO")


if __name__ == "__main__":
    main()
