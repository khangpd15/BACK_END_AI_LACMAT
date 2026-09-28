"""Train a Korean-source candidate robust to 10--15 FPS webcam sampling.

This is a research comparison model. It uses only labelled Korean source
recordings. Unlabelled RemiCare sessions may be used for inference diagnostics,
but never as training labels or for threshold/probability calibration.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_15fps_transfer_candidate as base  # noqa: E402
from app.services.korean_adapter import parse_korean_csv  # noqa: E402
from app.services.shared_feature_contract import ALL_SHARED_FEATURES, extract_shared_features_vector  # noqa: E402


TARGET_RATES_HZ = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FIXED_PHASE_FRACTIONS = [0.0, 0.25, 0.5, 0.75]
IRREGULAR_PROFILE_COUNT = 3
REGULARIZATION_C = [0.01, 0.03, 0.1, 0.3, 1.0]
CLASS_WEIGHTS: list[Any] = [
    None,
    {0: 1.15, 1: 1.0},
    {0: 1.25, 1: 1.0},
    {0: 1.4, 1: 1.0},
    "balanced",
]
OUTPUT_DIR = ROOT / "outputs" / "retraining" / "10_15fps_candidate"
FEATURE_CACHE = ROOT / "data" / "processed" / "korean_10_15fps_augmented_features.csv"
APP_MODEL_PATH = ROOT / "app" / "models" / "remicare_15fps_candidate.joblib"
CANDIDATE_MODEL_PATH = ROOT / "models" / "candidates" / "remicare_15fps_candidate.joblib"


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def irregular_downsample(
    samples: list[dict[str, Any]],
    seed: int,
    min_hz: float = 10.0,
    max_hz: float = 15.0,
) -> list[dict[str, Any]]:
    """Downsample with deterministic frame-to-frame rate jitter and dropped frames."""
    if len(samples) < 2:
        return list(samples)
    rng = np.random.default_rng(seed)
    source_times = np.asarray([float(sample["t"]) for sample in samples], dtype=float)
    target = float(source_times[0]) + float(rng.uniform(0.0, 1.0 / min_hz))
    end = float(source_times[-1])
    selected_indices: list[int] = []
    while target <= end:
        right = int(np.searchsorted(source_times, target, side="left"))
        choices = [index for index in (right - 1, right) if 0 <= index < len(samples)]
        if choices:
            best = min(choices, key=lambda index: abs(source_times[index] - target))
            if not selected_indices or best != selected_indices[-1]:
                selected_indices.append(best)
        instantaneous_hz = float(rng.uniform(min_hz, max_hz))
        target += 1.0 / instantaneous_hz
        # Simulate an occasional dropped webcam frame without changing labels.
        if rng.random() < 0.08:
            target += 1.0 / instantaneous_hz
    return [samples[index] for index in selected_indices]


def sampling_views(samples: list[dict[str, Any]], sample_id: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for rate_hz in TARGET_RATES_HZ:
        for phase_fraction in FIXED_PHASE_FRACTIONS:
            offset_seconds = phase_fraction / rate_hz
            views.append({
                "profile": f"fixed_{rate_hz:g}hz_phase_{phase_fraction:g}",
                "nominalHz": rate_hz,
                "samples": base.timestamp_downsample(samples, rate_hz, offset_seconds),
            })
    for profile_index in range(IRREGULAR_PROFILE_COUNT):
        seed = stable_seed(f"{sample_id}:{profile_index}")
        views.append({
            "profile": f"variable_10_15hz_drop_{profile_index}",
            "nominalHz": None,
            "samples": irregular_downsample(samples, seed),
        })
    return views


def build_feature_cache() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path, label in base.collect_tasks():
        record, parse_errors = parse_korean_csv(str(path), label=label)
        if record is None or parse_errors:
            errors.extend(parse_errors)
            continue
        for view in sampling_views(record.samples, record.sampleId):
            features = extract_shared_features_vector(view["samples"])
            if any(features[name] is None for name in ALL_SHARED_FEATURES):
                errors.append(f"Missing shared feature: {path} profile={view['profile']}")
                continue
            rows.append({
                "sampleId": record.sampleId,
                "participantId": base.participant_id(record.sampleId),
                "label": 1 if label == "STRABISMUS" else 0,
                "samplingProfile": view["profile"],
                "nominalHz": view["nominalHz"],
                "sourceEstimatedHz": record.sampling["estimatedHz"],
                "downsampledCount": len(view["samples"]),
                **features,
            })
    frame = pd.DataFrame(rows)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURE_CACHE, index=False, encoding="utf-8")
    return frame, errors


def profile_probability_std(frame: pd.DataFrame, probability: np.ndarray) -> float:
    scored = frame[["sampleId"]].copy()
    scored["probability"] = probability
    return float(scored.groupby("sampleId")["probability"].std(ddof=0).fillna(0.0).mean())


def inner_score(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    names = base.FEATURE_CONFIGS[config["feature_config"]]
    x = base.prepare_matrix(frame.iloc[indices], names)
    subset_y = y[indices]
    subset_groups = groups[indices]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    probability = np.full(len(indices), np.nan)
    for train_index, valid_index in splitter.split(x, subset_y, subset_groups):
        model = base.make_model(config["C"], config["class_weight"])
        model.fit(x[train_index], subset_y[train_index])
        probability[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Incomplete inner-CV predictions")
    prediction = (probability >= 0.5).astype(int)
    brier = float(brier_score_loss(subset_y, probability))
    balanced_accuracy = float(balanced_accuracy_score(subset_y, prediction))
    stability = profile_probability_std(frame.iloc[indices], probability)
    # Korean source labels are imbalanced, so raw accuracy/Brier alone can favor
    # the majority class. Use a pre-declared multi-objective score that gives
    # balanced accuracy meaningful weight while penalizing rate instability.
    # The decision threshold itself remains fixed at 0.5.
    objective = brier + 0.25 * (1.0 - balanced_accuracy) + 0.50 * stability
    return {
        "inner_objective": objective,
        "inner_brier": brier,
        "inner_balanced_accuracy_at_0_5": balanced_accuracy,
        "inner_profile_probability_std": stability,
    }


def select_config(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []
    candidate_configs = [
        {"feature_config": feature_config, "C": c_value, "class_weight": class_weight}
        for feature_config in ["15FPS_STABILITY_14", "15FPS_NO_VIEWPORT_26"]
        for c_value in REGULARIZATION_C
        for class_weight in CLASS_WEIGHTS
    ]
    for config in candidate_configs:
        ranking.append({**config, **inner_score(frame, y, groups, indices, config, seed)})
    ranking.sort(key=lambda row: (row["inner_objective"], row["inner_brier"]))
    return ranking[0], ranking


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
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1015)
    oof_probability = np.full(len(frame), np.nan)
    baseline_probability = np.full(len(frame), np.nan)
    fold_reports: list[dict[str, Any]] = []

    for fold, (train_index, valid_index) in enumerate(outer.split(placeholder, y, groups), start=1):
        selected, _ = select_config(frame, y, groups, train_index, 1000 + fold)
        names = base.FEATURE_CONFIGS[selected["feature_config"]]
        train_x = base.prepare_matrix(frame.iloc[train_index], names)
        valid_x = base.prepare_matrix(frame.iloc[valid_index], names)
        model = base.make_model(selected["C"], selected["class_weight"])
        model.fit(train_x, y[train_index])
        oof_probability[valid_index] = model.predict_proba(valid_x)[:, 1]

        baseline_names = base.FEATURE_CONFIGS["15FPS_STABILITY_14"]
        # Reproduce the previous candidate's training exposure: fixed 15 FPS
        # phase views only. Validation still covers every 10--15 FPS profile.
        baseline_train_mask = frame.iloc[train_index]["samplingProfile"].str.startswith("fixed_15hz_").to_numpy()
        baseline_train_index = train_index[baseline_train_mask]
        baseline_train_x = base.prepare_matrix(frame.iloc[baseline_train_index], baseline_names)
        baseline_valid_x = base.prepare_matrix(frame.iloc[valid_index], baseline_names)
        baseline_model = base.make_model(0.03, None)
        baseline_model.fit(baseline_train_x, y[baseline_train_index])
        baseline_probability[valid_index] = baseline_model.predict_proba(baseline_valid_x)[:, 1]

        fold_reports.append({
            "fold": fold,
            "selected_config": selected,
            "train_participants": len(np.unique(groups[train_index])),
            "validation_participants": len(np.unique(groups[valid_index])),
            "participant_overlap": sorted(set(groups[train_index]).intersection(groups[valid_index])),
        })

    if np.isnan(oof_probability).any() or np.isnan(baseline_probability).any():
        raise RuntimeError("Incomplete outer-CV predictions")

    selected, ranking = select_config(frame, y, groups, all_indices, 101515)
    names = base.FEATURE_CONFIGS[selected["feature_config"]]
    final_x = base.prepare_matrix(frame, names)
    final_model = base.make_model(selected["C"], selected["class_weight"])
    final_model.fit(final_x, y)

    candidate_metrics = base.metrics(y, oof_probability)
    baseline_metrics = base.metrics(y, baseline_probability)
    candidate_stability = profile_probability_std(frame, oof_probability)
    baseline_stability = profile_probability_std(frame, baseline_probability)

    artifact = {
        "model": final_model,
        "name": "Korean 10-15 FPS robust transfer candidate",
        "version": "remicare-transfer-10to15fps-candidate-v1.1.0",
        "status": "RESEARCH_COMPARISON_ONLY",
        "feature_names": names,
        "feature_count": len(names),
        "feature_config": selected["feature_config"],
        "expected_sampling_hz_min": min(TARGET_RATES_HZ),
        "expected_sampling_hz_max": max(TARGET_RATES_HZ),
        "source_sampling_hz": "approximately 60Hz",
        "downsampling_method": "fixed 10-15Hz plus deterministic irregular 10-15Hz/drop-frame augmentation",
        "training_sampling_profiles": sorted(frame["samplingProfile"].unique().tolist()),
        "ipd_scale_factor": base.IPD_SCALE_FACTOR,
        "training_transform": "Korean horizontal disparity divided by ipd_scale_factor; RemiCare input as-is.",
        "selected_hyperparameters": {"C": selected["C"], "class_weight": selected["class_weight"]},
        "nested_participant_group_cv_metrics": candidate_metrics,
        "mean_profile_probability_std": candidate_stability,
        "training_rows_augmented": len(frame),
        "unique_source_sessions": int(frame["sampleId"].nunique()),
        "training_participants": int(frame["participantId"].nunique()),
        "calibration_applied": False,
        "threshold_tuned": False,
        "remicare_labels_used": 0,
        "clinical_meaning": None,
        "ui_label": "Korean 10-15 FPS candidate",
        "sampling_profile": "Korean recordings augmented across fixed and variable 10-15 FPS with simulated frame drops",
        "notice": "Research-only 10-15 FPS transfer output. Not a diagnosis or validated clinical probability.",
    }
    joblib.dump(artifact, APP_MODEL_PATH)
    joblib.dump(artifact, CANDIDATE_MODEL_PATH)

    results = {
        "target_hz_range": [min(TARGET_RATES_HZ), max(TARGET_RATES_HZ)],
        "fixed_rates_hz": TARGET_RATES_HZ,
        "fixed_phase_fractions": FIXED_PHASE_FRACTIONS,
        "irregular_profile_count": IRREGULAR_PROFILE_COUNT,
        "source_files_discovered": len(base.collect_tasks()),
        "augmented_rows_after_exclusion": len(frame),
        "unique_sessions_after_exclusion": int(frame["sampleId"].nunique()),
        "unique_participants_after_exclusion": int(frame["participantId"].nunique()),
        "conflicting_participants_excluded": conflicting_participants,
        "excluded_source_rows": excluded_rows,
        "extraction_error_count": len(extraction_errors),
        "extraction_errors": extraction_errors,
        "selected_config": selected,
        "nested_oof_metrics": candidate_metrics,
        "mean_profile_probability_std": candidate_stability,
        "previous_15fps_configuration_same_multirate_folds": baseline_metrics,
        "previous_15fps_mean_profile_probability_std": baseline_stability,
        "folds": fold_reports,
        "ranked_search": ranking,
        "calibration_applied": False,
        "threshold_tuned": False,
        "remicare_labels_used": 0,
    }
    base.write_json(OUTPUT_DIR / "training_results.json", results)

    report = f"""# Korean 10--15 FPS Robust Transfer Candidate

## Training outcome

- Source files: `{len(base.collect_tasks())}` Korean recordings
- Sampling augmentation: fixed `{TARGET_RATES_HZ}` FPS (four phases each) plus `{IRREGULAR_PROFILE_COUNT}` variable/drop-frame profiles
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

## Participant-group validation on identical 10--15 FPS views

| Metric | New 10--15 FPS candidate | Previous 15 FPS configuration | Delta |
|---|---:|---:|---:|
| Brier score (lower better) | {candidate_metrics['brier_score']:.8f} | {baseline_metrics['brier_score']:.8f} | {candidate_metrics['brier_score'] - baseline_metrics['brier_score']:+.8f} |
| Log loss (lower better) | {candidate_metrics['log_loss']:.8f} | {baseline_metrics['log_loss']:.8f} | {candidate_metrics['log_loss'] - baseline_metrics['log_loss']:+.8f} |
| ROC AUC | {candidate_metrics['roc_auc']:.8f} | {baseline_metrics['roc_auc']:.8f} | {candidate_metrics['roc_auc'] - baseline_metrics['roc_auc']:+.8f} |
| Balanced accuracy at 0.5 | {candidate_metrics['balanced_accuracy_at_0_5']:.8f} | {baseline_metrics['balanced_accuracy_at_0_5']:.8f} | {candidate_metrics['balanced_accuracy_at_0_5'] - baseline_metrics['balanced_accuracy_at_0_5']:+.8f} |
| Mean probability std across rate profiles (lower better) | {candidate_stability:.8f} | {baseline_stability:.8f} | {candidate_stability - baseline_stability:+.8f} |

## Safety conclusion

The model is selected only from participant-separated Korean validation. The saved RemiCare session is unlabelled and was not used as a training target. This remains a research transfer output under Korean eye-tracker to webcam domain shift, not a diagnosis or clinically validated probability.
"""
    (OUTPUT_DIR / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")

    print(f"10-15 FPS candidate saved: {APP_MODEL_PATH}")
    print(f"Selected: {selected}")
    print(f"Candidate Brier: {candidate_metrics['brier_score']:.8f}")
    print(f"Baseline Brier: {baseline_metrics['brier_score']:.8f}")
    print(f"Candidate balanced accuracy: {candidate_metrics['balanced_accuracy_at_0_5']:.8f}")
    print(f"Baseline balanced accuracy: {baseline_metrics['balanced_accuracy_at_0_5']:.8f}")
    print(f"Candidate profile std: {candidate_stability:.8f}")
    print(f"Baseline profile std: {baseline_stability:.8f}")
    print(f"Extraction errors: {len(extraction_errors)}")
    print("Production Korean B2 replaced: NO")


if __name__ == "__main__":
    main()
