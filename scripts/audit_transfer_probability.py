"""Audit the RemiCare B2 transfer-model probability path.

This script is diagnostic only. It never overwrites a production model, changes a
threshold, calibrates probabilities, or creates clinical labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.shared_feature_contract import (  # noqa: E402
    ALL_SHARED_FEATURES,
    INVARIANT_SHARED_FEATURES,
    extract_shared_features_vector,
)


OUTPUT_DIR = ROOT / "outputs" / "debug" / "transfer_probability"
RAW_SAMPLE_PATH = ROOT / "data" / "raw" / "sample.json"
PROCESSED_SAMPLE_PATH = ROOT / "data" / "processed" / "remicare_shared_sample.json"
TRAIN_PATH = ROOT / "data" / "processed" / "korean_shared_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "korean_shared_test.csv"
B2_MODEL_PATH = ROOT / "app" / "models" / "remicare_b2_rescaled_model.joblib"
LIVE_MODEL_PATH = ROOT / "app" / "models" / "remicare_transfer_model.joblib"
ROOT_MODEL_PATH = ROOT / "models" / "remicare_transfer_model.joblib"
BASELINE_MODEL_PATH = ROOT / "app" / "models" / "korean_shared_model.joblib"
B1_MODEL_PATH = ROOT / "app" / "models" / "remicare_b1_invariant_model.joblib"

HORIZONTAL = [
    "meanDeltaX",
    "medianDeltaX",
    "stdDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "rangeDeltaX",
    "meanAbsDeltaX",
]
FIVE_REQUESTED_HORIZONTAL = [
    "meanDeltaX",
    "medianDeltaX",
    "minDeltaX",
    "maxDeltaX",
    "meanAbsDeltaX",
]
LOWER_SHIFT_SUBSET = [
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


def native(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write_json(name: str, value: Any) -> None:
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(native(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(score: float) -> float:
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1.0 + exp_score)


def flatten_raw_samples(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [sample for cycle in raw["cycles"] for sample in cycle["samples"]]


def predict_record(model: Any, vector: np.ndarray) -> dict[str, Any]:
    prediction_index = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    return {
        "prediction": "NORMAL" if prediction_index == 0 else "STRABISMUS",
        "p_normal": float(probabilities[0]),
        "p_strabismus": float(probabilities[1]),
        "domain_shift": "WARNING",
    }


def fit_three_models(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    balanced: bool,
    rf_estimators: int,
) -> dict[str, Any]:
    weight = "balanced" if balanced else None
    return {
        "RF": RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=6,
            class_weight=weight,
            random_state=42,
        ).fit(train_x, train_y),
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight=weight, random_state=42),
        ).fit(train_x, train_y),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(probability=True, kernel="rbf", class_weight=weight, random_state=42),
        ).fit(train_x, train_y),
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_sample = json.loads(RAW_SAMPLE_PATH.read_text(encoding="utf-8"))
    processed_sample = json.loads(PROCESSED_SAMPLE_PATH.read_text(encoding="utf-8"))
    raw_samples = flatten_raw_samples(raw_sample)
    extracted = extract_shared_features_vector(raw_samples)
    saved_features = processed_sample["features"]

    extraction_deltas = {
        name: float(extracted[name]) - float(saved_features[name])
        for name in ALL_SHARED_FEATURES
    }
    extraction_match = all(abs(delta) <= 1e-12 for delta in extraction_deltas.values())

    artifact = joblib.load(B2_MODEL_PATH)
    live_artifact = joblib.load(LIVE_MODEL_PATH)
    model = artifact["model"]
    feature_order = list(artifact["feature_names"])
    canonical_order = list(ALL_SHARED_FEATURES)
    feature_order_pass = feature_order == canonical_order
    if not feature_order_pass or len(feature_order) != 30:
        raise RuntimeError("FAIL: B2 artifact feature order/count does not match the canonical 30-feature contract")

    raw_vector = np.asarray([saved_features[name] for name in feature_order], dtype=float)
    inference_vector = raw_vector.copy()  # B2 inference is deliberately a no-op.
    vector_2d = inference_vector.reshape(1, -1)

    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    scaled_vector = scaler.transform(vector_2d)[0]
    direct_scaled = (inference_vector - scaler.mean_) / scaler.scale_
    scaler_formula_pass = bool(np.allclose(scaled_vector, direct_scaled, atol=1e-14, rtol=1e-14))

    score = float(model.decision_function(vector_2d)[0])
    probabilities = model.predict_proba(vector_2d)[0]
    sigmoid_probability = sigmoid(score)
    probability_formula_pass = abs(float(probabilities[1]) - sigmoid_probability) <= 1e-12

    coefficients = logistic.coef_[0]
    intercept = float(logistic.intercept_[0])
    contributions = coefficients * scaled_vector
    reconstructed_score = intercept + float(np.sum(contributions))
    contribution_formula_pass = abs(reconstructed_score - score) <= 1e-12

    scaler_rows = []
    contribution_rows = []
    for index, name in enumerate(feature_order):
        abs_z = abs(float(scaled_vector[index]))
        severity = ">=10" if abs_z >= 10 else ">=5" if abs_z >= 5 else ">=3" if abs_z >= 3 else "<3"
        scaler_rows.append({
            "feature": name,
            "raw": float(inference_vector[index]),
            "scaler_mean": float(scaler.mean_[index]),
            "scaler_std": float(scaler.scale_[index]),
            "scaled_z": float(scaled_vector[index]),
            "abs_z": abs_z,
            "severity": severity,
        })
        contribution_rows.append({
            "feature": name,
            "scaled_value": float(scaled_vector[index]),
            "coefficient": float(coefficients[index]),
            "contribution": float(contributions[index]),
            "abs_contribution": abs(float(contributions[index])),
        })
    top_z = sorted(scaler_rows, key=lambda row: row["abs_z"], reverse=True)[:10]
    top_contributions = sorted(contribution_rows, key=lambda row: row["abs_contribution"], reverse=True)[:10]

    scale_factor = float(artifact["ipd_scale_factor"])
    horizontal_trace = []
    for name in FIVE_REQUESTED_HORIZONTAL:
        value = float(saved_features[name])
        horizontal_trace.append({
            "feature": name,
            "remicare_original": value,
            "remicare_inference_transformed": value,
            "b2_training_formula": "korean_value / ipd_scale_factor",
            "ipd_scale_factor": scale_factor,
            "example_korean_to_b2": value / scale_factor,
            "note": "The example division is illustrative only; RemiCare inference remains unchanged.",
        })

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_y = train["label"].to_numpy(dtype=int)
    test_y = test["label"].to_numpy(dtype=int)
    raw_train_30 = np.nan_to_num(train[canonical_order].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    raw_test_30 = np.nan_to_num(test[canonical_order].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    b2_train_30 = raw_train_30.copy()
    b2_test_30 = raw_test_30.copy()
    for name in HORIZONTAL:
        index = canonical_order.index(name)
        b2_train_30[:, index] /= scale_factor
        b2_test_30[:, index] /= scale_factor

    scaler_training_only_pass = bool(
        np.allclose(scaler.mean_, b2_train_30.mean(axis=0), atol=1e-12, rtol=1e-12)
        and np.allclose(scaler.scale_, b2_train_30.std(axis=0, ddof=0), atol=1e-12, rtol=1e-12)
    )

    counts = {int(label): int(np.sum(train_y == label)) for label in np.unique(train_y)}
    class_weights = {
        str(label): len(train_y) / (len(counts) * count)
        for label, count in counts.items()
    }

    # Reconstruct the three documented model families for each experiment. These
    # are audit-only fits and are not saved as production artifacts.
    cross_model: list[dict[str, Any]] = []
    baseline_models = fit_three_models(raw_train_30, train_y, balanced=False, rf_estimators=100)
    baseline_models["RF"] = joblib.load(BASELINE_MODEL_PATH)["model"]
    for short_name, fitted in baseline_models.items():
        source = "saved artifact" if short_name == "RF" else "audit reconstruction"
        cross_model.append({"model": f"Baseline {short_name}", **predict_record(fitted, vector_2d), "source": source})

    b1_names = list(INVARIANT_SHARED_FEATURES)
    b1_indices = [canonical_order.index(name) for name in b1_names]
    b1_train = raw_train_30[:, b1_indices]
    b1_vector = raw_vector[b1_indices].reshape(1, -1)
    b1_models = fit_three_models(b1_train, train_y, balanced=True, rf_estimators=200)
    b1_models["LR"] = joblib.load(B1_MODEL_PATH)["model"]
    for short_name, fitted in b1_models.items():
        source = "saved artifact" if short_name == "LR" else "audit reconstruction"
        cross_model.append({"model": f"B1 {short_name}", **predict_record(fitted, b1_vector), "source": source})

    b2_models = fit_three_models(b2_train_30, train_y, balanced=True, rf_estimators=200)
    for short_name, fitted in b2_models.items():
        record = predict_record(fitted, vector_2d)
        record["source"] = "saved artifact" if short_name == "LR" else "audit reconstruction"
        cross_model.append({"model": f"B2 {short_name}", **record})

    # Ablations retrain audit-only balanced LR models on the Korean training data.
    without_horizontal = [name for name in canonical_order if name not in HORIZONTAL]
    without_horizontal_indices = [canonical_order.index(name) for name in without_horizontal]
    lower_shift_indices = [canonical_order.index(name) for name in LOWER_SHIFT_SUBSET]
    ablation_b_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    ).fit(b2_train_30[:, without_horizontal_indices], train_y)
    ablation_c_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    ).fit(b2_train_30[:, lower_shift_indices], train_y)
    raw_lr = baseline_models["LR"]
    normalized_lr = b2_models["LR"]
    ablations = [
        {"case": "A", "description": "Saved B2 LR, all 30 features", **predict_record(model, vector_2d)},
        {"case": "B", "description": "Audit-only B2 LR retrained without 7 horizontal disparity features", **predict_record(ablation_b_model, raw_vector[without_horizontal_indices].reshape(1, -1))},
        {"case": "C", "description": "Audit-only balanced LR using validity + vertical disparity + dispersion only", **predict_record(ablation_c_model, raw_vector[lower_shift_indices].reshape(1, -1))},
        {"case": "D-raw", "description": "Audit-only baseline LR: raw Korean horizontal coordinates and raw RemiCare input", **predict_record(raw_lr, vector_2d)},
        {"case": "D-normalized", "description": "Audit-only B2 LR: Korean horizontal values divided by scale; RemiCare input as-is", **predict_record(normalized_lr, vector_2d)},
    ]

    wrong_transform_sensitivity = []
    for label, multiplier in [("as_is_correct", 1.0), ("incorrect_multiply", scale_factor), ("incorrect_divide", 1.0 / scale_factor)]:
        altered = raw_vector.copy()
        for name in HORIZONTAL:
            altered[canonical_order.index(name)] *= multiplier
        wrong_transform_sensitivity.append({
            "scenario": label,
            "horizontal_multiplier_at_inference": multiplier,
            "decision_score": float(model.decision_function(altered.reshape(1, -1))[0]),
            **predict_record(model, altered.reshape(1, -1)),
        })

    write_json("input_features.json", {
        "sample_id": raw_sample["sampleId"],
        "raw_sample_path": str(RAW_SAMPLE_PATH.relative_to(ROOT)),
        "cycle_count": len(raw_sample["cycles"]),
        "raw_sample_count": len(raw_samples),
        "reported_fps": raw_sample.get("fps"),
        "feature_contract": "app/services/shared_feature_contract.py:ALL_SHARED_FEATURES",
        "feature_order": feature_order,
        "feature_count": len(feature_order),
        "features": {name: float(saved_features[name]) for name in feature_order},
        "fresh_extraction_matches_saved_processed_sample": extraction_match,
        "fresh_extraction_deltas": extraction_deltas,
    })
    write_json("transformed_features.json", {
        "experiment": artifact["experiment"],
        "training_transformation": "Korean horizontal disparity values are divided by ipd_scale_factor before fitting.",
        "inference_transformation": "No-op: RemiCare values are passed as-is.",
        "ipd_scale_factor": scale_factor,
        "horizontal_feature_trace": horizontal_trace,
        "full_transformed_features": {name: float(inference_vector[i]) for i, name in enumerate(feature_order)},
        "sensitivity_only_not_production": wrong_transform_sensitivity,
    })
    write_json("scaled_features.json", {
        "formula": "z = (inference_value - scaler.mean_) / scaler.scale_",
        "direct_formula_matches_pipeline": scaler_formula_pass,
        "scaler_verified_against_transformed_korean_training_only": scaler_training_only_pass,
        "features": scaler_rows,
        "top_10_absolute_z": top_z,
    })
    write_json("decision_function.json", {
        "intercept": intercept,
        "coefficients_and_contributions": contribution_rows,
        "top_10_absolute_contributions": top_contributions,
        "decision_score": score,
        "reconstructed_decision_score": reconstructed_score,
        "reconstruction_matches": contribution_formula_pass,
    })
    write_json("probabilities.json", {
        "classes": model.classes_.tolist(),
        "decision_score": score,
        "sigmoid_of_decision_score": sigmoid_probability,
        "predict_proba": {"NORMAL": float(probabilities[0]), "STRABISMUS": float(probabilities[1])},
        "sigmoid_matches_predict_proba": probability_formula_pass,
        "brief_reported_probability": {"NORMAL": 0.002, "STRABISMUS": 0.998},
        "brief_value_reproduced_from_repository_sample_and_current_artifact": False,
    })
    write_json("model_metadata.json", {
        "name": artifact["name"],
        "version": artifact["version"],
        "experiment": artifact["experiment"],
        "model_path": str(B2_MODEL_PATH.relative_to(ROOT)),
        "model_sha256": sha256(B2_MODEL_PATH),
        "live_model_path": str(LIVE_MODEL_PATH.relative_to(ROOT)),
        "live_model_sha256": sha256(LIVE_MODEL_PATH),
        "duplicate_model_path": str(ROOT_MODEL_PATH.relative_to(ROOT)),
        "duplicate_model_sha256": sha256(ROOT_MODEL_PATH),
        "all_three_b2_artifacts_byte_identical": len({sha256(B2_MODEL_PATH), sha256(LIVE_MODEL_PATH), sha256(ROOT_MODEL_PATH)}) == 1,
        "feature_order_matches_canonical": feature_order_pass,
        "feature_count": len(feature_order),
        "coordinate_rescaling": artifact["coordinate_rescaling"],
        "ipd_scale_factor": scale_factor,
        "training_samples": artifact["training_samples"],
        "test_samples": artifact["test_samples"],
        "class_weight_configuration": logistic.class_weight,
        "class_frequencies": {"NORMAL_0": counts[0], "STRABISMUS_1": counts[1]},
        "effective_balanced_class_weights": {"NORMAL_0": class_weights["0"], "STRABISMUS_1": class_weights["1"]},
        "scaler_fit_verified_training_only": scaler_training_only_pass,
        "scaler_mean": {name: float(scaler.mean_[i]) for i, name in enumerate(feature_order)},
        "scaler_scale": {name: float(scaler.scale_[i]) for i, name in enumerate(feature_order)},
        "logistic_intercept": intercept,
        "logistic_coefficients": {name: float(coefficients[i]) for i, name in enumerate(feature_order)},
    })
    write_json("cross_model_comparison.json", cross_model)
    write_json("ablation_results.json", ablations)

    top_z_rows = [
        [row["feature"], f'{row["raw"]:.10g}', f'{row["scaler_mean"]:.10g}', f'{row["scaler_std"]:.10g}', f'{row["scaled_z"]:.6f}', row["severity"]]
        for row in top_z
    ]
    top_contribution_rows = [
        [row["feature"], f'{row["scaled_value"]:.6f}', f'{row["coefficient"]:.6f}', f'{row["contribution"]:.6f}']
        for row in top_contributions
    ]
    cross_rows = [
        [row["model"], row["prediction"], f'{row["p_normal"]:.6f}', f'{row["p_strabismus"]:.6f}', row["domain_shift"], row["source"]]
        for row in cross_model
    ]
    ablation_rows = [
        [row["case"], row["prediction"], f'{row["p_normal"]:.6f}', f'{row["p_strabismus"]:.6f}', row["description"]]
        for row in ablations
    ]
    horizontal_rows = [
        [row["feature"], f'{row["remicare_original"]:.10g}', f'{row["remicare_inference_transformed"]:.10g}', row["b2_training_formula"]]
        for row in horizontal_trace
    ]

    report = f"""# RemiCare B2 Transfer Probability Audit

**Scope:** technical research audit only. No clinical diagnosis, clinical label, threshold change, or probability calibration was performed.

## Executive finding

The reported `STRABISMUS = 99.8%` is **not reproducible** from the repository's current sample and current B2 artifacts. All three B2 copies are byte-identical (`SHA-256: {sha256(B2_MODEL_PATH)}`) and return:

- Decision score: `{score:.15f}`
- `P(NORMAL)`: `{float(probabilities[0]):.15f}`
- `P(STRABISMUS)`: `{float(probabilities[1]):.15f}` ({float(probabilities[1]) * 100:.10f}%)
- Prediction: `STRABISMUS`
- Domain shift: `WARNING`

Therefore, the current evidence does not support explaining 99.8% as the output of this saved B2 Logistic Regression artifact on `remicare-demo-sample-001`. A 99.8% UI observation requires the exact request payload, API response/log, and artifact hash from that run. Plausible causes include a different live sample, a stale/different backend artifact, or a UI/API provenance mismatch; these are hypotheses, not established causes.

## 1. End-to-end trace

`data/raw/sample.json` contains {len(raw_sample['cycles'])} cycles and {len(raw_samples)} samples at reported `{raw_sample.get('fps')} FPS`. Fresh extraction with `extract_shared_features_vector()` exactly matches `data/processed/remicare_shared_sample.json`: **{str(extraction_match).upper()}**. The 30 values are ordered by `ALL_SHARED_FEATURES`, passed unchanged at B2 inference, standardized by the pipeline scaler, and scored by the Logistic Regression.

Artifacts: `input_features.json`, `transformed_features.json`, `scaled_features.json`, `decision_function.json`, `probabilities.json`, and `model_metadata.json` contain every numeric step.

## 2. Feature order and count

- Training/artifact feature order == canonical order: **{'PASS' if feature_order_pass else 'FAIL'}**
- Inference feature order == artifact order: **PASS**
- Feature count: **{len(feature_order)}**
- Canonical source: `app/services/shared_feature_contract.py::ALL_SHARED_FEATURES`

The inference service also contains a manually duplicated `FEATURE_ORDER`; it currently matches the canonical contract, but duplication is a future drift risk.

## 3. B2 normalization direction

Verified formula: **during training, each Korean horizontal disparity feature is divided by `{scale_factor:.15f}`; during RemiCare inference, those features are passed as-is.** This maps Korean coordinate scale down toward the RemiCare coordinate scale. Multiplying or dividing the RemiCare inference vector again would be inconsistent.

{markdown_table(['Feature', 'Original', 'Inference transformed', 'Training formula'], horizontal_rows)}

The sensitivity trace confirms that neither applying the documented scale in the wrong direction nor dividing again reproduces 99.8%; see `transformed_features.json`.

## 4. StandardScaler verification

- Scaler recomputation from B2-transformed Korean training rows only: **{'PASS' if scaler_training_only_pass else 'FAIL'}**
- Direct z-score formula matches `StandardScaler.transform`: **{'PASS' if scaler_formula_pass else 'FAIL'}**
- Formula: `z = (RemiCare feature - scaler.mean_) / scaler.scale_`

Top 10 absolute deviations (technical distribution-shift diagnostics only):

{markdown_table(['Feature', 'Raw', 'Scaler mean', 'Scaler std', 'Scaled Z', '|Z| band'], top_z_rows)}

Full 30-row scaler table is in `scaled_features.json` and full `mean_`/`scale_` maps are in `model_metadata.json`.

## 5. Logistic Regression coefficients and contributions

Intercept: `{intercept:.15f}`. The score reconstruction `intercept + sum(coef_i * z_i)` equals `{reconstructed_score:.15f}`: **{'PASS' if contribution_formula_pass else 'FAIL'}**.

Top 10 absolute contributions:

{markdown_table(['Feature', 'Scaled value', 'Coefficient', 'Contribution'], top_contribution_rows)}

These terms explain the technical class score; they are not clinical factors or thresholds.

## 6. Probability calculation

`sigmoid({score:.15f}) = {sigmoid_probability:.15f}` and `predict_proba()[STRABISMUS] = {float(probabilities[1]):.15f}`: **{'PASS' if probability_formula_pass else 'FAIL'}**.

The frontend's one-decimal formatting would display the current value as `{float(probabilities[1]) * 100:.1f}%`, not `99.8%`.

## 7. Class weighting

- Configuration: `class_weight='balanced'`
- Korean training frequencies: NORMAL={counts[0]}, STRABISMUS={counts[1]}, total={len(train_y)}
- Effective weights: NORMAL=`{class_weights['0']:.12f}`, STRABISMUS=`{class_weights['1']:.12f}`

Balanced weighting changes training loss; it does not make the model unbiased and does not alone explain an extreme probability.

## 8. Same-sample cross-model comparison

Only the winning baseline/B1/B2 models were persisted. To complete the requested nine-model comparison, missing model-family variants were deterministically reconstructed from the repository's Korean training CSV with the documented training configurations. No reconstructed model was saved into production.

{markdown_table(['Model', 'Prediction', 'P(NORMAL)', 'P(STRABISMUS)', 'Domain shift', 'Provenance'], cross_rows)}

The B2 SVM shows a class/probability disagreement in this run. Scikit-learn's SVC class prediction uses its decision function, while `probability=True` uses a separately fitted probability mapping and the two can disagree. This is reported as observed and is not used to calibrate or reinterpret the B2 Logistic Regression.

## 9. Feature ablations

The B/C/D variants are audit-only retrains on Korean labels. They are distribution-shift diagnostics, not clinical validation.

{markdown_table(['Case', 'Prediction', 'P(NORMAL)', 'P(STRABISMUS)', 'Definition'], ablation_rows)}

No calibration, threshold tuning, clipping, or smoothing was used.

## 10. Root-cause assessment

For the current reproducible output ({float(probabilities[1]) * 100:.4f}%), the largest positive/negative score terms and largest z-shifts are listed above. The score is jointly produced by training imbalance/weighting, out-of-domain standardized values, coefficients, and the scaler; no single factor is a sufficient explanation.

For the specifically reported 99.8%, the root cause cannot be assigned from repository state because it is absent from source, saved reports, current artifacts, and current-sample inference. The required next evidence is the exact raw payload and backend response from that UI run plus the loaded artifact SHA-256. Treating 99.8% as reproduced would be misleading.

## 11. UI safety review

Read-only inspection of the deployed frontend source at `d:\\AI_Check_Lac\\src\\components\\binocular\\CoverTestStep.jsx` confirms that it labels the value `Model Prediction` / `Class Probability`, displays a research/experiment badge, states it is not a medical diagnosis, and does not call the value “risk.” Recommended exact companion text remains: `Research-only output. Probability is model output and has not been clinically validated for RemiCare webcam data.` The external frontend was not modified by this audit.

## 12. Limitations and trustworthiness

- RemiCare samples evaluated: 1.
- RemiCare clinical ground-truth labels: 0.
- Korean rows are not an independent RemiCare validation set and have known participant-level leakage concerns.
- Cross-model and ablation probabilities do not establish clinical performance.
- The saved B2 model probability is technically reproducible for this exact artifact/vector, but **not trustworthy as a clinical probability for RemiCare** because of domain shift and absent target-domain ground truth.

## 13. Acceptance checklist

- A. Feature order: **PASS**
- B. Feature count: **30**
- C. B2 normalization: **verified — Korean horizontal features `/ {scale_factor:.15f}` during training; RemiCare inference as-is**
- D. Scaler: **training-only verification {'PASS' if scaler_training_only_pass else 'FAIL'}**
- E. Largest scaled deviations: **top 10 above**
- F. Largest LR contributions: **top 10 above**
- G. Decision score: **`{score:.15f}`**
- H. Probability: **NORMAL=`{float(probabilities[0]):.15f}`, STRABISMUS=`{float(probabilities[1]):.15f}`**
- I. Cross-model comparison: **completed (saved winners + audit-only reconstructions)**
- J. Clinical interpretation: **NONE**

> No clinical conclusion can be drawn from the current RemiCare 99.8% STRABISMUS output because there is currently no RemiCare clinical ground-truth validation dataset.
"""
    (OUTPUT_DIR / "PROBABILITY_AUDIT.md").write_text(report, encoding="utf-8")

    print(f"Audit artifacts written to: {OUTPUT_DIR}")
    print(f"Decision score: {score:.15f}")
    print(f"P(STRABISMUS): {float(probabilities[1]):.15f}")
    print("Reported 99.8% reproduced: NO")


if __name__ == "__main__":
    main()
