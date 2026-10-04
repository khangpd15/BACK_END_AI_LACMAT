# Model Registry

This document is the current file-based registry until a database-backed `model_registry` table is added.

## Production/Runtime Models

| Model ID | Artifact | Status | Feature Version | Input Dimension | Threshold | Notes |
|---|---|---|---|---:|---:|---|
| `cover-test-10-15fps` | `app/models/remicare_15fps_candidate.joblib` | active candidate | `shared-v1.0.0` subset | artifact-defined, default 14 | model-defined | Primary Cover Test consensus path when `ENABLE_AI_INFERENCE_ON_SAVE=true`. |
| `korean-transfer-b2` | `app/models/remicare_transfer_model.joblib` | research comparison | `shared-v1.0.0` | artifact-defined, expected 30 or subset | model-defined | Research transfer only, not clinical diagnosis. |
| `korean-shared-fallback` | `app/models/korean_shared_model.joblib` | fallback research | `shared-v1.0.0` | artifact-defined | model-defined | Used if transfer artifact missing. |
| `bilateral-roi-onnx` | `app/models/best_model.onnx` | active image ROI screening | image preprocessing v1 | image tensor `1x3x224x224` | `0.20` | Requires bilateral eye ROI, not full face. |

## Research Candidate Models (Non-Production, Exploratory)

These models are exploratory research candidates trained on research manifests. They are strictly NON-PRODUCTION, NON-CLINICAL, and NOT accessible from `/api/v1/strabismus/predict`.

| Model ID | Artifact | Status | Feature Version | Input Dimension | Metrics (Test Split) | Governance Notes |
|---|---|---|---|---:|---|---|
| `hirschberg-candidate-v0.1` | `app/models/research/hirschberg_candidate_v0.1.joblib` | research_candidate | `hirschberg-crop-v0.1` | 73 | BalAcc: 0.7597, MacroF1: 0.7532, Sens: 0.9100, Spec: 0.7600, ROC-AUC: 0.9124 | Trained on `hirschberg_folder_labels.jsonl` (696 crops). HistGradientBoosting pipeline. Exploratory research only. |

## Planned Research Models

These entries are placeholders for reproducibility planning only. They are not trained and are not production candidates.

| Model ID | Artifact | Status | Feature Version | Input Dimension | Threshold | Metrics |
|---|---|---|---|---:|---:|---|
| `hirschberg-cover-research-v0.1` | null | planned, not trained | `research-geometry-v0.1` | null | null | null |

Required future metadata:

- `datasetVersion`: `remicare-hirschberg-cover-dataset-v0.1`
- `preprocessingVersion`: null until locked
- `trainedAt`: null
- `seed`: null until approved run
- `sensitivity`, `specificity`, `PPV`, `NPV`, `ROC-AUC`, `PR-AUC`, `confusion_matrix`: null
- `MAE`, `RMSE`, `bias`, `Bland-Altman`: null

The backend chooses the production model. Any `modelVersion` sent by a client is ignored for model selection.

## Mandatory Runtime Checks

At model load or before inference:

- Artifact must include `model` or `pipeline`.
- Artifact `feature_names` must exist for tabular sklearn models.
- If model exposes `n_features_in_`, it must equal `len(feature_names)`.
- Extracted feature vector length must equal model expected input.
- Missing/non-finite features must stop inference.

## Registry Fields To Add In Database

Recommended future table: `model_registry`

```text
model_id
version
status
model_type
artifact_path
feature_version
input_dimension
threshold
preprocessing_version
trained_at
dataset_version
metrics_json
created_at
```

No production model was retrained or modified in this pass.
