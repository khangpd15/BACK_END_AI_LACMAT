# Model Registry

This document is the current file-based registry until a database-backed `model_registry` table is added.

## Production/Runtime Models

| Model ID | Artifact | Status | Feature Version | Input Dimension | Threshold | Notes |
|---|---|---|---|---:|---:|---|
| `cover-test-10-15fps` | `app/models/remicare_15fps_candidate.joblib` | active candidate | `shared-v1.0.0` subset | artifact-defined, default 14 | model-defined | Primary Cover Test consensus path when `ENABLE_AI_INFERENCE_ON_SAVE=true`. |
| `korean-transfer-b2` | `app/models/remicare_transfer_model.joblib` | research comparison | `shared-v1.0.0` | artifact-defined, expected 30 or subset | model-defined | Research transfer only, not clinical diagnosis. |
| `korean-shared-fallback` | `app/models/korean_shared_model.joblib` | fallback research | `shared-v1.0.0` | artifact-defined | model-defined | Used if transfer artifact missing. |
| `bilateral-roi-onnx` | `app/models/best_model.onnx` | active image ROI screening | image preprocessing v1 | image tensor `1x3x224x224` | `0.20` | Requires bilateral eye ROI, not full face. |

## Mandatory Runtime Checks

At model load or before inference:

- Artifact must include `model`.
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
