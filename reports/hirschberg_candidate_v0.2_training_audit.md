# Hirschberg Candidate v0.2 Training Audit

Date: 2026-10-04

Status: **blocked before training**. No `hirschberg_candidate_v0.2.joblib` artifact was produced.

## Scope

This audit reviewed the backend Hirschberg research training path without changing production inference:

- `app/services/research_measurement_service.py`
- `app/services/hirschberg_ai_service.py`
- `scripts/train_hirschberg_research_model.py`
- `scripts/hirschberg_folder_manifest.py`
- `scripts/benchmark_hirschberg_detectors.py`
- `tests/research/`

The production endpoint `/api/v1/strabismus/predict` and the production ONNX model were not changed.

## Existing Candidate

Existing artifact:

- `app/models/research/hirschberg_candidate_v0.1.joblib`

Bundle summary:

- `model_id`: `hirschberg-candidate-v0.1`
- `status`: `research_candidate`
- `model_type`: `HistGradientBoosting`
- `classes`: `esotropia`, `exotropia`, `normal`
- `feature_count`: 73
- `is_production`: `False`

Feature families:

- iris geometry
- corneal reflex detection metrics
- pupil detection metrics
- normalized Hirschberg displacement vectors
- dark centroid
- left/right intensity asymmetry
- horizontal intensity profile
- 4x4 spatial grid statistics
- frozen ONNX bilateral ROI logits

Existing v0.1 test metrics from `D:/AI_Check_Lac/manifests/hirschberg_candidate_eval.json`:

- test accuracy: 0.76
- test balanced accuracy: 0.7597
- test macro-F1: 0.7532
- binary strabismus-vs-normal sensitivity: 0.91
- binary strabismus-vs-normal specificity: 0.76
- binary ROC-AUC: 0.9124

These metrics remain research-only because the data are 224x224 crops with unverified participant identity.

## Dataset Findings

Local folder `harschberg_data_detect/`:

| Folder | Count |
|---|---:|
| `esotropia_harschberg` | 55 |
| `exotropia_harschberg` | 40 |
| `normal_harschberg` | 41 |
| `rejected` | 366 |
| `debug` | 20 |

This local accepted set has only 136 trainable images and is not enough for a trustworthy v0.2 candidate.

Existing manifest:

- `D:/AI_Check_Lac/manifests/hirschberg_folder_labels.jsonl`
- records: 696
- class counts: esotropia 246, exotropia 270, normal 180
- generated split counts: train 467, validation 104, test 125
- domain: `legacy_or_unknown_hirschberg_crop`
- training use: `exploratory_only_until_participant_id_and_protocol_are_confirmed`

Blocking issue: manifest image paths point to `data_hirschberg/eye-classification/...`, but that dataset root is not present in the current backend or frontend workspace. A v0.2 training run with `--force-recompute` resolved zero trainable images, so training was stopped.

## Script Update

`scripts/train_hirschberg_research_model.py` was updated to support a v0.2 research run without overwriting v0.1:

- default model id: `hirschberg-candidate-v0.2`
- default artifact: `app/models/research/hirschberg_candidate_v0.2.joblib`
- default eval report: `reports/hirschberg_candidate_v0.2_eval.json`
- default model card: `reports/hirschberg_candidate_v0.2_model_card.md`
- records false negatives and false positives for review
- writes known limitations into the eval report
- fails early when train/val/test splits are empty after image path resolution

No v0.2 model artifact was saved because the resolved dataset was not trainable.

## Decision

Do not train or deploy v0.2 from the currently available files.

Reasons:

1. The 696-record manifest references missing image files.
2. The local 136-image accepted folder is too small and crop-only.
3. Participant IDs are not available, so patient-level leakage cannot be ruled out.
4. The data are 224x224 crops, not the full-face phone-flash uploads used by the current product flow.
5. There is no separate clinician label manifest in this repository.

## Next Required Inputs

Before training v0.2:

1. Restore or provide the dataset root referenced by the manifest:
   `data_hirschberg/eye-classification/...`
2. Or rebuild a fresh manifest from an available dataset root.
3. Add participant/group IDs if possible.
4. Add a clinician-reviewed label manifest if available.
5. Prefer collecting representative full-face phone-flash Hirschberg uploads from the actual product flow.

## Recommended Next Run

After restoring the image dataset:

```powershell
.\.venv312\Scripts\python.exe scripts\train_hirschberg_research_model.py `
  --force-recompute `
  --model-id hirschberg-candidate-v0.2 `
  --model-output app\models\research\hirschberg_candidate_v0.2.joblib `
  --eval-output reports\hirschberg_candidate_v0.2_eval.json `
  --model-card-output reports\hirschberg_candidate_v0.2_model_card.md `
  --cache-features reports\hirschberg_candidate_v0.2_features.npz
```

The resulting model must remain `research_candidate` and must not be promoted to production without clinician/data governance review.
