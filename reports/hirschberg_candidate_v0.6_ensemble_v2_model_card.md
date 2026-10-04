# Hirschberg Candidate v0.6 Ensemble v2 — Model Card

**Model ID:** `hirschberg-candidate-v0.6-ensemble-v2`
**Status:** `research_candidate`
**Created:** `2026-10-04T08:19:17.233736+00:00`
**Target:** Real-World Eye-Crop Hirschberg Analysis (No Full-Face Requirement)

---

## 1. Benchmark: v0.5 vs v0.2 Eye-Crop vs v0.6 v1 vs v0.6 v2 on `harschberg_data_detect/`

| Model | Bal Acc | Macro F1 | Coverage | Eso Sens | Exo Sens | Normal Spec | Brier |
|---|---|---|---|---|---|---|---|
| **v0.5 forced (100%)** | 0.3610 | 0.3267 | 1.0000 | 0.7636 | 0.1000 | 0.2195 | 0.6568 |
| **v0.5 abstention** | N/A | N/A | 0.0000 | N/A | N/A | N/A | N/A |
| **v0.2 Eye-Crop (New)** | 0.8151 | 0.8200 | 1.0000 | 0.9636 | 0.7500 | 0.7317 | 0.4168 |
| **v0.6 Ensemble v1** | 0.3576 | 0.3007 | 0.3309 | 0.7727 | 0.0000 | 0.3000 | 0.6652 |
| **v0.6 Ensemble v2 (Eye-Crop)** | **0.9461** | **0.9484** | **0.7721** | **0.9787** | **0.8929** | **0.9667** | **0.4203** |

> Dataset: 136 images ({'esotropia': 55, 'exotropia': 40, 'normal': 41})
> Key Improvement: **Exotropia Sensitivity upgraded from 0.0% (v1) to 0.8929 (v2)** without full-face dependency.

---

## 2. Architecture & Eye-Crop Geometry Contract

1. **Input:** Eye crop pair (e.g. 224x224 BGR image containing both OD and OS).
2. **Dedicated Eye-Crop Geometry Extractor (`eye_crop_geometry_service.py`):**
   - Iris / pupil localization via adaptive dark component segmentation + Hough circle.
   - Corneal light reflex detection via specular highlight peak & connected components.
   - Corneal displacement vectors: `dx_left_mm`, `dy_left_mm`, `dx_right_mm`, `dy_right_mm`.
   - Palpebral fissure corners (inner/nasal, outer/temporal) and `scleral_ratio`.
   - Intercanthal distance, bilateral symmetry deviation, head roll angle.
   - Complete 19-dimensional geometric feature vector extracted without MediaPipe face mesh.
3. **v0.2 Eye-Crop Branch (`hirschberg_branch_v02_eyecrop.joblib`):**
   - 19 Geometry features + 16 EfficientNet-B0 embeddings = 35 features.
   - CalibratedClassifierCV on RandomForestClassifier with class weighting tuned for Exotropia.
4. **v0.6 Ensemble Fusion Policy:**
   - Prior-normalized weighted fusion: `alpha = 0.25` for v0.5, `(1 - alpha) = 0.75` for v0.2.
   - Confidence threshold: `0.4`, Margin threshold: `0.03`.
   - Abstention: returns `UNCERTAIN` for low confidence / ambiguous samples.

---

## 3. Governance & Safety

- Models `v0.5` and `v0.2` preserved (NOT overwritten).
- Prior version `v0.6_ensemble` preserved.
- Tuning performed strictly on the validation split.
- Research candidate — NOT for standalone clinical diagnosis without clinician confirmation.
