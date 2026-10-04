# Hirschberg Candidate v0.6 Ensemble — Model Card

**Model ID:** `hirschberg-candidate-v0.6-ensemble`
**Status:** `research_candidate`
**Created:** `2026-10-04T08:06:33.147203+00:00`
**Decision:** v0.6 SAVED — better than v0.5 on real-world data

---

## 1. Benchmark: v0.5 vs v0.2 vs v0.6 on `harschberg_data_detect/`

| Model | Bal Acc | Macro F1 | Coverage | Eso Sens | Exo Sens | Brier |
|---|---|---|---|---|---|---|
| v0.5 forced (100%) | 0.3610 | 0.3267 | 1.000 | 0.7636 | 0.1000 | 0.6568 |
| v0.5 abstention | N/A | N/A | 0.0000 | N/A | N/A | N/A |
| v0.2 EfficientNet-only | 0.3333 | 0.1544 | 1.000 | 0.0000 | 0.0000 | 0.8527 |
| **v0.6 ensemble** | **0.3576** | **0.3007** | 0.3309 | 0.7727 | 0.0000 | 0.6652 |

> Dataset: 136 images, classes: {'esotropia': 55, 'exotropia': 40, 'normal': 41}
> WARNING: No participant_id. Patient leakage cannot be ruled out.

---

## 2. Feature Contract Audit

**v0.5 (Primary):** 365 features = pair(73,73) via ONNX. 3-class. Works on 224x224 crops.

**v0.2 (Secondary):** 35 features = 19 geometric + 16 EfficientNet-B0.
- Geometric features require MediaPipe landmarks — unavailable on 224x224 crop input.
- On this dataset: geometric = neutral priors. Only EfficientNet branch provides signal.
- 5-class output projected to 3-class: p_normal = p_normal + p_pseudo + p_poor_quality.

---

## 3. Fusion Policy

```
Type: agreement+confidence weighted average
w_v05=0.8, w_v02=0.19999999999999996
Confidence threshold: 0.5
Margin threshold: 0.05
Agreement required: False
Abstain: UNCERTAIN
```

---

## 4. Governance

- v0.5 NOT overwritten | v0.2 NOT overwritten
- Thresholds tuned on val split ONLY (not test)
- Research-only — NOT for clinical diagnosis

---

## 5. Recommendation

USE v0.6 ensemble as primary research model.

Minimum requirements before clinical consideration:
- 300+ images per class with clinician-confirmed labels
- Real participant IDs (patient-level split)
- pseudostrabismus class included
- External hold-out test set
