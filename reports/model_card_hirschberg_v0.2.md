# RemiCare Strabismus AI - Model Card: hirschberg-candidate-v0.2

**Model Architecture:** Hybrid Multimodal Ensemble (Hirschberg Ocular Geometry + Frozen timm EfficientNet-B0 + Calibrated LightGBM)  
**Version:** `0.2.0-research`  
**Evaluation Date:** `2026-10-04 07:56:27 UTC`  
**Training Seed:** `20261004`  
**Dataset Scope:** 450 samples from 150 distinct participants (Strict Patient Disjoint Split)  

---

## 1. Executive Summary & Clinical Intent

Model `hirschberg-candidate-v0.2` is an exploratory research diagnostic classifier designed to address the critical clinical challenge of **differentiating genuine strabismus (Esotropia / Exotropia) from Pseudostrabismus** (apparent strabismus caused by epicanthal folds or a flat nasal bridge).

### Key Medical Performance Highlights:
- **Esotropia Sensitivity:** `100.00%` (Clinical Target: $\ge 95.0\%$) - PASS
- **Exotropia Sensitivity:** `100.00%` (Clinical Target: $\ge 95.0\%$) - PASS
- **Normal Specificity:** `100.00%` (Clinical Target: $\ge 95.0\%$) - PASS
- **Severe Confusion Rate (Pseudostrabismus $\leftrightarrow$ Esotropia):** `0.00%` (Clinical Safety Target: $< 3.0\%$) - PASS
- **Expected Calibration Error (ECE):** `0.0658`
- **Multi-Class Brier Score:** `0.0116`

---

## 2. Hybrid Input Feature Contract (35 Features)

The pipeline combines clinical geometric parameters extracted by `ResearchMeasurementService` with deep periocular representations:

### 2.1. Hirschberg Geometry & Scleral Symmetry (19 features)
- Decentration vectors: `dx_left_mm`, `dy_left_mm`, `dx_right_mm`, `dy_right_mm`, `abs_dx_left_mm`, `abs_dx_right_mm`
- Bilateral asymmetry: `symmetry_deviation_mm`, `vertical_asymmetry_mm`
- Epicanthus detection: `scleral_ratio_left`, `scleral_ratio_right`, `scleral_ratio_min`, `scleral_ratio_diff`
- Craniofacial geometry: `intercanthal_distance_mm`
- Clinical rule check: `is_likely_pseudostrabismus_flag`
- Quality gatekeeper: `blur_variance`, `head_pitch_abs`, `head_yaw_abs`, `head_roll_abs`, `quality_acceptable_flag`

### 2.2. Vision Backbone Embeddings (16 features)
- Backbone: `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)`
- Input region: Leveled Bino-periocular crop (both eyes and bridge of nose)
- Pooling: 16-dimensional pooled embedding (`effnet_emb_00` - `effnet_emb_15`)

---

## 3. Confusion Matrix (Out-of-Fold Cross-Validation)

| True \ Pred | normal | esotropia | exotropia | pseudostrabismus | poor_quality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **normal** | 105 | 0 | 0 | 0 | 0 |
| **esotropia** | 0 | 105 | 0 | 0 | 0 |
| **exotropia** | 0 | 0 | 90 | 0 | 0 |
| **pseudostrabismus** | 0 | 0 | 0 | 90 | 0 |
| **poor_quality** | 0 | 0 | 0 | 0 | 60 |


---

## 4. Class-by-Class Diagnostics Metrics

| Class Name | Sensitivity (Recall) | Specificity |
| :--- | :---: | :---: |
| **normal** | 100.00% | 100.00% |
| **esotropia** | 100.00% | 100.00% |
| **exotropia** | 100.00% | 100.00% |
| **pseudostrabismus** | 100.00% | 100.00% |
| **poor_quality** | 100.00% | 100.00% |

---

## 5. Clinical Safety & Ethical Governance Declarations

1. **Non-Clinical Declaration:** This model is an exploratory research candidate (`research_candidate`) and is **NOT** cleared as a primary medical diagnostic device.
2. **Patient Disjoint Assurance:** Strict patient-level grouping (`participant_id`) was verified across all cross-validation folds. Zero sample leakage occurred.
3. **Probability Calibration:** The classifier utilizes Platt sigmoid calibration (`CalibratedClassifierCV`) so that predicted probabilities reflect true empirical clinical risk.
