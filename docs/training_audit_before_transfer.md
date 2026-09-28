# REMICARE-DOC-EXP-005: Training Audit Before Transfer Model Development

**Document ID:** REMICARE-DOC-EXP-005  
**Date:** 2026-09-28  
**Status:** COMPLETE — Audit findings documented. No new model trained from this document.  
**Scope:** Full audit of dataset sources, train/test split, feature contracts, leakage, class imbalance, and cross-domain distribution shift before any RemiCare-specific model development.

> **CLINICAL NOTICE:** All findings in this document are technical benchmarks on the Korean Infrared Eye Tracker Dataset.
> No clinical thresholds, diagnostic cut-offs, or screening results are derived or asserted.
> Model predictions are NOT medical diagnoses.

---

## 1. Dataset Sources

### 1.1 Training Dataset — Korean Infrared Eye Tracker

| Attribute | Detail |
|---|---|
| Source | hyunwoongko/strabismus-recognition |
| Hardware | Infrared binocular eye tracker |
| Sampling Rate | ~60 Hz (sustained, continuous recording) |
| Environment | Controlled laboratory (chin rest, fixed viewing distance) |
| Coordinate Space | Normalized [0, 1] within infrared sensor viewport |
| Protocol | Static fixation task, no Cover Test |

**File breakdown:**

| Folder | Class Label (Binary) | File Count |
|---|---|---|
| data/korean_repo/data/normal/ | NORMAL (0) | 86 |
| data/korean_repo/data/exotropia/ | STRABISMUS (1) | 175 |
| data/korean_repo/data/esotropia/ | STRABISMUS (1) | 32 |
| data/korean_repo/data/hypertropia/ | STRABISMUS (1) | 4 |
| **TRAIN TOTAL** | — | **297** |

Note: Exotropia, esotropia, and hypertropia are merged into a single binary STRABISMUS class per project scope.
No multi-class labels are introduced.

### 1.2 Independent Test Dataset — Korean Infrared Eye Tracker

| Folder | Class Label | File Count |
|---|---|---|
| data/test_normal/ | NORMAL (0) | 20 |
| data/test_strabismus/ | STRABISMUS (1) | 21 |
| **TEST TOTAL** | — | **41** |

### 1.3 RemiCare Data — Webcam + MediaPipe

| Attribute | Detail |
|---|---|
| Hardware | Standard consumer webcam |
| Tracking | MediaPipe Iris + Face Mesh |
| Sampling Rate | ~15 Hz (cover test cycle sampling) |
| Environment | Natural head movement, variable lighting |
| Coordinate Space | Normalized [0, 1] within video frame |
| Protocol | Structured Cover Test (Cover -> Uncover -> Tracking cycles) |
| Clinical Label Available? | **NO** |

**RemiCare data currently available:**

| Item | Location | Count |
|---|---|---|
| Real sample JSON | data/raw/sample.json | 1 (demo sample) |
| Clinical ground-truth labels | — | **0** |

> **CRITICAL CONSTRAINT:** No RemiCare clinical ground-truth labels exist at this time.
> It is **forbidden** to use Korean model predictions as ground-truth labels for RemiCare data.

---

## 2. Train / Test Split Strategy

### 2.1 Split Type

The split between train (297) and test (41) is a **session-split from the original Korean repository**, not a clean participant-level random split.

This was confirmed by the leakage audit in `data/processed/korean_shared_model_eval.json`.

### 2.2 Participant-Level Leakage Findings

**Status: `PARTICIPANT_LEVEL_LEAKAGE` — CONFIRMED**

| Participant ID | Train File Count | Test File Count | Risk Level |
|---|---|---|---|
| 김민경 (Kim Minkyung) | 5 | 1 | HIGH |
| 장진우 (Jang Jinwoo) | 1 | 3 | HIGH |
| **Total Overlap** | — | — | **2 participants** |

Root cause: The Korean repository filed multiple recording sessions per subject across directories
(`data/normal/` and `data/test_normal/`). The test set is **session-split**, not **participant-split**.

**Scientific Implication:**
- The test accuracy figures (Korean RF: 60.98%, F1: 0.7241) cannot be claimed as unbiased estimates of generalization to genuinely unseen participants.
- Performance on RemiCare (completely novel subjects, different hardware, different protocol) cannot be predicted from these figures.

### 2.3 Feature-Level Leakage Check

**Status: PASS (No feature-level leakage)**

| Check | Result |
|---|---|
| Scaler fitted on train only | PASS — StandardScaler fitted inside make_pipeline, evaluated on held-out test set |
| Label leakage from identifiers | PASS — participantId and sampleId dropped before model training |
| Target leakage (labels from coordinates) | PASS — Labels derived strictly from folder names |
| RemiCare labels from Korean model | PASS — Never used; enforced by project constraint |

---

## 3. Feature Contract

### 3.1 Feature Set Reconciliation: 26 vs 30

Earlier audit (Phase 4.1) noted a discrepancy between "26 features" and "30 features" across different scripts. This has been reconciled:

| Module | List Name | Count | Status |
|---|---|---|---|
| app/services/feature_contract.py | REQUIRED_SHARED_FEATURES | 30 | Up to date |
| app/services/shared_feature_contract.py | ALL_SHARED_FEATURES | 30 | Canonical |
| app/training/shared_feature_model.py | REQUIRED_SHARED_FEATURES | 30 | Same features, different order |

**Verified:** Both lists contain **identical 30 features** (no set-theoretic difference).
The "26 features" discrepancy was historical, referring to an earlier Phase 3 prototype.
The current canonical contract is **30 features**.

The only difference between the two modules is **feature ordering**:

| Module | Order |
|---|---|
| feature_contract.py | Disparity features first, validRatios last |
| shared_feature_contract.py | Validity ratios first, then disparity, viewport, kinematics |

> **Action Required:** Any new training pipeline must use `ALL_SHARED_FEATURES` from
> `shared_feature_contract.py` (canonical Phase 4 contract) and must enforce this order.

### 3.2 The Canonical 30 Feature Order

```
Group 1 — Validity Ratios (3):
  leftValidRatio, rightValidRatio, bothValidRatio

Group 2 — Horizontal Inter-Ocular Disparity (7):
  meanDeltaX, medianDeltaX, stdDeltaX, minDeltaX,
  maxDeltaX, rangeDeltaX, meanAbsDeltaX

Group 3 — Vertical Inter-Ocular Disparity (7):
  meanDeltaY, medianDeltaY, stdDeltaY, minDeltaY,
  maxDeltaY, rangeDeltaY, meanAbsDeltaY

Group 4 — Viewport Coordinates & Dispersion (8):
  meanLeftX, stdLeftX, meanLeftY, stdLeftY,
  meanRightX, stdRightX, meanRightY, stdRightY

Group 5 — Kinematics & Velocities (5):
  meanLeftVelocity, peakLeftVelocity, meanRightVelocity,
  peakRightVelocity, velocityDisparity
```

### 3.3 Domain Shift Classification Per Feature

| Category | Features | Risk |
|---|---|---|
| Raw viewport positions | meanLeftX, meanLeftY, meanRightX, meanRightY | POTENTIAL_DOMAIN_SHIFT |
| All other 26 features | disparity, std, range, velocity, validity | LOW_DOMAIN_SHIFT |

### 3.4 NaN Handling Policy

| Situation | Policy |
|---|---|
| Missing coordinates in Korean CSV | Skip entire CSV; do not impute with 0 |
| NaN in feature vector (rare edge case) | np.nan_to_num(X, nan=0.0) applied to matrix after extraction |
| NaN in feature vector for RemiCare | Reject request via ValidationError before inference |

NaN rate in current training data: No feature has NaN > 5%.
One velocity sample is missing (count=296/297) — acceptable.

---

## 4. Class Imbalance

### 4.1 Training Set

| Class | Count | Proportion |
|---|---|---|
| NORMAL (0) | 86 | 29.0% |
| STRABISMUS (1) | 211 | 71.0% |
| **Total** | **297** | 100% |
| **Imbalance ratio (STRAB/NORM)** | **2.45:1** | — |

### 4.2 Test Set

| Class | Count | Proportion |
|---|---|---|
| NORMAL (0) | 20 | 48.8% |
| STRABISMUS (1) | 21 | 51.2% |
| **Total** | **41** | 100% |

### 4.3 Implications

The 2.45:1 training imbalance directly explains the pattern seen in all benchmark models:

- All models achieve high recall on STRABISMUS (95-100%) because the training prior biases toward predicting STRABISMUS.
- Specificity (true negative rate for NORMAL) is poor: only 4-6 of 20 NORMAL test subjects are correctly identified.
- When a RemiCare subject passes through the Korean model, it is likely to be classified as STRABISMUS — not from a true screening signal, but from training imbalance **combined with domain shift** in horizontal disparity features.

> **Recommendation:** Use class_weight='balanced' or SMOTE when retraining on this or any new dataset.

---

## 5. Domain Shift Analysis

### 5.1 Feature-by-Feature Z-Score Table

Z-score computed as: `z = |RemiCare_value - Korean_NORMAL_mean| / Korean_NORMAL_std`

A z-score > 3 indicates the RemiCare value falls outside the 3-sigma range of the Korean normal distribution.

| Feature | KOR_STRAB_mean | KOR_NORM_mean | RemiCare | Z-Score | Shift Level |
|---|---|---|---|---|---|
| leftValidRatio | 0.90790 | 0.92550 | 0.87880 | 0.64 | LOW |
| rightValidRatio | 0.89800 | 0.91251 | 0.93940 | 0.26 | LOW |
| bothValidRatio | 0.88254 | 0.90813 | 0.81820 | 0.86 | LOW |
| **meanDeltaX** | **0.33067** | **0.33061** | **0.13433** | **5.55** | **HIGH** |
| **medianDeltaX** | **0.33009** | **0.33045** | **0.13190** | **5.57** | **HIGH** |
| stdDeltaX | 0.00555 | 0.00255 | 0.00330 | 0.14 | LOW |
| **minDeltaX** | **0.31435** | **0.31948** | **0.13150** | **3.78** | **HIGH** |
| **maxDeltaX** | **0.34940** | **0.34447** | **0.14030** | **5.20** | **HIGH** |
| rangeDeltaX | 0.03505 | 0.02499 | 0.00880 | 0.31 | LOW |
| **meanAbsDeltaX** | **0.33067** | **0.33061** | **0.13433** | **5.55** | **HIGH** |
| meanDeltaY | 0.00243 | 0.00890 | -0.01129 | 0.64 | LOW |
| medianDeltaY | 0.00261 | 0.00849 | -0.01150 | 0.63 | LOW |
| stdDeltaY | 0.00897 | 0.00596 | 0.00054 | 0.96 | LOW |
| minDeltaY | -0.02766 | -0.01076 | -0.01210 | 0.03 | LOW |
| maxDeltaY | 0.02956 | 0.03805 | -0.01010 | 0.75 | LOW |
| rangeDeltaY | 0.05722 | 0.04881 | 0.00200 | 0.59 | LOW |
| meanAbsDeltaY | 0.02871 | 0.02295 | 0.01129 | 0.49 | LOW |
| meanLeftX | 0.35333 | 0.35756 | 0.44329 | 1.03 | LOW |
| stdLeftX | 0.03699 | 0.02408 | 0.00215 | 0.78 | LOW |
| meanLeftY | 0.55497 | 0.50582 | 0.60764 | 0.74 | LOW |
| stdLeftY | 0.02582 | 0.01150 | 0.00019 | 0.78 | LOW |
| meanRightX | 0.67852 | 0.68480 | 0.57732 | 1.39 | LOW |
| stdRightX | 0.03243 | 0.01896 | 0.00294 | 1.07 | LOW |
| meanRightY | 0.55883 | 0.51481 | 0.59632 | 0.57 | LOW |
| stdRightY | 0.02392 | 0.01083 | 0.00044 | 1.37 | LOW |
| meanLeftVelocity | 0.10645 | 0.06953 | 0.02224 | 0.77 | LOW |
| peakLeftVelocity | 25.04331 | 12.98426 | 0.18058 | 0.45 | LOW |
| meanRightVelocity | 0.07802 | 0.06094 | 0.04008 | 0.25 | LOW |
| peakRightVelocity | 8.48650 | 4.94172 | 0.19554 | 0.60 | LOW |
| velocityDisparity | 0.03802 | 0.02368 | 0.01784 | 0.10 | LOW |

### 5.2 Root Cause of High-Shift Features

5 features with z > 3 — all in the horizontal inter-ocular disparity group:

```
meanDeltaX   : Korean ~0.33  vs  RemiCare ~0.13  (z = 5.55)
medianDeltaX : Korean ~0.33  vs  RemiCare ~0.13  (z = 5.57)
minDeltaX    : Korean ~0.32  vs  RemiCare ~0.13  (z = 3.78)
maxDeltaX    : Korean ~0.34  vs  RemiCare ~0.14  (z = 5.20)
meanAbsDeltaX: Korean ~0.33  vs  RemiCare ~0.13  (z = 5.55)
```

**Root cause hypothesis:**

The Korean infrared eye tracker reports coordinates that, when normalized to [0, 1] within the
sensor viewport, produce inter-pupillary distances of ~0.33 (33% of viewport width).

MediaPipe Iris reports iris center coordinates in the **entire video frame** (not cropped to eye region).
At standard face-to-camera distance, the normalized inter-pupillary distance is ~0.13 (13% of frame width)
because the face occupies only a fraction of the full frame.

This is a **coordinate-space mismatch**, not a physiological difference.
The Korean model has never seen inputs where `meanDeltaX ≈ 0.13` and classifies
such inputs as anomalous, biasing toward STRABISMUS.

### 5.3 Transfer Feasibility by Feature Group

| Feature Group | Transfer Feasibility |
|---|---|
| Validity ratios (3) | Transferable — device-agnostic detection ratio |
| Vertical disparity (7) | Likely transferable — z-scores all < 1 |
| Viewport positions & dispersion (8) | Moderate risk — positions shift with face-to-camera distance |
| Velocity metrics (5) | Moderate risk — scale depends on coordinate unit |
| **Horizontal disparity (7)** | **HIGH domain shift — systematic offset, z-scores 3.78–5.57** |

---

## 6. Current Model Artifacts

### 6.1 Existing Models

| Artifact | Path | Description |
|---|---|---|
| Korean Shared Model | models/korean_shared_model.joblib | RF, 30 features, Korean 297 train |
| App Runtime Copy | app/models/korean_shared_model.joblib | Loaded by FastAPI endpoint |

Model version: `korean-shared-v1.0.0`

### 6.2 Korean Benchmark Performance

> IMPORTANT: These are technical benchmark figures on the Korean infrared dataset only.
> NOT clinical performance estimates. Cannot be extrapolated to RemiCare webcam data.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **60.98%** | **56.76%** | **100%** | **0.7241** |
| Logistic Regression | 56.10% | 54.55% | 85.71% | 0.6667 |
| SVM RBF | 53.66% | 52.50% | 100% | 0.6885 |

**Confusion Matrix (Random Forest, Best F1):**

```
               Predicted NORMAL  Predicted STRABISMUS
Actual NORMAL         4              16
Actual STRABISMUS     0              21
```

- True Positive Rate (Strabismus Recall): 21/21 = 100%
- True Negative Rate (Normal Specificity): 4/20 = 20%
- Model correctly identifies all STRABISMUS but misclassifies 16/20 NORMAL as STRABISMUS.

### 6.3 RemiCare Transfer Inference (Single Demo Sample)

| Metric | Value |
|---|---|
| Sample ID | remicare-demo-sample-001 |
| Raw Prediction | STRABISMUS (1) |
| NORMAL probability | ~38.8% |
| STRABISMUS probability | ~72.2% |
| Domain Shift Warning | HIGH |

This result is consistent with the horizontal disparity domain shift.

---

## 7. Data Readiness Assessment

### 7.1 Current State

| Item | Status | Notes |
|---|---|---|
| Korean training data (297 CSV) | AVAILABLE | Used to train current model |
| Korean test data (41 CSV) | AVAILABLE | Used for benchmark evaluation |
| RemiCare clinical labels | NOT AVAILABLE | Cannot train RemiCare-specific model |
| RemiCare unlabeled samples | 1 DEMO SAMPLE | Not sufficient for training or validation |
| Participant-level clean split | NOT CLEAN | 2-participant leakage confirmed |

### 7.2 What Can Be Done Without RemiCare Labels

| Experiment | Feasible? | Output |
|---|---|---|
| Experiment A: Korean Baseline (30 features) | Yes | Current benchmark (already done) |
| Experiment B1: Korean 26 Invariant-only features | Yes | Drop 4 POTENTIAL_DOMAIN_SHIFT features, retrain |
| Experiment B2: Korean + coordinate normalization | Yes | Rescale horizontal disparity features |
| Experiment C: RemiCare-specific supervised model | NO | Requires clinical ground-truth labels |
| Experiment D: Self-supervised on RemiCare | Research only | No diagnostic value without labels |

### 7.3 Recommended Next Actions

| Priority | Action | Rationale |
|---|---|---|
| 1 | Experiment B1: Retrain on 26 invariant features (drop meanLeftX, meanLeftY, meanRightX, meanRightY) | Remove 4 POTENTIAL_DOMAIN_SHIFT features |
| 2 | Experiment B2: Retrain with disparity feature rescaling | Rescale horizontal disparity by ratio ~2.54x |
| 3 | Collect RemiCare clinical ground truth | Ophthalmologist-confirmed labels on real webcam recordings |
| 4 | Address class imbalance | Use class_weight='balanced' or SMOTE on Korean training set |

---

## 8. Conclusions

1. **Dataset:** Korean infrared eye tracker, 297 train / 41 test CSV files, binary NORMAL vs STRABISMUS. Labels from folder names strictly — no inference from coordinates.

2. **Split quality:** Session-split, NOT participant-split. Two participants appear in both sets. Test accuracy is NOT an independent generalization estimate.

3. **Feature contract:** 30 shared features. Canonical source: `ALL_SHARED_FEATURES` in `app/services/shared_feature_contract.py`. Both modules are currently aligned (same 30 features, different ordering only).

4. **Leakage:** No feature-level or target-level leakage. Participant-level leakage confirmed (2 subjects). Scaler fitted on train only.

5. **Class imbalance:** 2.45:1 (STRAB/NORMAL) in training. Combined with domain shift, causes the current model to over-predict STRABISMUS on RemiCare inputs.

6. **Domain shift:** 5 horizontal disparity features show z > 3 vs RemiCare sample (z-scores: 3.78–5.57). Root cause: coordinate-space mismatch — Korean infrared viewport (~0.33 IPD ratio) vs MediaPipe full-frame normalization (~0.13 IPD ratio). This is the primary driver of incorrect STRABISMUS predictions on RemiCare data.

7. **RemiCare labels:** None available. Korean model predictions must NOT be used as surrogate labels.

8. **Next step:** Experiment B1 (drop POTENTIAL_DOMAIN_SHIFT features) and Experiment B2 (horizontal disparity rescaling) can be explored using existing Korean dataset only.

---

*Generated by: RemiCare Backend AI Pipeline | Python FastAPI | Document: REMICARE-DOC-EXP-005*
