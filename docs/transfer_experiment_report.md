# RemiCare Transfer Experiment Report: Korean Shared Feature Model

**Document ID:** REMICARE-DOC-EXP-004  
**Date:** September 2026  
**Status:** COMPLETE / EXPERIMENTAL ONLY  
**Classification:** Research Cross-Domain Transfer Experiment (Not a Clinical Diagnosis)

---

## 1. Executive Summary

This report documents the Phase 4 transfer experiment evaluating whether a binary classifier trained exclusively on laboratory eye-tracking data (the Korean Infrared Eye-Tracker dataset) can process and evaluate feature representations extracted from consumer webcam video (RemiCare Cover Test via MediaPipe Iris).

### Primary Findings

1. **Feature Space Compatibility:**  
   The standardized **Shared Feature Contract** (`shared-v1.0.0`) successfully extracted all **30 shared technical features** from `data/raw/sample.json` into `data/processed/remicare_shared_sample.json` without requiring any synthetic imputation, zero-filling, or fake phase tags.
2. **Inference Execution:**  
   The Korean-trained Random Forest model (`models/korean_shared_model.joblib`) processed the 30-feature vector and outputted:
   - **Status:** `TRANSFER_EXPERIMENT`
   - **Input Compatibility:** `true` (30/30 features present, 0 missing)
   - **Prediction:** `STRABISMUS`
   - **Class Probabilities:** `STRABISMUS: 72.23%`, `NORMAL: 27.77%`
3. **Critical Domain Shift Warning:**  
   `domainShiftWarning: true` is triggered and strictly enforced. The model was trained on 60Hz corneal reflection tracking with chin rests, whereas RemiCare operates at 30FPS via webcam facial landmark regression with natural head motion.
4. **Production Prohibition:**  
   This prediction **CANNOT** be deployed or mapped to `SCREENING_CLEAR` or `SCREENING_ATTENTION` in production. High technical performance on the Korean test set reflects pattern matching within its source hardware distribution, not clinical accuracy on webcams.

---

## 2. Cross-Domain Pipeline Architecture

The pipeline enforces strict domain separation:

```
[Korean Infrared Eye Tracker Dataset]
     (297 Train CSVs | ~60Hz)
               ↓
    [Shared Feature Extractor]
               ↓
    [Random Forest Training]
  (F1: 0.7241, Recall: 100.00% on 41 Korean Test CSVs)
               ↓
 [models/korean_shared_model.joblib]
               |
               | (Zero RemiCare data used during training)
               ↓
[RemiCare sample.json] (Consumer Webcam + MediaPipe, ~30 FPS)
               ↓
    [Validation Gate] (validate_screening_request)
               ↓
   [Preprocessing Gate] (preprocess_screening_request)
               ↓
[Shared Feature Contract Extractor] (extract_remicare_shared_payload)
               ↓
[data/processed/remicare_shared_sample.json] (No imputation)
               ↓
  [Transfer Inference Engine] (run_shared_transfer_inference)
               ↓
[Transfer Experiment Result + Domain Shift Warnings]
```

---

## 3. Handling the RemiCare Cover Test Phase Problem

A foundational requirement of Phase 4 is addressing temporal phase asymmetry:
- **RemiCare Cover Test:** Structured into sequential protocol phases (`BASELINE`, `COVER`, `UNCOVER`, `TRACKING`).
- **Korean Dataset:** `phase = UNKNOWN` (continuous binocular fixation/free-viewing protocol without occlusion phases).

### Enforcement Rule
- **No Fake Phases:** Korean data was **never** injected with synthetic phase tags.
- **Phase-Agnostic Feature Extraction:** The 30 shared features are computed globally across all recorded time steps ($t$). Features that fundamentally depend on occlusion transitions (e.g., refixation latency post-uncover, baseline-to-cover displacement) were **strictly excluded** from the shared model feature space.

---

## 4. Domain Shift & Feature Risk Categorization

The 30 shared features are stratified by cross-domain transfer risk:

| Category | Features | Formula / Property | Domain Shift Risk |
| :--- | :--- | :--- | :--- |
| **Tracking Validity** | `leftValidRatio`, `rightValidRatio`, `bothValidRatio` | Ratio of detected frames | **LOW** (Protocol-invariant detection confidence) |
| **Horizontal Disparity** | `meanDeltaX`, `medianDeltaX`, `stdDeltaX`, `minDeltaX`, `maxDeltaX`, `rangeDeltaX`, `meanAbsDeltaX` | $X_{right} - X_{left}$ | **LOW** (Translation-invariant inter-ocular distance) |
| **Vertical Disparity** | `meanDeltaY`, `medianDeltaY`, `stdDeltaY`, `minDeltaY`, `maxDeltaY`, `rangeDeltaY`, `meanAbsDeltaY` | $Y_{right} - Y_{left}$ | **LOW** (Translation-invariant vertical divergence) |
| **Dispersion & Kinematics** | `stdLeftX`, `stdLeftY`, `stdRightX`, `stdRightY`, `meanLeftVelocity`, `peakLeftVelocity`, `meanRightVelocity`, `peakRightVelocity`, `velocityDisparity` | Standard deviation & temporal derivatives | **LOW-MEDIUM** (Affected by frame rate: 60Hz vs 30FPS) |
| **Raw Viewport Positions** | `meanLeftX`, `meanLeftY`, `meanRightX`, `meanRightY` | Raw normalized coordinate means | **POTENTIAL_DOMAIN_SHIFT** (Vulnerable to camera crop, head pose, distance) |

### Analysis of Potential Domain Shift Features
Raw coordinate positions (`meanLeftX`, `meanLeftY`, etc.) are heavily influenced by the user's distance from the camera, facial centering in the viewport, and MediaPipe bounding box normalization. While preserved in the shared contract to maintain mathematical compatibility with the Korean feature dictionary, they must **not** be interpreted as clinically meaningful indicators of strabismus.

---

## 5. RemiCare Shared Sample Extraction Results

File: `data/processed/remicare_shared_sample.json`

```json
{
  "sampleId": "remicare-demo-sample-001",
  "source": "REMICARE_WEBCAM_MEDIAPIPE",
  "featureContractVersion": "shared-v1.0.0",
  "features": {
    "leftValidRatio": 0.8788,
    "rightValidRatio": 0.9394,
    "bothValidRatio": 0.8182,
    "meanDeltaX": 0.13433,
    "medianDeltaX": 0.1319,
    "stdDeltaX": 0.003296,
    "minDeltaX": 0.1315,
    "maxDeltaX": 0.1403,
    "rangeDeltaX": 0.0088,
    "meanAbsDeltaX": 0.13433,
    "meanDeltaY": -0.011289,
    "medianDeltaY": -0.0115,
    "stdDeltaY": 0.000544,
    "minDeltaY": -0.0121,
    "maxDeltaY": -0.0101,
    "rangeDeltaY": 0.002,
    "meanAbsDeltaY": 0.011289,
    "meanLeftX": 0.44329,
    "stdLeftX": 0.00215,
    "meanLeftY": 0.607641,
    "stdLeftY": 0.000187,
    "meanRightX": 0.577316,
    "stdRightX": 0.002936,
    "meanRightY": 0.596319,
    "stdRightY": 0.000442,
    "meanLeftVelocity": 0.022243,
    "peakLeftVelocity": 0.18058,
    "meanRightVelocity": 0.040084,
    "peakRightVelocity": 0.195541,
    "velocityDisparity": 0.01784
  },
  "missingFeatures": []
}
```

### Observation on Extraction Integrity
- **Total Shared Features Extracted:** 30
- **Missing Features:** 0 (empty list `[]`)
- **Imputation:** None. No zero-fill, median-fill, or placeholder constants were injected.

---

## 6. Transfer Inference Output & Disclaimers

Execution script: `scripts/run_remicare_transfer.py`

```json
{
  "status": "TRANSFER_EXPERIMENT",
  "inputCompatible": true,
  "model": "Random Forest",
  "prediction": "STRABISMUS",
  "probabilities": {
    "NORMAL": 0.2777,
    "STRABISMUS": 0.7223
  },
  "domainShiftWarning": true,
  "domainShift": {
    "sourceDomain": "KOREAN_INFRARED_EYE_TRACKER",
    "targetDomain": "REMICARE_WEBCAM_MEDIAPIPE",
    "riskLevel": "HIGH",
    "potentialShiftFeatures": [
      "meanLeftX",
      "meanLeftY",
      "meanRightX",
      "meanRightY"
    ],
    "warning": "Source domain is laboratory infrared eye tracker (~60Hz, calibrated physical pupil). Target domain is consumer webcam (~30FPS, MediaPipe landmark estimation). Feature distributions may diverge significantly."
  },
  "notice": "Research transfer experiment only — not a diagnosis.",
  "productionConstraint": "Transfer experiment output cannot be used directly as SCREENING_CLEAR or SCREENING_ATTENTION in production."
}
```

---

## 7. Five-Pillar Architectural Separation

To prevent misleading claims, the project maintains strict boundaries between five distinct analytical tiers:

1. **Tier A — Korean Model Performance:**  
   The Random Forest achieved 60.98% Accuracy, 56.76% Precision, 100.00% Recall, and 0.7241 F1 on 41 Korean test CSVs. This is solely a technical benchmark on infrared eye-tracking data.
2. **Tier B — Feature Compatibility:**  
   Both domains can compute 30 identical mathematical functions. Mathematical identity does not imply statistical equivalence.
3. **Tier C — Domain Shift:**  
   Sensor differences (infrared corneal reflection vs webcam RGB landmark estimation) and frame rates (60Hz vs 30FPS) induce substantial covariate shift.
4. **Tier D — RemiCare Transfer Experiment:**  
   Demonstrates that the software contracts and inference pipeline execute without crash or schema mismatch.
5. **Tier E — Clinical Validity:**  
   Currently **UNPROVEN (NONE)**. Validating clinical sensitivity, specificity, and positive predictive value requires independent clinical data collected with RemiCare under institutional review board (IRB) approved ophthalmological protocols.
