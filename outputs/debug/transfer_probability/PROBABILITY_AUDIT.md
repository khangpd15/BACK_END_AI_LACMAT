# RemiCare B2 Transfer Probability Audit

**Scope:** technical research audit only. No clinical diagnosis, clinical label, threshold change, or probability calibration was performed.

## Executive finding

The reported `STRABISMUS = 99.8%` is **not reproducible** from the repository's current sample and current B2 artifacts. All three B2 copies are byte-identical (`SHA-256: 8a12758be482330bcf994f9a5cc2c76c73614e97a5c2e2448d9240d0cfb3a2d8`) and return:

- Decision score: `0.815184122501696`
- `P(NORMAL)`: `0.306786892188952`
- `P(STRABISMUS)`: `0.693213107811048` (69.3213107811%)
- Prediction: `STRABISMUS`
- Domain shift: `WARNING`

Therefore, the current evidence does not support explaining 99.8% as the output of this saved B2 Logistic Regression artifact on `remicare-demo-sample-001`. A 99.8% UI observation requires the exact request payload, API response/log, and artifact hash from that run. Plausible causes include a different live sample, a stale/different backend artifact, or a UI/API provenance mismatch; these are hypotheses, not established causes.

## 1. End-to-end trace

`data/raw/sample.json` contains 3 cycles and 33 samples at reported `30.0 FPS`. Fresh extraction with `extract_shared_features_vector()` exactly matches `data/processed/remicare_shared_sample.json`: **TRUE**. The 30 values are ordered by `ALL_SHARED_FEATURES`, passed unchanged at B2 inference, standardized by the pipeline scaler, and scored by the Logistic Regression.

Artifacts: `input_features.json`, `transformed_features.json`, `scaled_features.json`, `decision_function.json`, `probabilities.json`, and `model_metadata.json` contain every numeric step.

## 2. Feature order and count

- Training/artifact feature order == canonical order: **PASS**
- Inference feature order == artifact order: **PASS**
- Feature count: **30**
- Canonical source: `app/services/shared_feature_contract.py::ALL_SHARED_FEATURES`

The inference service also contains a manually duplicated `FEATURE_ORDER`; it currently matches the canonical contract, but duplication is a future drift risk.

## 3. B2 normalization direction

Verified formula: **during training, each Korean horizontal disparity feature is divided by `2.502577710386657`; during RemiCare inference, those features are passed as-is.** This maps Korean coordinate scale down toward the RemiCare coordinate scale. Multiplying or dividing the RemiCare inference vector again would be inconsistent.

| Feature | Original | Inference transformed | Training formula |
| --- | --- | --- | --- |
| meanDeltaX | 0.13433 | 0.13433 | korean_value / ipd_scale_factor |
| medianDeltaX | 0.1319 | 0.1319 | korean_value / ipd_scale_factor |
| minDeltaX | 0.1315 | 0.1315 | korean_value / ipd_scale_factor |
| maxDeltaX | 0.1403 | 0.1403 | korean_value / ipd_scale_factor |
| meanAbsDeltaX | 0.13433 | 0.13433 | korean_value / ipd_scale_factor |

The sensitivity trace confirms that neither applying the documented scale in the wrong direction nor dividing again reproduces 99.8%; see `transformed_features.json`.

## 4. StandardScaler verification

- Scaler recomputation from B2-transformed Korean training rows only: **PASS**
- Direct z-score formula matches `StandardScaler.transform`: **PASS**
- Formula: `z = (RemiCare feature - scaler.mean_) / scaler.scale_`

Top 10 absolute deviations (technical distribution-shift diagnostics only):

| Feature | Raw | Scaler mean | Scaler std | Scaled Z | |Z| band |
| --- | --- | --- | --- | --- | --- |
| meanRightX | 0.577316 | 0.6780544242 | 0.09384196118 | -1.073490 | <3 |
| stdRightX | 0.002936 | 0.02841953872 | 0.0241192294 | -1.056565 | <3 |
| stdLeftX | 0.00215 | 0.03312847475 | 0.03022567489 | -1.024906 | <3 |
| meanLeftX | 0.44329 | 0.3533618889 | 0.08786166661 | 1.023519 | <3 |
| meanLeftVelocity | 0.022243 | 0.09540216162 | 0.08222979831 | -0.889692 | <3 |
| stdRightY | 0.000442 | 0.02005018182 | 0.02273117744 | -0.862612 | <3 |
| stdDeltaY | 0.000544 | 0.008069447811 | 0.00913801951 | -0.823532 | <3 |
| meanAbsDeltaY | 0.011289 | 0.0269449899 | 0.02146788615 | -0.729275 | <3 |
| maxDeltaY | -0.0101 | 0.03191518519 | 0.05835485493 | -0.719995 | <3 |
| stdLeftY | 0.000187 | 0.02158916498 | 0.0309601599 | -0.691281 | <3 |

Full 30-row scaler table is in `scaled_features.json` and full `mean_`/`scale_` maps are in `model_metadata.json`.

## 5. Logistic Regression coefficients and contributions

Intercept: `0.927797745430341`. The score reconstruction `intercept + sum(coef_i * z_i)` equals `0.815184122501695`: **PASS**.

Top 10 absolute contributions:

| Feature | Scaled value | Coefficient | Contribution |
| --- | --- | --- | --- |
| stdDeltaX | 0.403798 | 3.158097 | 1.275234 |
| stdRightY | -0.862612 | 1.123334 | -0.969001 |
| meanLeftVelocity | -0.889692 | 0.828329 | -0.736958 |
| stdRightX | -1.056565 | 0.689873 | -0.728896 |
| rightValidRatio | 0.382296 | 1.439645 | 0.550371 |
| maxDeltaY | -0.719995 | -0.692157 | 0.498349 |
| meanRightVelocity | -0.503310 | -0.943859 | 0.475053 |
| meanLeftX | 1.023519 | -0.386779 | -0.395875 |
| meanRightX | -1.073490 | 0.362682 | -0.389335 |
| leftValidRatio | -0.337131 | -1.005128 | 0.338859 |

These terms explain the technical class score; they are not clinical factors or thresholds.

## 6. Probability calculation

`sigmoid(0.815184122501696) = 0.693213107811048` and `predict_proba()[STRABISMUS] = 0.693213107811048`: **PASS**.

The frontend's one-decimal formatting would display the current value as `69.3%`, not `99.8%`.

## 7. Class weighting

- Configuration: `class_weight='balanced'`
- Korean training frequencies: NORMAL=86, STRABISMUS=211, total=297
- Effective weights: NORMAL=`1.726744186047`, STRABISMUS=`0.703791469194`

Balanced weighting changes training loss; it does not make the model unbiased and does not alone explain an extreme probability.

## 8. Same-sample cross-model comparison

Only the winning baseline/B1/B2 models were persisted. To complete the requested nine-model comparison, missing model-family variants were deterministically reconstructed from the repository's Korean training CSV with the documented training configurations. No reconstructed model was saved into production.

| Model | Prediction | P(NORMAL) | P(STRABISMUS) | Domain shift | Provenance |
| --- | --- | --- | --- | --- | --- |
| Baseline RF | STRABISMUS | 0.277724 | 0.722276 | WARNING | saved artifact |
| Baseline LR | STRABISMUS | 0.090249 | 0.909751 | WARNING | audit reconstruction |
| Baseline SVM | STRABISMUS | 0.153823 | 0.846177 | WARNING | audit reconstruction |
| B1 RF | STRABISMUS | 0.183933 | 0.816067 | WARNING | audit reconstruction |
| B1 LR | STRABISMUS | 0.153042 | 0.846958 | WARNING | saved artifact |
| B1 SVM | STRABISMUS | 0.145461 | 0.854539 | WARNING | audit reconstruction |
| B2 RF | STRABISMUS | 0.473361 | 0.526639 | WARNING | audit reconstruction |
| B2 LR | STRABISMUS | 0.306787 | 0.693213 | WARNING | saved artifact |
| B2 SVM | NORMAL | 0.396225 | 0.603775 | WARNING | audit reconstruction |

The B2 SVM shows a class/probability disagreement in this run. Scikit-learn's SVC class prediction uses its decision function, while `probability=True` uses a separately fitted probability mapping and the two can disagree. This is reported as observed and is not used to calibrate or reinterpret the B2 Logistic Regression.

## 9. Feature ablations

The B/C/D variants are audit-only retrains on Korean labels. They are distribution-shift diagnostics, not clinical validation.

| Case | Prediction | P(NORMAL) | P(STRABISMUS) | Definition |
| --- | --- | --- | --- | --- |
| A | STRABISMUS | 0.306787 | 0.693213 | Saved B2 LR, all 30 features |
| B | STRABISMUS | 0.488287 | 0.511713 | Audit-only B2 LR retrained without 7 horizontal disparity features |
| C | NORMAL | 0.541020 | 0.458980 | Audit-only balanced LR using validity + vertical disparity + dispersion only |
| D-raw | STRABISMUS | 0.090249 | 0.909751 | Audit-only baseline LR: raw Korean horizontal coordinates and raw RemiCare input |
| D-normalized | STRABISMUS | 0.306787 | 0.693213 | Audit-only B2 LR: Korean horizontal values divided by scale; RemiCare input as-is |

No calibration, threshold tuning, clipping, or smoothing was used.

## 10. Root-cause assessment

For the current reproducible output (69.3213%), the largest positive/negative score terms and largest z-shifts are listed above. The score is jointly produced by training imbalance/weighting, out-of-domain standardized values, coefficients, and the scaler; no single factor is a sufficient explanation.

For the specifically reported 99.8%, the root cause cannot be assigned from repository state because it is absent from source, saved reports, current artifacts, and current-sample inference. The required next evidence is the exact raw payload and backend response from that UI run plus the loaded artifact SHA-256. Treating 99.8% as reproduced would be misleading.

## 11. UI safety review

Read-only inspection of the deployed frontend source at `d:\AI_Check_Lac\src\components\binocular\CoverTestStep.jsx` confirms that it labels the value `Model Prediction` / `Class Probability`, displays a research/experiment badge, states it is not a medical diagnosis, and does not call the value “risk.” Recommended exact companion text remains: `Research-only output. Probability is model output and has not been clinically validated for RemiCare webcam data.` The external frontend was not modified by this audit.

## 12. Limitations and trustworthiness

- RemiCare samples evaluated: 1.
- RemiCare clinical ground-truth labels: 0.
- Korean rows are not an independent RemiCare validation set and have known participant-level leakage concerns.
- Cross-model and ablation probabilities do not establish clinical performance.
- The saved B2 model probability is technically reproducible for this exact artifact/vector, but **not trustworthy as a clinical probability for RemiCare** because of domain shift and absent target-domain ground truth.

## 13. Acceptance checklist

- A. Feature order: **PASS**
- B. Feature count: **30**
- C. B2 normalization: **verified — Korean horizontal features `/ 2.502577710386657` during training; RemiCare inference as-is**
- D. Scaler: **training-only verification PASS**
- E. Largest scaled deviations: **top 10 above**
- F. Largest LR contributions: **top 10 above**
- G. Decision score: **`0.815184122501696`**
- H. Probability: **NORMAL=`0.306786892188952`, STRABISMUS=`0.693213107811048`**
- I. Cross-model comparison: **completed (saved winners + audit-only reconstructions)**
- J. Clinical interpretation: **NONE**

> No clinical conclusion can be drawn from the current RemiCare 99.8% STRABISMUS output because there is currently no RemiCare clinical ground-truth validation dataset.
