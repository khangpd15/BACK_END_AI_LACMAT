# Participant-Group Retraining Report

## Outcome

A new **candidate-only** Logistic Regression model was trained and saved at `models/candidates/korean_groupcv_transfer_candidate.joblib`. The production B2 model was not modified.

Selected configuration:

- Feature set: `B2_NO_VIEWPORT_26` (26 features)
- `C`: `3.0`
- `class_weight`: `None`
- Korean training rows: `332`
- Rows excluded for contradictory labels: `6` from participants `['김민경']`
- Unique participants: `126`
- Participant overlap inside every outer train/validation fold: `0`
- Calibration applied: `False`
- Threshold tuning: `False`

## Nested unseen-participant probability metrics

- Brier score (lower is better): `0.18247425`
- Log loss (lower is better): `0.64515526`
- ROC AUC: `0.76114957`
- ECE, 10 bins: `0.06706418`
- Accuracy at 0.5: `0.72590361`
- Balanced accuracy at 0.5: `0.64856304`
- F1 at 0.5: `0.81081081`
- Confusion matrix: `[[46, 59], [32, 195]]`

Legacy B2 configuration on the same outer participant folds:

- Brier score: `0.20053802`
- Log loss: `0.63805534`
- ROC AUC: `0.74713656`
- ECE, 10 bins: `0.13346286`

The nested candidate improves Korean unseen-participant Brier by `0.01806377` relative to the legacy configuration on the same folds. Its log loss changes by `+0.00709992` (positive means worse), so the candidate is not uniformly better across probability metrics.

These are Korean source-domain, unseen-participant estimates. They are not RemiCare validation metrics.

## Same RemiCare sample

- Current B2 `P(STRABISMUS)`: `0.693213107811`
- Candidate `P(STRABISMUS)`: `0.946412664256`
- RemiCare ground truth: `NONE`
- Domain shift: `WARNING`

The candidate probability being numerically lower or higher does not make it more clinically accurate. There is no RemiCare label against which to measure its error.

## Deployment decision

**DO NOT replace the production model yet.** The candidate removes participant leakage from validation and optimizes uncalibrated Korean probability quality, but target-domain probability accuracy remains unknown. Promotion requires an independent, participant-level RemiCare validation set with ophthalmologist-confirmed labels and a pre-specified evaluation protocol.

No clinical conclusion can be drawn from either probability.
