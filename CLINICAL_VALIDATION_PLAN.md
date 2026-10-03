# Clinical Validation Plan

## Goal

Validate RemiCare screening outputs against independent clinical ground truth. The current model output is screening/research support only and is not a diagnosis.

## Ground Truth Sources

- Ophthalmologist examination.
- Orthoptist measurements.
- Prism Alternate Cover Test (PACT).
- Clinical diagnosis recorded independently from model output.

Do not use previous model predictions as ground truth.

## Dataset Rules

- Split by patient, not by frame.
- Keep all sessions/frames from one patient in only one split.
- Track site/device/camera when available.
- Preserve low-quality and failed captures for quality analysis.

## Metrics

Classification:

- Sensitivity
- Specificity
- ROC-AUC
- PR-AUC
- PPV
- NPV
- Confusion matrix

Continuous/angle-related future models:

- MAE
- Bland-Altman
- Calibration curves

## Validation Phases

1. Internal retrospective validation with patient-level split.
2. Prospective silent-mode validation where model does not influence clinical workflow.
3. Threshold calibration with locked validation protocol.
4. External validation on a different clinic/device population.

## Deployment Gate

A model version can move to production only after:

- Feature contract is frozen.
- Model registry row/file is complete.
- Patient-level validation metrics meet predefined criteria.
- Clinical disclaimer and intended-use text are reviewed.
- Monitoring and rollback plan exists.
