# RemiCare Research Dataset Schema

Status: Phase 5 preparation only. No model training is approved by this document.

## Dataset Version

`remicare-hirschberg-cover-dataset-v0.1`

## Unit Of Split

The split unit is `participant_id`.

All images, Cover samples, labels, sessions, and repeated distance-bucket samples from the same participant must stay in exactly one of:

- `train`
- `validation`
- `test`

Frame-level or session-level random splitting is forbidden because it leaks participant identity and repeated anatomy into validation/test.

## Required Participant Record

Each JSONL manifest line describes one participant/session measurement record:

```json
{
  "participant_id": "pseudonymous-id",
  "domain": "hirschberg_cover_protocol_v1",
  "consent_research": true,
  "consent_raw_video_or_landmarks": false,
  "age_years": 9,
  "glasses_on": false,
  "distance_bucket": "TARGET_20_25_CM",
  "hirschberg": {
    "image_path": "storage/path/or/local/path",
    "quality": {
      "focusStatus": "GOOD",
      "reflexCountPerEye": { "OD": 1, "OS": 1 }
    },
    "measurements": {
      "delta_h": null
    },
    "metadata": {
      "lighting": {},
      "device": {},
      "camera": {}
    }
  },
  "cover": {
    "samples_path": "storage/path/or/local/path",
    "quality": {},
    "phases": ["BASELINE", "COVER", "UNCOVER", "TRACKING"]
  },
  "ground_truth": {
    "label_source": "doctor",
    "exam_date": "YYYY-MM-DD",
    "horizontal_strabismus_type": "UNKNOWN",
    "intermittent": "UNKNOWN",
    "glasses_group": "UNKNOWN",
    "pseudostrabismus": "UNKNOWN",
    "exam_method": "UNKNOWN"
  }
}
```

## Domain Rules

Allowed for future training only after approval:

- `hirschberg_cover_protocol_v1`

Not allowed for threshold selection or training:

- `legacy_non_hirschberg`
- model predictions
- unlabeled production data
- data without research consent

Legacy data may be used only for exploratory detector/quality-gate development, and reports must say it is not clinical evidence.

## Ground Truth

Ground truth must come from:

- doctor
- orthoptist

Required label fields:

- `participant_id`
- `label_source`
- `exam_date`
- `horizontal_strabismus_type`
- `intermittent`
- `glasses_group`
- `pseudostrabismus`
- `exam_method`

Model output must never be copied into any ground-truth label field.

## Distance Experiment Mode

Distance-bucket reports may compare:

- focus pass ratio
- exact one-reflex-per-eye ratio
- repeatability
- `delta_h` distribution

The report must not choose a standard distance or diagnostic threshold. The researcher/clinician decides after review.

## Training Metrics To Report In Future

Classification:

- sensitivity
- specificity
- PPV
- NPV
- ROC-AUC
- PR-AUC
- confusion matrix

Measurement:

- MAE
- RMSE
- bias
- Bland-Altman summary

These fields remain `null` until an approved dataset is available and training is explicitly authorized.
