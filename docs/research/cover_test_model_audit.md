# Cover Test Model Audit

Date: 2026-10-04

## Runtime Finding

The Cover Test runtime was still coupled to Korean-transfer artifacts:

- `app/models/korean_shared_model.joblib`
- `app/models/remicare_transfer_model.joblib`
- `app/models/remicare_15fps_candidate.joblib`
- `models/candidates/remicare_15fps_candidate.joblib`

The `remicare_15fps_candidate.joblib` artifact metadata showed:

- `name`: Korean 10-15 FPS robust transfer candidate
- `status`: RESEARCH_COMPARISON_ONLY
- `remicare_labels_used`: 0
- `source_sampling_hz`: approximately 60Hz
- `clinical_meaning`: None

This means it was not a RemiCare-trained Cover Test model and should not be used
to produce current screening results.

## Runtime Change

Korean transfer runtime has been retired:

- Removed `/api/v1/transfer/strabismus` router registration.
- Removed startup preload of Korean transfer service.
- Removed `app/api/transfer.py`.
- Removed `app/services/korean_transfer.py`.
- Removed Korean `.joblib` artifacts listed above.
- `CoverTestSessionService` now stores only the `FpsModelService` result and no
  longer lets Korean comparison output override the primary result.

Until a RemiCare-trained Cover Test model is promoted, Cover Test model inference
returns safe `INCONCLUSIVE` rather than a Korean-transfer prediction.

## Local Cover Test Video Dataset

Current local folder:

`data/VIDEO_COVER_TEST/`

Observed files:

| Class folder | Video count |
|---|---:|
| `ESOTROPIA` | 2 |
| `EXOTROPIA` | 3 |
| `NORMAL` | 1 |

This is useful for bootstrapping a RemiCare-specific pipeline, but it is too small
for a reliable clinical model by itself.

## Next Training Direction

Train a new candidate only from RemiCare Cover Test videos or clinician-labeled
Cover Test numeric trajectories:

1. Extract MediaPipe iris/eye landmarks frame-by-frame from each video.
2. Normalize each video into Cover Test phases if phase labels are available.
3. Extract refixation and temporal features:
   - baseline median position
   - peak displacement after uncover
   - time to peak
   - settling time
   - velocity/acceleration peak
   - cycle consistency
   - left/right asymmetry
   - quality and valid-frame ratios
4. Train only a research candidate first:
   - Logistic Regression balanced
   - RandomForest balanced
   - ExtraTrees balanced
   - abstention policy for low confidence
5. Require patient/video-level split. Do not split frames from the same video into
   both train and test.
6. Do not use Korean predictions as labels.

## Deployment Recommendation

Do not deploy a Cover Test classifier until more RemiCare-labeled videos are added.
Current production-safe behavior is to persist raw numeric trajectories and return
`INCONCLUSIVE` for model inference when no approved RemiCare artifact is loaded.
