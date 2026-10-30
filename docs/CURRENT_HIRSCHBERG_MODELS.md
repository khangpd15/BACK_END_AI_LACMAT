# Current Hirschberg inference bundle — 2026-10-30

The API now loads `classifier_retrain_20261030_v2` with the exact iris, pupil,
and reflex weights used to regenerate its training features. `model_bundle.json`
records all source paths and SHA256 hashes; `manifest.json` verifies the packaged
artifacts and inference code before model load. Classifier/localizer provenance is
also checked when constructing the engine. Source training checkpoints are copied
byte-for-byte, including the checkpoint's learned feature normalization.

| Component | Completed source run |
|---|---|
| Classifier | `cv_visibility_prior_20261030_v1/classifier_retrain_v2` |
| Pupil | `cv_visibility_prior_20261030_v1/successful_500_target15_refine_v2/block_03` |
| Reflex | `cv_visibility_prior_20261030_v1/successful_500_target15_refine_v2/reflex_03` |
| Iris | `cv_visibility_prior_20261030_v1/all_old_direct_v1/iris_pupil` |
| Eyelid / network initializers | Existing frozen runtime dependencies |

Pupil training was stopped at the owner's request: mean TRAIN error 1.798 source
pixels on 249 retained visible eyes, not below the earlier 1.5-pixel target.
Eight failed eye crops were excluded from that fitting metric. The unfinished
next training block is not packaged. Iris mean TRAIN error is 1.540 pixels;
reflex mean is 1.075 pixels on 260 retained visible eyes.

The three-class classifier was trained for 30 fixed epochs on 112 existing
labeled images (54 normal, 39 esotropia, 19 exotropia). New localizer features
and normalization, class-weighted loss, image-embedding dropout and auxiliary
geometry loss were used. TRAIN accuracy is 112/112; this is not independently
validated screening performance. The development-folder evaluation proposed
13 normal, one exotropia and four inconclusive cases under its warning policy,
with no esotropia proposal. No clinical ground truth was available for those
18 images. Visibility quality and generalization remain limitations.

## API behavior

Existing endpoints, CPU execution, upload limits, capture metadata requirements,
privacy/storage behavior and production quality gates are preserved. All inputs
now use online ROI detection; there is no exact-image training ROI lookup. The
production `research_capture_v2` policy still does not treat common vector
direction as a diagnosis or an automatic rejection. Thus its counts may differ
from the strict development script; the development warning policy is not enabled
as the API default. No missing measurement is converted into a normal result.

Responses remain `researchOnly: true`, `clinicalDiagnosis: null`, and scores are
uncalibrated. Updating model files does not establish clinical readiness.

## Reproduce the import

```powershell
.\.venv312\Scripts\python.exe scripts/import_trained_hirschberg.py --source D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD
.\.venv312\Scripts\python.exe -m pytest tests/model
```

The importer refreshes the vendored inference subset and then installs this
explicitly selected bundle. `scripts/update_trained_hirschberg_bundle.py` can
refresh the bundle alone. No training images, uploads, review CSVs, patient
annotations or environment credentials are imported or committed.

## Backend verification

36 model/API tests passed. The real API processed all18 local development images
without persisting uploads:12 NORMAL,1 EXOTROPIA,5 INCONCLUSIVE. On28 mutually
detected eyes, maximum coordinate drift was below0.008px. ROI detection coverage
differed on3 images between environments; end-to-end numerical parity therefore
is NOT established. The aggregate report explicitly sets numerical_parity_passed
to false. No clinical accuracy is inferred from these results.
