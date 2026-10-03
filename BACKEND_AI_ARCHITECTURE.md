# Backend AI Architecture

## Current Runtime

```text
Frontend AI_CHECK_LAC
  -> /api/v1/cover-test/sessions
  -> FastAPI validation
  -> PostgreSQL/Supabase metadata
  -> Supabase raw JSON storage
  -> 10-15 FPS model consensus
  -> cover_test_results
  -> response
```

Fallback research endpoint:

```text
Frontend
  -> /api/v1/transfer/strabismus
  -> Data Integrity Gate
  -> 30 shared features
  -> Korean transfer/candidate models
  -> research-only response
```

Image ROI endpoint:

```text
Bilateral eye ROI image
  -> MIME/size validation
  -> ROI safety guard
  -> image quality gate
  -> 224x224 RGB ImageNet normalization
  -> ONNX Runtime
  -> metadata-only DB save
```

## Model Abstractions

Current concrete services:

- `FpsModelService`: tabular 10-15 FPS Cover Test candidate.
- `KoreanTransferService`: research transfer/comparison model.
- `StrabismusInferenceService`: bilateral ROI image ONNX model.

Recommended next abstractions:

- `StaticFeatureModel`
- `TemporalFeatureModel`
- `ImageModel`
- `FusionEngine`

Do not introduce fusion weights until each source has validated calibration and quality gates.

## Health

Current `GET /health` returns service version, transfer model info, keep-alive status, and optional DB check. Image ROI model has `GET /api/v1/strabismus/health`.

Recommended addition:

```json
{
  "status": "ok",
  "modelLoaded": true,
  "modelVersion": "...",
  "featureVersion": "shared-v1.0.0"
}
```

## API Compatibility

No endpoint contract was changed in this pass. Existing frontend paths remain:

- `/api/v1/cover-test/sessions`
- `/api/v1/transfer/strabismus`
- `/api/cover-test/sessions`
- `/api/v1/strabismus/predict`
