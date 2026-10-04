# RemiCare Backend AI Audit

Audit date: 2026-10-03  
Scope: `D:\REMICARE-STRABISMUS-AI` backend. `D:\AI_Check_Lac` was inspected read-only for frontend API/feature context and was not modified.

## 1. Current Architecture

FastAPI backend with two active AI/data paths:

- `POST /api/v1/cover-test/sessions`: primary frontend path. Receives multipart Cover Test session metadata and raw cycle JSON, stores metadata in PostgreSQL/Supabase-compatible tables, uploads raw JSON to Supabase Storage, then optionally runs the 10-15 FPS model consensus path.
- `POST /api/v1/strabismus/predict`: bilateral eye ROI image endpoint. Performs MIME/size validation, ROI safety guard, quality gate, ONNX inference, and metadata-only persistence.

The backend initializes Cover Test FPS service and bilateral ONNX services during FastAPI lifespan. Korean transfer services and artifacts were retired from runtime.

## 2. API Flow

Primary Cover Test flow:

1. Frontend builds `multipart/form-data` in `D:\AI_Check_Lac\src\api\coverTestApi.js`.
2. Backend parses `session_metadata` and `cycle_N_raw` in `app/api/cover_test.py`.
3. `CoverTestSessionService.persist_session` validates raw trajectories.
4. Repository persists `cover_test_sessions`, `cover_test_cycles`, and optionally `cover_test_results`.
5. Raw JSON is uploaded to Supabase Storage using `storage_service.py`.
6. Optional AI inference runs `fps_model_service.py`. If no approved RemiCare-trained artifact is present, the service returns `INCONCLUSIVE`.

Fallback/legacy flow:

- `POST /api/cover-test/sessions` stores JSON locally in `data/cover_test/sessions`. This is still registered because frontend fallback code exists in `aiBackendService.js`.

Retired transfer flow:

- `POST /api/v1/transfer/strabismus` and Korean transfer model comparison were removed from runtime because the artifacts were trained on Korean infrared eye-tracker data and are not useful for current RemiCare Cover Test deployment.

## 3. Feature Pipeline

Current backend feature extraction is centralized in `app/services/shared_feature_contract.py`:

- Contract version: `shared-v1.0.0`
- Shared feature count: 30
- Feature source: normalized iris coordinates, validity flags, timestamps.
- Velocity uses timestamp delta and does not assume fixed FPS.
- Missing shared features return `None`; model services must reject them.

Frontend Cover Test currently sends raw samples, not model-ready features, through the primary backend path. This is good because Python remains the source of truth for backend model inference.

## 4. Model Pipeline

Runtime model paths:

- Cover Test RemiCare-trained candidate:
  - pending; Korean transfer artifacts were deleted.
  - `FpsModelService` remains as the interface and returns safe `INCONCLUSIVE` without an approved artifact.
- Bilateral ROI ONNX:
  - default `app/models/best_model.onnx`, configurable by `MODEL_PATH`

Model loading is cached in singleton services. The code now validates model input dimension metadata against artifact `feature_names` where sklearn exposes `n_features_in_`.

## 5. Database Schema

Current relational tables:

- `cover_test_sessions`
- `cover_test_cycles`
- `cover_test_results`
- `strabismus_screenings`

Object storage stores raw Cover Test JSON and manifest paths under `cover-test-raw/YYYY/MM/session_id`. The current code deliberately does not persist eye/facial images for privacy.

## 6. Current Model/Version

Known declared versions:

- Shared feature contract: `shared-v1.0.0`
- Cover Test FPS service default version: `10-15fps-v1.1.0` interface only; no approved runtime artifact after Korean cleanup.
- Bilateral ROI ONNX service version: `remicare-bilateral-resnet18-v1`
- Bilateral ROI threshold: `0.20`

Exact artifact metadata should be checked in a working Python 3.11 environment because the local `.venv` interpreter currently points to a missing Python installation.

## 7. Current Feature Contract

Backend model contract:

- 30 shared features for transfer path.
- 14 selected features for the current 10-15 FPS candidate when its artifact declares that list.
- No raw client feature vector is trusted.
- Production code now rejects missing/non-finite required features instead of zero-filling.

Frontend static ONNX contract:

- `D:\AI_Check_Lac\FEATURE_CONTRACT_V1.md` documents a separate 10-feature client-side/static ONNX contract.
- This is not the same as the backend Cover Test 30/14-feature contract.

## 8. Potential JS/Python Mismatch

Observed risk areas:

- Frontend static 10-feature ONNX feature contract is separate from backend Cover Test model contract.
- Backend docs/README mention old tests and module names that are no longer present.
- Frontend timestamp sampler uses real elapsed time and preserves `t`; backend now rejects duplicate/non-monotonic timestamps on both current and legacy save paths.
- Backend shared extractor uses Python/NumPy; parity test now compares it to a JS reference extractor in `tests/feature_parity`.

## 9. Performance Bottlenecks

- Startup loads transfer and ONNX services; DB init also runs at startup.
- Multipart session persistence uploads multiple JSON blobs plus DB writes.
- Optional inference after save can add latency to the storage endpoint.
- `auto_migrate_schema` runs DDL checks at startup; acceptable for small deployment, should be reviewed for production migration discipline.
- Current performance logging measures total HTTP duration, but stage-level timing is not yet exposed per request.

## 10. Data Integrity Risks

Fixed in this pass:

- 10-15 FPS model no longer substitutes missing required features with `0.0`.
- Missing model now returns INCONCLUSIVE instead of fallback NORMAL.
- Duplicate timestamps are rejected on current and legacy Cover Test storage paths.
- Transfer/candidate model artifacts now validate `n_features_in_` against `feature_names`.

Remaining:

- Legacy local file save still exists; keep only while frontend fallback needs it.
- Auto-migration can drop obsolete image table in PostgreSQL; verify this is acceptable before production startup.

## 11. Privacy Risks

Strengths:

- Primary session path stores raw numeric trajectories, not face/eye images.
- Strabismus ROI endpoint discards image bytes and persists metadata only.
- PII keys are blocked in metadata/sample payloads on session persistence.

Remaining:

- Numeric eye trajectories are sensitive biometric-adjacent data and need retention/deletion policy enforcement outside code.
- Local legacy save path writes raw samples to disk; production should disable or restrict it once frontend fallback is removed.
- Logs include session IDs and predictions; avoid logging raw trajectories/images.

## 12. Proposed Improvements

Priority:

1. Keep feature/model contract checks mandatory.
2. Add normalized error envelope for all AI endpoints.
3. Add model registry file/table and require model metadata at service load.
4. Add stage-level timing for validation, storage, feature extraction, inference, and DB.
5. Add database migrations for explicit `schema_version`, `feature_version`, and model registry tables.
6. Create a dedicated offline experiment pipeline with patient-level split.

## 13. Files Need Sửa / Đã Sửa

Changed in this pass:

- `app/services/fps_model_service.py`
- `app/services/cover_test/session_service.py`
- `app/api/cover_test_session.py`
- `tests/feature_parity/*`
- `tests/model/test_fps_model_contract.py`
- `tests/timeseries/test_session_timestamp_validation.py`
- Documentation files at repo root.

## 14. Files Không Nên Sửa

- `D:\AI_Check_Lac\*`: frontend reference only for this task.
- Production model binaries under `app/models/*.joblib` and `app/models/*.onnx`: do not alter without retraining/validation/registry update. Korean transfer `.joblib` artifacts were deleted because they were no longer useful for Cover Test runtime.
- Historical normalized datasets under `data/normalized/*`: use read-only for audit/training provenance unless doing explicit data curation.
