# Database Schema

Current schema is defined by SQLAlchemy models and SQL migrations.

## Current Tables

### `cover_test_sessions`

Session-level metadata:

- `session_id`
- `created_at`
- `completed_at`
- `cycle_count`
- `sampling_rate_hz`
- `source_device`
- `tracker`
- `raw_schema_version`
- `storage_root`
- `processing_status`
- `error_message`
- `client_metadata`
- `updated_at`

### `cover_test_cycles`

Cycle-level metadata:

- `cycle_id`
- `session_id`
- `cycle_number`
- `covered_eye`
- `tracked_eye`
- `sample_count`
- `valid_sample_count`
- `valid_ratio`
- `mean_tracking_quality`
- `cycle_status`
- `raw_storage_path`
- `duration_ms`
- `created_at`

### `cover_test_results`

Model output metadata:

- `result_id`
- `session_id`
- `model_name`
- `model_version`
- `model_source`
- `feature_schema_version`
- `status`
- `input_compatible`
- `prediction`
- `class_probabilities`
- `confidence`
- `domain_shift_warning`
- `features_snapshot`
- `comparison_models`
- `notice`
- `created_at`

### `strabismus_screenings`

Image ROI screening metadata:

- `id`
- `created_at`
- `status`
- `strabismus_probability`
- `confidence`
- `quality_score`
- `threshold`
- `model_version`
- `inference_latency_ms`
- `image_saved`
- `failure_reason`

## Recommended Next Tables

- `model_registry`
- `model_inference_logs`
- `screening_quality`
- `session_timeseries` only if raw object storage is insufficient for analytics.

Keep large raw time-series and images in object storage, not large JSON blobs in a single relational row.
