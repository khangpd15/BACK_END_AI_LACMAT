# Timestamp Training Status

Date: 2026-10-05. Branch: codex/cover-test-timestamp-30-60.

## Completed

- Implemented timestamp-seconds feature extraction and a fixed Logistic
  Regression training candidate in research/cover_test_timestamp/.
- Inspected 41 local Korean recordings: 20 NORMAL, 21 STRABISMUS, 62,404 samples.
- Found 38 clock resets across 38 recordings; 3 recordings were monotonic.
  Explicit segmentation yielded 79 monotonic segments. Source timestamps and
  labels are unchanged; no patient IDs or Cover Test phases were inferred.
- Observed median sampling rates across segments: minimum 61.1247 Hz,
  median 61.8429 Hz, maximum 62.3441 Hz. There are 5 gaps longer than 100ms.
- Feature extraction passes for all 41 recordings, with 237 segment/rate
  variants at 30, 45 and 60 Hz, and 0 rejected variants. This is a preprocessing
  audit, not model fitting or an accuracy result. Variants are not independent
  patients. Evidence: audit_segments.json.
- Synthetic training, grouped OOF, serialization, timestamp derivatives,
  missing observations, clock resets and source mismatch rejection were tested.
  Focused regression command returned 18 passed:

```powershell
.venv312/Scripts/python.exe -m pytest tests/model/test_cover_timestamp_research.py tests/model/test_fps_model_contract.py tests/timeseries/test_session_timestamp_validation.py -q
```

## Not Completed

No Korean-data model was fitted or deployed. The source-specific data license and
consent/reuse permission remain unverified. The source repository root and README
were opened on 2026-10-05; they do not establish data reuse rights. A request to
fetch LICENSE failed; no absence-of-license claim is made. See the license ledger.

The current 41 files match the old test-set conversion report. The former 297
training CSV files are absent from the local data/korean_repo directory. No new
upstream data was downloaded.

All 62,404 sample phase values are UNKNOWN. Training on these trajectories can
learn source eye-tracking motion patterns, but cannot teach a Cover Test model
which motion occurred after uncovering. Native Cover Test phase/event timestamps
and RemiCare labeled trajectories remain necessary to evaluate that behavior.

No verified patient mapping exists. A future permitted full-data research fit
will therefore report no accuracy until a custodian supplies verified grouping.
The old session split is not reused: docs/training_audit_before_transfer.md
previously documented patient overlap in that split.

Training entry point, permissions evidence format and grouped evaluation format:
research/cover_test_timestamp/README.md. Existing source data, models and
production runtime files were not changed.
