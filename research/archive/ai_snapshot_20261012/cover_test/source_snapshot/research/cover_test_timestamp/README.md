# Timestamp 30-60 FPS research

Isolated from production. Source data are binocular infrared eye tracking, not
annotated Cover Test cover/uncover cycles. This pipeline fits a phase-agnostic
NORMAL/STRABISMUS research classifier when documented source-specific permission
is available. It cannot learn post-uncover refixation from UNKNOWN phase labels.

Run from the repository root with the existing .venv312 Python:

```powershell
.venv312/Scripts/python.exe -m research.cover_test_timestamp.train --output reports/cover_timestamp_30_60/audit_segments.json
.venv312/Scripts/python.exe -m pytest tests/model/test_cover_timestamp_research.py -q
```

Training after a reviewer verifies the upstream data license or written reuse
permission and consent: fill permission.example.json in a separate permission
file and reference the actual evidence document. The example deliberately does
not authorize training. Use a new output directory for each run.

```powershell
.venv312/Scripts/python.exe -m research.cover_test_timestamp.train --train --permission research/cover_test_timestamp/permission.verified.json --output models/research/cover_timestamp_30_60/run_001
```

No patient IDs are inferred from filenames or hashes. Without a verified source
patient mapping, the full-data research fit has no accuracy/generalization
estimate. The original train/test directory split was previously found to have
patient overlap; reusing it is not an independent evaluation.

Optional --patient-mapping expects JSON with identity_basis set to
verified_source_patient_mapping, verified_by, and a recordings object mapping
paths relative to --data to real source patient codes supplied by the custodian.
This enables grouped out-of-fold evaluation. Train-side 30/45/60 Hz variants
stay with their original recording and patient; scaler fitting uses each fold's
training data. Reports count recordings and verified patients separately, show
all rejected variants, and use patient bootstrap 95% intervals. These intervals
describe the Korean source only and are not clinical accuracy on RemiCare.

Native t is explicitly in seconds. Downsampling keeps the nearest original
observation and timestamp. No upsampling, synthetic phase tags, missing-point
zeros, or interpolation across missing frames is performed. Velocity uses actual
timestamp differences; gaps >100ms break derivatives. Clock resets split a
recording into monotonic segments with original timestamps untouched, retaining
the same recording owner and label. These boundaries are not known Cover Test
phases or verified MEDIA_ID transitions; their provenance needs the raw logs.
Sampling checks use
27-65 Hz tolerance, >=2 seconds, >=50% jointly valid frames, >=30 jointly valid
observations and >=15 consecutive pairs. Undefined correlations cause rejection.
These are engineering quality gates, not clinical thresholds.

The 20 features summarize centered movement, change in binocular disparity,
timestamp-derived speed and motion correlation. Absolute viewport position,
recording length, filename and rate are excluded as predictors. Coordinate
translation invariance does not establish webcam/infrared equivalence. Cover Test
occlusion can reduce jointly valid observations; the rejection rate must be
assessed on actual Cover Test data before adapting this to the application.

The fixed candidate is standardized balanced Logistic Regression, C=0.1,
seed=20261005. Each original recording gets equal total weight regardless of how
many rate variants pass. Probabilities are uncalibrated. No architecture or
threshold is selected from evaluation results. Model artifacts carry the source,
timestamp contract, source hashes, software version and permission evidence hash.
predict_recording rejects source mismatch rather than claiming webcam validation.

Current source rights: see LICENSE_LEDGER.md. Current local audit: see
reports/cover_timestamp_30_60/audit_segments.json. The earlier audit.json preserves
the initial whole-recording rejection findings. Existing production loading remains
independent of this research directory.
