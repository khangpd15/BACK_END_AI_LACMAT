# Data Governance

## Data Stored

Current primary Cover Test storage:

- Session metadata.
- Cycle metadata.
- Raw numeric eye trajectory JSON.
- AI result metadata and feature snapshots.
- Model name/version and feature schema version.

Current image endpoint:

- Screening metadata only.
- No raw image bytes are persisted by the endpoint.

## Data Not Stored By Default

- Full face images.
- Eye crop images in the current primary Cover Test session path.
- Clinical labels or ground truth in production inference payloads.
- Personal identifiers such as name, phone, email, address, CCCD/CMND/SSN.

## Why Stored

- Session/cycle metadata: auditability and operational traceability.
- Raw numeric trajectories: future research review and reproducible feature extraction.
- Model result metadata: model monitoring and debugging.

## Access Control

Use Supabase/PostgreSQL role-based access. Service role keys must only exist in backend server environment variables, never in frontend code.

## Retention

Recommended default:

- Production screening metadata: 90-180 days unless user consent says otherwise.
- Raw numeric trajectories: shorter retention or research-only retention with explicit consent.
- Training/validation datasets: separate governed storage, not automatically populated from production.

## Deletion Policy

Every session should be deletable by `session_id` across:

- `cover_test_sessions`
- `cover_test_cycles`
- `cover_test_results`
- Supabase Storage raw JSON prefix

Add an audited deletion endpoint or admin job before broad production capture.

## Training Use

Production data must not be used for training by default. Training capture requires explicit consent, ground truth workflow, de-identification, and patient-level split.
