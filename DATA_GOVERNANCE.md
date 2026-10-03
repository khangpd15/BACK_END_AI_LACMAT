# Data Governance

## Data Stored

Current primary Cover Test storage:

- Session metadata.
- Cycle metadata.
- Raw numeric eye trajectory JSON.
- AI result metadata and feature snapshots.
- Model name/version and feature schema version.

Research-only Phase 5 capture, when explicit guardian/research consent exists:

- Pseudonymous `participant_id`.
- Eligibility and consent metadata.
- Hirschberg original image path, quality metadata, iris/reflex measurements, lighting metadata, glasses metadata, device/camera metadata, and `distance_bucket`.
- Cover phase sequence, real timestamps, iris/canthus coordinates, visibility, quality, and raw landmarks when separate consent allows replay.
- Clinician/orthoptist ground-truth fields: horizontal strabismus type, intermittency, glasses group, pseudostrabismus note, exam method, exam date, and label source.

Current image endpoint:

- Screening metadata only.
- No raw image bytes are persisted by the endpoint.

## Data Not Stored By Default

- Full face images.
- Eye crop images in the current primary Cover Test session path.
- Clinical labels or ground truth in production inference payloads.
- Personal identifiers such as name, phone, email, address, CCCD/CMND/SSN.
- Research video or raw landmarks unless consent explicitly allows replay for algorithm versioning.
- Production screening sessions in training datasets by default.

## Why Stored

- Session/cycle metadata: auditability and operational traceability.
- Raw numeric trajectories: future research review and reproducible feature extraction.
- Model result metadata: model monitoring and debugging.
- Hirschberg/Cover research records: protocol validation, distance-bucket experiment reporting, and future reproducible training only after clinician labels are available.
- Ground truth: supplied only by doctor/orthoptist workflow; model output is never ground truth.

## Access Control

Use Supabase/PostgreSQL role-based access. Service role keys must only exist in backend server environment variables, never in frontend code.

Research datasets require a narrower access group than production operations. Direct download of original images, raw landmarks, or videos should be limited to approved research/admin roles and audited outside the application runtime.

## Retention

Recommended default:

- Production screening metadata: 90-180 days unless user consent says otherwise.
- Raw numeric trajectories: shorter retention or research-only retention with explicit consent.
- Training/validation datasets: separate governed storage, not automatically populated from production.

Research retention duration is not approved yet. Keep retention as `POLICY_PENDING` in manifests until a consent form and deletion policy are approved.

## Deletion Policy

Every session should be deletable by `session_id` across:

- `cover_test_sessions`
- `cover_test_cycles`
- `cover_test_results`
- Supabase Storage raw JSON prefix

Add an audited deletion endpoint or admin job before broad production capture.

Research deletion must cover:

- PostgreSQL rows.
- Object storage paths for Hirschberg images, Cover raw JSON, raw landmarks, and consented videos.
- Exported manifests and derived datasets.
- Training/evaluation artifacts. Do not promise deletion from an already-trained model unless a retraining/removal process exists.

## Training Use

Production data must not be used for training by default. Training capture requires explicit consent, ground truth workflow, de-identification, and patient-level split.

Additional training rules:

- Split by `participant_id`, never by frame/image/session.
- `legacy_non_hirschberg` data must not be used for Hirschberg threshold selection or training.
- Model predictions must not be used as labels.
- Only horizontal strabismus protocol data with doctor/orthoptist labels can become eligible for future training.
- Phase 5 scripts may create manifests, reports, and training plans, but must not train without explicit user approval after dataset review.
