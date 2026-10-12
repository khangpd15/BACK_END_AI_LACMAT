# Runtime data

Training datasets are maintained in the separate projects:

- `D:/REMICARE_TRAIN_AI/COVER_TEST_METHOD`
- `D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD`

The legacy `POST /api/cover-test/sessions` endpoint creates session directories
under `cover_test/sessions/` and writes `metadata.json` and `sampling.json`.
Session contents are excluded from Git. Do not delete real session data as part
of training-dataset cleanup.

The newer `POST /api/v1/cover-test/sessions` endpoint uses PostgreSQL and Supabase.
Inference artifacts stay in `app/models/` and `app/hirschberg_runtime/`.
