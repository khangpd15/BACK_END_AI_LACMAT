# Frontend/backend synchronization audit — 2026-10-12

Frontend source: `D:/AI_Check_Lac`. Backend source: `D:/REMICARE-STRABISMUS-AI`.
Read-only audit; frontend files were not modified.

## Conclusion

Local frontend is not fully synchronized with the current backend. The deployed
Vercel bundle differs materially from the local source. Live backend contract and
CORS verification could not complete: the presumed Render URL returned 502/503
on the first probe and timed out after 25 seconds on the second probe. This does
not establish that the deployment failed; verify that this is the current URL
and inspect service logs/availability.

## Compatible contracts in source

- `POST /api/v1/cover-test/sessions?run_inference=true`: multipart names
  `session_metadata` and `cycle_1_raw` through `cycle_3_raw` match the backend.
  The response keys consumed by the persistence service also match.
- `POST /api/v1/research/measurements`: frontend JSON schema/feature version,
  eligibility, imageDataUrl and request identifiers match the backend schema.
  This endpoint calls the new trained Hirschberg CNN via `predict_hirschberg`.
  Frontend does not need to switch to the multipart `/api/v1/hirschberg/predict`
  merely to use the new CNN.
- Legacy storage and bilateral ONNX routes referenced by frontend still exist.

## Findings in local frontend

1. **Hirschberg classes are not fully mapped.** Backend returns NORMAL,
   ESOTROPIA, EXOTROPIA or INCONCLUSIVE. `HirschbergResultStep.jsx:38` only treats
   STRABISMUS/SUSPICIOUS as an attention result. ESOTROPIA/EXOTROPIA therefore
   fall through to the default insufficient-information text. The final result
   card has the same incomplete AI class mapping.
2. **Review/abstention status is discarded.** Backend classifier proposals have
   status HUMAN_REVIEW; quality abstentions have status INCONCLUSIVE.
   `HirschbergStep.jsx:135` sets COMPLETED whenever predictedClass is nonempty,
   including the literal INCONCLUSIVE. Preserve backend status and require an
   actual accepted class before declaring a prediction completed.
3. **Timeout is too short for startup warmup.** `HirschbergStep.jsx:97` sets
   25000 ms; `researchMeasurementApi.js:8` defaults to 20000 ms. CNN warmup runs
   in background and prediction waits on the loader lock. Check readiness and
   provide a loading state before inference; set a suitable request timeout.
4. **Retired transfer fallback remains reachable.** `CoverTestStep.jsx:247`,
   `:261`, `:276`, `:289` and retry UI call `analyzeCoverTest`, which posts to
   `/api/v1/transfer/strabismus`. This route is absent from current backend
   registration. It is reached when storage fails or aiResult is missing.
5. **Cover Test displays invented model metadata/probabilities when missing.**
   Backend currently returns aiResult INCONCLUSIVE, reason MODEL_NOT_LOADED,
   model.loaded false and classProbabilities null when no approved artifact is
   loaded. `CoverTestStep.jsx:1128` through `:1183` defaults to retired Korean
   model details and zero probabilities instead of rendering model unavailable.
6. **Storage messaging does not match behavior.** Frontend sends eye crops and
   says images were saved; backend intentionally does not read/store them and
   returns imagesSaved=0. Update the success text to numeric trajectory storage.
7. **Local environment points to localhost.** `.env.local:1` contains
   VITE_AI_BACKEND_URL=http://localhost:8000. This is valid for local development,
   but a production build without an explicit environment override inherits it.

## Deployed frontend

URL: https://aidetecteye.vercel.app
Asset: `/assets/index-CzrKnMC8.js`; HTTP 200 on both probes.

The asset contains the presumed Render URL, contains MODEL_NOT_LOADED handling,
and does not contain the old transfer endpoint or the 25-second timeout literal
checked by the audit. It does not contain `uncalibratedModelScores`. These are
bundle text checks, not proof of correct UI rendering or end-to-end inference.
Do not redeploy the older local source without reconciling these differences.

## Verification

`node scripts/runAllTests.js` in frontend: all 14 existing checks passed.
They cover camera transforms, prescreening, camera settings and error mapping;
they do not validate the current AI response contracts.

No real images were uploaded and no cloud sessions were written.
Raw HTTP evidence: `frontend_backend_sync_20261012.json` and
`frontend_backend_sync_20261012_recheck.json` beside this report.
Reusable read-only probe: `scripts/check_frontend_backend_sync.py`.
