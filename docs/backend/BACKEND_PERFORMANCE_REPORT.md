# Backend Performance Report

Date: 2026-10-03

## Current Instrumentation

- HTTP middleware adds `Server-Timing: total;dur=...`.
- Logs method, path, status, total duration, and response size.
- Strabismus ROI inference returns `inference_latency_ms`.
- DB health endpoint returns DB latency for `SELECT 1`.

## Current Bottlenecks

- Startup model preload and DB init.
- Multipart raw JSON parsing and Supabase uploads.
- Optional synchronous inference during `/api/v1/cover-test/sessions`.
- Stage-level timing is not yet recorded for Cover Test validation/storage/inference separately.

## Recommended Timing Fields

For development/debug responses only:

```json
{
  "timing": {
    "validationMs": 0,
    "storageMs": 0,
    "featureMs": 0,
    "inferenceMs": 0,
    "databaseMs": 0,
    "totalMs": 0
  }
}
```

In production, log aggregated metrics instead of exposing internals.

## Measured In This Pass

Full latency benchmarks were not run because the local Python virtual environment is broken and no running DB/Supabase service was available in this execution context.

## Action Items

- Add per-stage timers in `CoverTestSessionService.persist_session`.
- Add model load timing at startup.
- Add inference window count and rejected-window count to logs.
- Keep model services singleton/cached.
- Avoid loading joblib/ONNX per request.
