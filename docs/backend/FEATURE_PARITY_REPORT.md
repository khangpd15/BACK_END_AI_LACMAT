# Feature Parity Report

Date: 2026-10-03

## Scope

Added automated JS/Python parity fixtures under:

- `tests/feature_parity/fixture_cover_test_samples.json`
- `tests/feature_parity/js_shared_feature_extractor.mjs`
- `tests/feature_parity/test_shared_feature_parity.py`

The parity test uses the same raw Cover Test sample fixture and compares:

- JavaScript reference extractor in the backend test folder.
- Python production extractor `extract_shared_features_vector`.

Tolerance: absolute tolerance `<= 1e-6`.

## Current Result

Executed with Codex bundled Python 3.12 plus repo `venv\remicare-deps` and `.venv\Lib\site-packages` on `PYTHONPATH`.

```text
pytest tests\feature_parity tests\model tests\timeseries -q
3 passed, 1 skipped in 0.81s
```

Passed:

- JS/Python shared feature parity fixture.
- 10-15 FPS model contract rejection for missing required feature.
- No-model 10-15 FPS inference returns `INCONCLUSIVE`.

Skipped:

- Timestamp service test skipped because this temporary runtime lacks a compatible `greenlet` binary for SQLAlchemy asyncio. It should run once the repo Python environment is repaired.

Additional syntax verification was run with bundled Python:

```text
python -m py_compile app/services/fps_model_service.py app/services/korean_transfer.py app/services/cover_test/session_service.py app/api/cover_test_session.py tests/...
```

## Expected Command

```bash
pytest tests/feature_parity tests/model tests/timeseries -q
```

If Node is not in PATH, set:

```bash
NODE_BINARY="C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
```

## Contract Risks Found

- Before this pass, `fps_model_service.py` replaced missing/non-finite required features with `0.0`. This has been fixed.
- Before this pass, missing 10-15 FPS model fell back to a NORMAL-like 0.5/0.5 result. This now returns `INCONCLUSIVE`.
- Duplicate timestamps could pass through the session storage validation path. This has been fixed.
