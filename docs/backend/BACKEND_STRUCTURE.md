# Backend directory structure

## Runtime

`app/main.py` is the FastAPI entrypoint. `app/api/` defines routes,
`app/schemas/` validates requests and responses, `app/services/` implements
inference and storage, and `app/db/` owns database models and repositories.

All shared CV code lives in `app/services/cv/`:

- `bilateral_quality_gate.py`: image quality and bilateral presence checks for ONNX.
- `quality_gate.py`: frame quality metrics for temporal processing.
- `temporal_aggregator.py`: consensus over valid frames.

Model artifacts stay at their existing paths in `app/models/`.
`app/hirschberg_runtime/` is a frozen inference bundle, including source modules,
checkpoints, metadata and a SHA-256 manifest. Its nested `research/` and `scripts/`
folders are required at runtime; their names do not mean they can be deleted.

## Development and research

- `scripts/`: offline dataset preparation, training, imports and integration checks.
  The former root `filter_harschberg.py` now lives here; run it from repository root.
- `research/`: experiment source and preserved historical snapshots under `archive/`.
- `tests/`: feature parity, model, research and timestamp checks.
- `migrations/`: SQL schema changes.
- `docs/`, `audits/`: current documentation and historical evidence.
- `data/cover_test/sessions/`: runtime session storage for the legacy API;
  session contents are excluded from Git.
- `reports/production_integration/`: retained deployment verification evidence.

Training datasets and research outputs were removed from this backend at the
user's request after confirming training runs in separate projects:
`D:/REMICARE_TRAIN_AI/COVER_TEST_METHOD` and
`D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD`. Offline scripts, research tests and
historical documents may reference the former dataset locations; they require
separately supplied data or should be run in the training projects.

The dataset cleanup removed 749 files (96,166,068 bytes, about 91.7 MiB) from
`data/normalized/`, `data/VIDEO_COVER_TEST/`, `data/korean_repo/`, `data/labels/`,
`datasets/`, `dataset_by_pedseye_manifest/`, `harschberg_data_detect/`,
`processed/` and research outputs under `reports/`. No inference code or model
artifact was removed. Deployment reports and runtime session storage remain.
After dataset removal, all 15 focused inference/API/timestamp tests passed again;
the Hirschberg SHA-256 manifest, ONNX artifact presence and retained runtime
directories were also verified.

## Cleanup on 12 October 2026

- Removed 822 files from the duplicate `ai/` workspace only after comparing
  SHA-256 with their canonical source files (197,111,162 bytes, about 188 MiB).
- Preserved seven differing snapshots in `research/archive/ai_snapshot_20261012/`.
- Removed obsolete workspace-copy metadata, READMEs and `scripts/copy_ai_workspaces.py`.
- Moved `cv/quality_gate.py` into `app/services/cv/bilateral_quality_gate.py`,
  updated imports and removed the redundant root `cv/` package.
- Moved the dataset filter into `scripts/` and replaced the stale root README.

Local dependency environments (`.venv/`, `.venv312/`, `venv/`) are retained:
they contain installed dependencies rather than duplicate backend source.
The supported runtime version is recorded in `.python-version` and `runtime.txt`.

## Verification

After cleanup, 15 focused runtime/model/timestamp tests and 16 feature-parity
and Cover Test research tests passed. Runtime tests used an in-memory SQLite
database and disabled keep-alive. The Windows sandbox blocked asyncio's internal
socket creation and pytest's default temporary directory; runtime tests were
rerun outside that sandbox and temporary files were placed inside the workspace.

The frozen Hirschberg bundle still matches its SHA-256 manifest. Moved source
files match their Git originals after newline normalization, and the seven
archived snapshots match the old copy inventory hashes. Python source syntax,
CV imports, the relocated filter's `--help` and `git diff --check` were verified.
The full test suite was interrupted when it encountered the sandbox limitation;
this cleanup does not claim a complete full-suite pass.
