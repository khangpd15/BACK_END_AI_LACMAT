# Hirschberg BE integration — 12/10/2026

Imported the active training pipeline into `app/hirschberg_runtime` and enabled it for the existing Hirschberg research measurement API and the new multipart `/api/v1/hirschberg/predict` API. The separate bilateral ONNX endpoint keeps its own model.

Active classifier: `screening_eyelid_branch_20261011_v1`. Active pupil: `pupil_color_features_20261012_v1/control_final.pt`. Eight checkpoint files are copied byte-for-byte; manifest hashes also cover supporting config, frozen detector ROI metadata and runtime Python modules. Source training files were not edited. Deployment needs only the backend files and its requirements.

Validation: 39 tests passed across the trained-runtime tests, production API runtime tests, Hirschberg research measurement v1/v2 tests and legacy ensemble v2 regression tests. The new tests cover real checkpoint loading, classifier forward execution, QC rejection without persisted images, corrupted-artifact rejection, existing inference routing and upload validation. Existing TestClient emits one dependency deprecation warning.

Live HTTP verification passed on three previously recorded real-image cases. All retained their reference `INCONCLUSIVE` status. Maximum pupil coordinate difference from the training runtime was 0.0013818443 pixels on CPU. This checks implementation parity, not clinical accuracy. Details and hashes are in `http_verification.json`; user image bytes are not copied into this report or saved by the API.

Backend verified running at `http://127.0.0.1:8000`; `/api/v1/hirschberg/health` reports `loaded: true`. PyTorch 2.7.1 CPU is installed in `.venv312`. Windows sandbox blocked asyncio socketpair creation during the initial tests; the successful tests and server ran with approved execution outside the sandbox.

The runtime preserves the original landmark/capture QC. Automatic mode leaves fixation and anatomical orientation unverified; explicit confirmed capture options are documented in `docs/backend/HIRSCHBERG_RUNTIME.md`. The updated pupil/classifier combination remains without independent evaluation. Classifier scores are uncalibrated, and no clinical diagnosis or pseudostrabismus class is returned.
