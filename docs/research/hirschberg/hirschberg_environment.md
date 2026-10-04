# Hirschberg Environment Repair

Date: 2026-10-03

Status: repaired for research tooling in `.venv312`.

Actions:

- Created `.venv312` with Python 3.12.14.
- Installed `requirements.txt`.
- Did not add PyTorch. CNN training is intentionally out of scope for this run.
- Pinned OpenCV to `<5.0.0`.
- Pinned MediaPipe to `0.10.14` because newer MediaPipe packages on Python 3.12 expose only `tasks` and do not provide the `mp.solutions.face_mesh` API used by this codebase.
- Added `.venv312/` to `.gitignore`.

Validated:

- `cv2.__version__ == 4.14.0`
- `mediapipe.__version__ == 0.10.14`
- `mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)` initializes successfully.
- `python -m pip check` reports no broken requirements.
- `python -m pytest tests/research/test_research_measurement_service.py tests/research/test_hirschberg_folder_manifest.py -q` passes: 13 passed.

Known note:

- Matplotlib warns that it cannot save a font cache under `C:\Users\PC\AppData\Local\matplotlib` in this sandbox. The research scripts avoid requiring that cache for core evaluation.
