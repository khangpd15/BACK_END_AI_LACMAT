# Audit Hirschberg - RemiCare Strabismus AI

Date: 2026-10-03

Scope: Hirschberg only. No Cover Test training was performed. Original data was not edited; this audit only created files under `reports/`.

## 1. Codebase

The backend is Python FastAPI. Main app registration is in `app/main.py`. The existing upload endpoint for image inference is `POST /api/v1/strabismus/predict` in `app/api/strabismus.py`; it accepts `UploadFile` multipart image data, validates MIME/size, reads image bytes in memory, calls `StrabismusInferenceService.predict`, and persists metadata only.

Current image inference is not a Hirschberg-specific 3-class production API. `app/services/strabismus_inference_service.py` loads `app/models/best_model.onnx`, applies a bilateral-eye-ROI contract, a quality gate, and outputs binary `NORMAL` / `STRABISMUS` with a locked threshold 0.20.

Existing Hirschberg-related code exists:

- `filter_harschberg.py`: filters/copies Hirschberg crops into `harschberg_data_detect/`, detects corneal reflexes, records `report.csv`, and contains heuristic displacement thresholds. It prioritizes source labels from folder/file names when present; heuristic labels are used only when source labels are unknown.
- `app/services/research_measurement_service.py`: research-only Hirschberg geometry measurement, including iris/pupil/reflex helper functions.
- `app/services/hirschberg_ai_service.py`: loads a research candidate `hirschberg_candidate_v0.1.joblib` and `best_model.onnx`, extracts 73 features from crops, returns 3-class probabilities.
- `scripts/hirschberg_folder_manifest.py`, `scripts/benchmark_hirschberg_detectors.py`, `scripts/train_hirschberg_research_model.py`: prior exploratory manifest, detector benchmark, and sklearn training scripts.
- Tests exist under `tests/research/`.

Existing model artifacts:

- `app/models/best_model.onnx` (~44.97 MB): active bilateral ROI screening model.
- `app/models/best_model.pt` (~45.05 MB): PyTorch checkpoint present.
- `app/models/research/hirschberg_candidate_v0.1.joblib` (~1.30 MB): research 3-class Hirschberg candidate.
- `app/models/korean_shared_model.joblib`, `app/models/remicare_transfer_model.joblib`, `app/models/remicare_15fps_candidate.joblib`: Cover/Test transfer or time-series candidates, not Hirschberg-only.

ML framework in requirements: scikit-learn, joblib, OpenCV, MediaPipe, ONNX Runtime, Pillow, pandas/numpy. No PyTorch is listed in `requirements.txt`, although a `.pt` file exists. The repo `.venv` is currently broken because it points to missing `C:\Users\PC\AppData\Local\Programs\Python\Python311\python.exe`; `python` and `py -3` are not usable globally. `.venv/Lib/site-packages` contains installed packages including FastAPI 0.141.1, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, mediapipe 1.0.1, onnxruntime 1.30.0, pillow 12.3.0. This needs repair before training.

Hardware audit:

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM, CUDA UMD 13.3, driver 610.62, 0 MiB compute memory in use at audit time.
- CPU/RAM: Windows system queries via `Get-CimInstance` and `systeminfo` returned access denied in this sandbox, so exact CPU/RAM could not be verified.
- Practical implication: if PyTorch/CUDA is installed later, MobileNet/EfficientNet-B0 fine-tuning should fit at small batch sizes. With current usable runtime, only CPU-only analysis was possible.

## 2. Dataset Audit

Audited root: `harschberg_data_detect/`.

Trainable class folders:

| Class | Folder | Count | Format | Size | Resolution |
|---|---:|---:|---|---:|---|
| esotropia | `esotropia_harschberg/` | 55 | JPEG | 0.769 MB | all 224x224 |
| exotropia | `exotropia_harschberg/` | 40 | JPEG | 0.557 MB | all 224x224 |
| normal | `normal_harschberg/` | 41 | JPEG | 0.612 MB | all 224x224 |

Other folders:

| Folder | Count | Note |
|---|---:|---|
| `rejected/` | 366 | filtered out, mostly missing one/both reflexes |
| `debug/` | 20 | debug visualizations, not training data |

The current trainable dataset is only 136 images. Class imbalance is mild by count (max/min ratio 1.375), but total size is small.

Image quality/statistics:

| Class | Brightness mean min/median/max | Blur Laplacian variance min/median/max |
|---|---|---|
| esotropia | 61.794 / 148.693 / 228.343 | 32.506 / 199.391 / 1402.053 |
| exotropia | 87.875 / 132.209 / 196.864 | 14.865 / 162.075 / 719.184 |
| normal | 44.525 / 140.060 / 189.016 | 15.053 / 167.414 / 1332.695 |

Bias risk: all trainable images are square 224x224 ocular crops, not full face/phone flash uploads. Brightness and blur distributions overlap, but esotropia has the brightest maximum and higher median blur variance. The model may learn crop/source style, grayscale/color differences, or face framing rather than Hirschberg physiology.

Labels:

- Current accepted labels come from folder/file-name source labels (`esotropia`, `exotropia`, `normal`), not from a separate clinical CSV/JSON manifest in this folder.
- `filter_harschberg.py` records both `source_label` and `heuristic_label`.
- In `report.csv`, accepted source labels vs heuristic labels disagree often:
  - esotropia source: 24 heuristic esotropia, 18 heuristic exotropia, 13 heuristic normal.
  - exotropia source: 25 heuristic exotropia, 3 heuristic esotropia, 12 heuristic normal.
  - normal source: 11 heuristic normal, 13 heuristic esotropia, 17 heuristic exotropia.
- Warning: if labels are not independently clinician-confirmed, training is not reliable. If labels are clinician-confirmed, the detector heuristic still looks noisy and should not be treated as ground truth.

Duplicates:

- Exact duplicate SHA-256 groups: 0.
- Near-duplicate candidates by coarse 16x16 average hash: 2 pairs found. Both were same-label pairs:
  - `esotropia_055.jpg` vs `esotropia_060.jpg` (distance 0)
  - `normal_066.jpg` vs `normal_070.jpg` (distance 8)
- Participant IDs are absent, so patient-level leakage cannot be ruled out even if file hashes are unique.

Filter / detector summary from `harschberg_data_detect/report.csv`:

- Total source rows: 502.
- PASS: 136 (27.1%).
- REJECT: 366 (72.9%).
- Detection method: 500 `PERIOCULAR_CROP_HEURISTIC`, 2 `FACEMESH_LANDMARKS`.
- Common reject reasons: missing both reflexes due to low peak brightness (132), missing left/right reflex, or no compact reflex.

MediaPipe Face Mesh full run:

- Not rerun during this audit because the project venv is broken and the fallback bundled Python does not include MediaPipe/OpenCV.
- Existing `report.csv` indicates Face Mesh was usable on only 2/502 source images; almost all processing used crop heuristics. Since the dataset is 224x224 eye crops, this is expected and not representative of full-face phone uploads.

Manual sample review:

- Sample grids saved:
  - `reports/audit_samples/esotropia_random20_grid.jpg`
  - `reports/audit_samples/exotropia_random20_grid.jpg`
  - `reports/audit_samples/normal_random20_grid.jpg`
- Visual review: many images are cropped tightly around both eyes, variable in color/grayscale, some overexposed, some low-resolution or blurred, and several have gaze direction/framing differences. Corneal highlights are visible in many but not all; some normal samples visually resemble non-primary gaze or possible mislabels. These do not look like the product flow of a phone flash full-face photo at 30-50 cm.

## 3. Dataset vs Real Product Use

There is a major domain shift:

- Product input: full face or at least phone-captured face image with flash, upright, 30-50 cm distance.
- Dataset input: pre-cropped 224x224 bilateral eye crops, mixed sources, unknown capture device, unknown distance, unknown participant identity.
- Current production image service even rejects full-face images for the bilateral ROI model, while the requested product flow says users upload face photos.

This means a model trained directly on these crops will not be reliable for raw user uploads unless the backend first implements a robust full-face quality gate and eye-crop extraction path, then trains/evaluates on crops generated by the same path.

## 4. Severity-Ranked Findings

### Blocking for trustworthy clinical-style training

1. Only 136 trainable accepted images in the current filtered dataset.
2. No participant IDs, so leakage across train/val/test cannot be ruled out.
3. Dataset is 224x224 crop-only and does not match phone full-face upload use.
4. The project Python environment is broken; MediaPipe/CV training pipeline cannot be reproduced until venv is repaired.
5. Current folder labels lack a separate clinician manifest in this dataset; label provenance is not auditable from files alone.

### Needs handling before any model candidate

1. Detector/reflex filtering is strict and rejects 72.9% of source images.
2. Prior detector benchmark on a 696-crop manifest found low joint pupil+reflex coverage (4.45%), so geometric features depending on both detectors may be sparse.
3. Normal samples visually include off-gaze/framing variation; possible label noise needs clinician review.
4. Need a full-face-to-eye-crop preprocessing contract if product input remains phone face photos.

### Notes

1. Class counts among accepted images are not severely imbalanced.
2. Exact duplicate files were not found.
3. Existing research candidate metrics on the older 696-crop manifest should not be interpreted as current dataset readiness or production performance.

## 5. Conclusion

The current `harschberg_data_detect/` dataset is not sufficient to train a trustworthy production-grade Hirschberg screening model for real phone uploads. It is sufficient for a small internal research baseline only, with strong caveats: crop-only domain, missing participant groups, uncertain label provenance, and broken reproducibility environment.

Minimum additions before reliable training:

- Rebuild a working Python environment and rerun the MediaPipe/reflex audit.
- Add a label manifest with clinician-confirmed label, participant ID, capture protocol, device/source, and consent/provenance.
- Collect representative phone flash full-face images or generate eye crops using the exact backend preprocessing planned for production.
- Clinician-review a sample of accepted and rejected images, especially normal and heuristic-disagreement cases.
