# Hirschberg model imported on 12/10/2026

Backend uses `screening_eyelid_branch_20261011_v1` (ResNet50 + Hirschberg geometry + eye-opening features), pupil `pupil_color_features_20261012_v1/control_final.pt`, and the frozen iris, reflex, eyelid and ROI seed CNNs. Artifacts, normalization values and inference code live in `app/hirschberg_runtime/`; `manifest.json` records their SHA-256 hashes. Loading verifies every hash. Runtime does not need the training folder or download pretrained weights.

Install `requirements.txt` and run:

```powershell
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For CPU deployments, install PyTorch from its CPU wheel index:

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

`GET /api/v1/hirschberg/health` and `GET /health` report readiness, classifier and pupil versions/hashes. `POST /api/v1/hirschberg/predict` accepts multipart `image` (JPEG, PNG, WebP, up to 10 MB and 25 megapixels), with optional `kind=auto|fullface|bilateral`, `orientation=auto|unmirrored|mirrored`, `fixation=unknown|confirmed|offaxis`.

```powershell
curl.exe -F "image=@D:/path/to/photo.jpg" http://127.0.0.1:8000/api/v1/hirschberg/predict
```

Automatic mode keeps orientation/fixation unverified, as the source pipeline does. For confirmed capture, provide an explicit `kind` and verified orientation/fixation. QC runs before classification. Results are `INCONCLUSIVE` or `HUMAN_REVIEW`, with three possible class proposals: NORMAL, ESOTROPIA, EXOTROPIA. Raw scores appear in `uncalibratedModelScores`; `probabilities` stays null. Failed QC exposes no scores. Images and visualization files are not persisted by this API.

Existing `/api/v1/research/measurements` with `testType=HIRSCHBERG` automatically uses the new model in `aiPrediction`, including its CNN landmarks and QC. The surrounding measurement response preserves its existing geometry schema. `/api/v1/strabismus/predict` continues its separate bilateral ONNX contract.

The source reports that the updated pupil/classifier combination has not been independently evaluated. Training pupil mean error is not a clinical accuracy claim. Runtime preserves research-only metadata and does not produce a diagnosis or a pseudostrabismus class.

To reproduce this import from the original training workspace: `python scripts/import_trained_hirschberg.py`. Source assets remain unchanged. Frozen exact-image detector ROIs retain only image hashes, boxes and contours; class labels and human target points are not included.
