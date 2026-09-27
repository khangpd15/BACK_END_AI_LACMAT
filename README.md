# RemiCare Strabismus AI Screening Backend

FastAPI Python backend for **RemiCare Strabismus Screening** — an AI-assisted webcam screening pipeline evaluating refixation movement time-series captured during the Cover Test.

> **CRITICAL SCIENTIFIC & CLINICAL NOTICE:**  
> This system is designed solely as a **RESEARCH TRANSFER EXPERIMENT & PRELIMINARY SCREENING TOOL**, and **NOT A MEDICAL DIAGNOSIS**.  
> - The output is **NOT** a clinical diagnosis, disease probability, or risk score.
> - It does **NOT** measure strabismus angles in prism diopters.
> - The Korean Random Forest model was trained on infrared eye-tracker data (~60 Hz) and applied to webcam MediaPipe sampling (~15 Hz stored). **Domain shift warning is permanently active (`domainShiftWarning: true`, `clinicalMeaning: null`)**.
> - It does **NOT** replace an examination by an ophthalmologist or certified orthoptist.

---

## 1. Architecture

```
AI_CHECK_LAC (Vercel / React + Vite)
        │
        │ HTTPS POST JSON (Raw 15 Hz Cover Test time-series)
        ▼
FastAPI AI Backend (remicare-strabismus-ai)
        │
        ├── 1. Data Integrity Gate (validation.py)
        │      └── Validates monotonic time, valid eyes, finite coords [0, 1]
        │
        ├── 2. Time-series Preprocessing (preprocessing.py)
        │      └── Computes coordinate deltas, eye velocities, irises relative motion
        │
        ├── 3. Canonical Feature Extraction (korean_transfer.py / feature_extraction.py)
        │      └── Extracts EXACTLY 30 shared features in strict canonical FEATURE_ORDER
        │      └── Excludes hardware-dependent protocol features (sampleCount, duration, Hz)
        │
        ├── 4. Korean Shared Model Inference (models/korean_shared_model.joblib)
        │      └── Preloaded once at startup (Random Forest Classifier)
        │      └── Outputs prediction ('NORMAL' / 'STRABISMUS') and classProbability
        │
        ▼
Transfer Experiment Response
        │
        ▼
AI_CHECK_LAC Frontend (Displays research transfer result with domain shift warning)
```

---

## 2. Directory Structure

```
remicare-strabismus-ai/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI initialization, CORS, lifespan preload, routes
│   ├── config.py                # Environment configuration (origins, model paths)
│   ├── schemas.py               # Pydantic v2 schemas (strict validation, transfer models)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── screening.py         # POST /api/v1/screening/analyze
│   │   └── transfer.py          # POST /api/v1/transfer/strabismus (Phase 4.2 Endpoint)
│   └── services/
│       ├── __init__.py
│       ├── validation.py        # Data Integrity Gate
│       ├── preprocessing.py     # Median baseline, relative coords, signed displacement
│       ├── feature_extraction.py# Cycle kinematics & inter-cycle consistency
│       ├── korean_adapter.py    # Korean dataset reader & binary adapter
│       ├── korean_transfer.py   # Canonical 30-feature extractor & Random Forest loader
│       └── screening.py         # Clinical screening pipeline orchestrator
├── data/
│   ├── raw/                     # Raw Cover Test JSON time-series (sample.json)
│   ├── processed/               # Preprocessed tabular feature datasets & training CSVs
│   └── normalized/              # Standardized Korean dataset records
├── models/
│   └── korean_shared_model.joblib # Pretrained Korean Random Forest model (30 shared features)
├── scripts/
│   ├── run_frontend_tests.mjs   # Node test runner for AI backend frontend tests
│   ├── test_live_frontend_integration.mjs # E2E test from frontend service to live backend
│   └── test_real_sample.py      # Real sample test script
├── tests/
│   ├── test_feature_contract.py # Contract and schema invariants
│   ├── test_korean_adapter.py   # Dataset ingestion & class mapping
│   ├── test_screening.py        # Screening pipeline validation
│   ├── test_shared_model.py     # 30-feature model training and evaluation
│   └── test_transfer_api.py     # 20 transfer API validation & security unit tests
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Environment Variables

### Backend Configuration (`.env` or server environment)
| Variable | Default | Purpose |
|---|---|---|
| `REMICARE_FRONTEND_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated list of allowed CORS origins. Never use `*` in production. |
| `MODEL_PATH` | `models/korean_shared_model.joblib` | Filesystem path to the trained Random Forest artifact. |
| `ENVIRONMENT` | `development` | Environment mode (`development` or `production`). |

### Frontend Configuration (`AI_CHECK_LAC/.env.local` or Vercel Environment Variables)
| Variable | Local Development | Vercel Production |
|---|---|---|
| `VITE_AI_BACKEND_URL` | `http://localhost:8000` | `https://your-ai-backend-domain.com` |

---

## 5. Running the Backend

Start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 6. API Reference

### 6.1 System Health Check
- **Method:** `GET`
- **Path:** `/health`
- **Response:**
```json
{
  "status": "ok",
  "service": "remicare-strabismus-ai",
  "version": "0.1.0"
}
```

### 6.2 AI Transfer Inference
- **Method:** `POST`
- **Path:** `/api/v1/transfer/strabismus`
- **Request Body (Raw Observation Data Only - No Target Labels):**
```json
{
  "schemaVersion": "1.0.0",
  "sampleId": "ef59ac79-0d66-4990-9405-cf4a99c0d292",
  "test": "COVER_TEST",
  "source": {
    "device": "WEBCAM",
    "tracker": "MEDIAPIPE_IRIS"
  },
  "cycles": [
    {
      "cycle": 1,
      "coveredEye": "LEFT",
      "trackedEye": "RIGHT",
      "samples": [
        {
          "index": 0,
          "t": 0,
          "phase": "BASELINE",
          "leftX": 0.4441,
          "leftY": 0.6076,
          "leftValid": true,
          "rightX": 0.5756,
          "rightY": 0.5961,
          "rightValid": true,
          "trackingQuality": 0.95
        }
      ]
    }
  ]
}
```

- **Successful Response (`200 OK`):**
```json
{
  "sampleId": "ef59ac79-0d66-4990-9405-cf4a99c0d292",
  "status": "TRANSFER_EXPERIMENT",
  "inputCompatible": true,
  "prediction": "STRABISMUS",
  "classProbability": {
    "NORMAL": 0.2777,
    "STRABISMUS": 0.7223
  },
  "domainShiftWarning": true,
  "clinicalMeaning": null,
  "model": {
    "name": "Random Forest",
    "version": "korean-shared-v1.0.0"
  },
  "domainShift": {
    "source": "KOREAN_INFRARED_EYE_TRACKER",
    "target": "REMICARE_WEBCAM_MEDIAPIPE",
    "warning": true,
    "potentialShiftFeatures": [
      "meanLeftX",
      "meanLeftY",
      "meanRightX",
      "meanRightY"
    ]
  },
  "features": {
    "leftValidRatio": 0.8788,
    "rightValidRatio": 0.9394,
    "bothValidRatio": 0.8182,
    "meanDeltaX": 0.13433,
    "medianDeltaX": 0.1319,
    "stdDeltaX": 0.003296,
    "minDeltaX": 0.1315,
    "maxDeltaX": 0.1403,
    "rangeDeltaX": 0.0088,
    "meanAbsDeltaX": 0.13433,
    "meanDeltaY": -0.011289,
    "medianDeltaY": -0.0115,
    "stdDeltaY": 0.000544,
    "minDeltaY": -0.0121,
    "maxDeltaY": -0.0101,
    "rangeDeltaY": 0.002,
    "meanAbsDeltaY": 0.011289,
    "meanLeftX": 0.44329,
    "stdLeftX": 0.00215,
    "meanLeftY": 0.607641,
    "stdLeftY": 0.000187,
    "meanRightX": 0.577316,
    "stdRightX": 0.002936,
    "meanRightY": 0.596319,
    "stdRightY": 0.000442,
    "meanLeftVelocity": 0.022243,
    "peakLeftVelocity": 0.18058,
    "meanRightVelocity": 0.040084,
    "peakRightVelocity": 0.195541,
    "velocityDisparity": 0.01784
  },
  "notice": "Research transfer experiment only — not a diagnosis."
}
```

- **Validation Error (`422 Unprocessable Content`):**
Returned if non-monotonic timestamps, non-finite coordinates, invalid eye configurations, or empty cycles are received.

---

## 7. Canonical 30 Feature Order

The Korean Random Forest model requires EXACTLY these 30 features in this deterministic order:

| Index | Feature Name | Description |
|---|---|---|
| 1 | `leftValidRatio` | Ratio of valid left eye tracking frames |
| 2 | `rightValidRatio` | Ratio of valid right eye tracking frames |
| 3 | `bothValidRatio` | Ratio of frames where both eyes are tracked |
| 4 | `meanDeltaX` | Mean horizontal iris disparity ($X_{right} - X_{left}$) |
| 5 | `medianDeltaX` | Median horizontal iris disparity |
| 6 | `stdDeltaX` | Standard deviation of horizontal disparity |
| 7 | `minDeltaX` | Minimum horizontal disparity |
| 8 | `maxDeltaX` | Maximum horizontal disparity |
| 9 | `rangeDeltaX` | Horizontal disparity range |
| 10 | `meanAbsDeltaX` | Mean absolute horizontal disparity |
| 11 | `meanDeltaY` | Mean vertical iris disparity ($Y_{right} - Y_{left}$) |
| 12 | `medianDeltaY` | Median vertical iris disparity |
| 13 | `stdDeltaY` | Standard deviation of vertical disparity |
| 14 | `minDeltaY` | Minimum vertical disparity |
| 15 | `maxDeltaY` | Maximum vertical disparity |
| 16 | `rangeDeltaY` | Vertical disparity range |
| 17 | `meanAbsDeltaY` | Mean absolute vertical disparity |
| 18 | `meanLeftX` | Mean left eye X position |
| 19 | `stdLeftX` | Standard deviation of left eye X position |
| 20 | `meanLeftY` | Mean left eye Y position |
| 21 | `stdLeftY` | Standard deviation of left eye Y position |
| 22 | `meanRightX` | Mean right eye X position |
| 23 | `stdRightX` | Standard deviation of right eye X position |
| 24 | `meanRightY` | Mean right eye Y position |
| 25 | `stdRightY` | Standard deviation of right eye Y position |
| 26 | `meanLeftVelocity` | Mean tangential velocity of left iris |
| 27 | `peakLeftVelocity` | Peak tangential velocity of left iris |
| 28 | `meanRightVelocity` | Mean tangential velocity of right iris |
| 29 | `peakRightVelocity` | Peak tangential velocity of right iris |
| 30 | `velocityDisparity` | Disparity between left and right mean velocities |

**Excluded Features:** `sampleCount`, `durationMs`, `estimatedHz`, `meanIntervalMs` are strictly omitted due to hardware protocol dependencies.

---

## 8. Verification & Test Execution

### Backend Pytest Suite (62/62 PASS)
```bash
pytest -v
```
Includes:
- Health check endpoint
- Request validation (missing sampleId, invalid cycle, invalid eyes, coveredEye == trackedEye, non-monotonic timestamps, NaN/Inf rejection, tracking quality)
- Feature contract & exact 30-feature canonical order
- Anti-leakage checks (target labels blocked)
- Model inference & probability bounds
- Domain shift & clinical null assertions

### Frontend AI Service Test Suite (10/10 PASS)
```bash
node scripts/run_frontend_tests.mjs
```
Includes:
- Request building without target leakage
- Backend URL resolution from `VITE_AI_BACKEND_URL`
- POST method and JSON headers
- Successful response parsing
- Error state mappings (`INPUT_INCOMPATIBLE`, `BACKEND_UNAVAILABLE`, `BACKEND_TIMEOUT`)
- Single-POST completion guard
- UI contract data integrity

### Live End-to-End Integration Test
```bash
node scripts/test_live_frontend_integration.mjs
```

---

## 9. Deployment Guidelines

### Local Development
- **Frontend (`AI_CHECK_LAC`):** Run `npm run dev` (served on `http://localhost:5173`). Uses `.env.local` with `VITE_AI_BACKEND_URL=http://localhost:8000`.
- **Backend (`remicare-strabismus-ai`):** Run `uvicorn app.main:app --port 8000`. `REMICARE_FRONTEND_ORIGINS` includes `http://localhost:5173`.

### Production Deployment
- **Frontend:** Deployed to **Vercel**. Set environment variable `VITE_AI_BACKEND_URL=https://your-ai-backend.com`.
  - No Python code, model files, or datasets are committed or shipped to Vercel.
- **Backend:** Deployed as an independent container/service (e.g. AWS, Render, Fly.io, DigitalOcean) with HTTPS.
  - Set `REMICARE_FRONTEND_ORIGINS=https://your-vercel-domain.vercel.app`.
  - CORS strictly blocks unauthorized origins (wildcards prohibited in production).
#   B A C K _ E N D _ A I _ L A C M A T  
 