# RemiCare Strabismus AI Screening Backend

FastAPI Python backend for **RemiCare Strabismus Screening** — an AI-assisted webcam screening pipeline evaluating refixation movement time-series captured during the Cover Test.

> **CRITICAL CLINICAL NOTICE:**  
> This system is designed solely as a **preliminary screening tool** and **NOT a medical or clinical diagnosis system**.  
> It does **NOT** measure strabismus angles in prism diopters and does **NOT** replace an examination by an ophthalmologist or certified orthoptist.

---

## 1. Architecture

```
React Frontend (MediaPipe Tracking)
       ↓  POST /api/v1/screening/analyze (JSON)
FastAPI Backend
       ↓
Data Integrity Gate (validation.py)
   ├── PASS   ──→ Preprocessing (preprocessing.py)
   └── FAIL   ──→ Return SCREENING_INCONCLUSIVE
                      ↓
           Feature Extraction (feature_extraction.py)
              ├── Cycle-by-cycle kinematics
              ├── Inter-cycle consistency
              └── Inter-ocular relative displacement
                      ↓
           ML Inference Abstraction (inference.py)
              ├── No model artifact: SCREENING_INCONCLUSIVE (safe)
              └── Trained model: SCREENING_NORMAL / SCREENING_ATTENTION
                      ↓
           Screening Response (schemas.py)
                      ↓
               React Frontend
```

---

## 2. Directory Structure

```
remicare-strabismus-ai/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI initialization, CORS, health endpoint
│   ├── schemas.py               # Pydantic models (request, response, telemetry)
│   ├── api/
│   │   ├── __init__.py
│   │   └── screening.py         # POST /api/v1/screening/analyze
│   ├── services/
│   │   ├── __init__.py
│   │   ├── validation.py        # Data Integrity Gate
│   │   ├── preprocessing.py     # Median baseline, relative coords, signed displacement
│   │   ├── feature_extraction.py# Cycle kinematics & inter-cycle consistency
│   │   ├── inference.py         # ML model abstraction (joblib / scikit-learn)
│   │   └── screening.py         # End-to-end pipeline orchestrator
│   └── models/
│       └── .gitkeep             # Trained ML model weights (.joblib)
├── data/
│   ├── raw/                     # Raw Cover Test JSON time-series (sample.json)
│   ├── processed/               # Preprocessed tabular feature datasets
│   └── labels/                  # Independent clinical ground truth labels (labels.csv)
├── scripts/
│   └── test_real_sample.py      # Test script reading sample.json and hitting API
├── tests/
│   ├── __init__.py
│   └── test_screening.py        # 15 unit tests covering all validation and feature cases
├── requirements.txt
├── .gitignore
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

## 4. Running the Backend

Start the development server with hot-reload:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The interactive documentation will be available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 5. API Reference

### 5.1 Health Check
- **Endpoint:** `GET /health`
- **Response:**
```json
{
  "status": "ok",
  "service": "remicare-strabismus-ai",
  "version": "0.1.0"
}
```

### 5.2 Analyze Cover Test Time-Series
- **Endpoint:** `POST /api/v1/screening/analyze`
- **Request Body (Frontend Compatible):**
```json
{
  "sampleId": "remicare-demo-sample-001",
  "test": "COVER_TEST",
  "cycles": [
    {
      "cycle": 1,
      "coveredEye": "LEFT",
      "trackedEye": "RIGHT",
      "samples": [
        {
          "index": 0,
          "t": 33.3,
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

- **Response:**
```json
{
  "sampleId": "remicare-demo-sample-001",
  "status": "SCREENING_INCONCLUSIVE",
  "modelVersion": null,
  "quality": {
    "status": "PASS",
    "reason": null,
    "sampleCount": 33,
    "validSampleCount": 33,
    "validRatio": 1.0,
    "meanTrackingQuality": 0.9448,
    "issues": []
  },
  "analysis": {
    "cyclesAnalyzed": 3,
    "cycleFeatures": [...],
    "aggregatedFeatures": {...},
    "consistency": {...}
  },
  "reason": "MODEL_NOT_YET_TRAINED_AWAITING_CLINICAL_GROUND_TRUTH",
  "notice": "Screening result only — not a diagnosis."
}
```

---

## 6. Running Tests

Run the full pytest suite (15 unit tests covering data integrity, edge cases, feature extraction, and API):
```bash
pytest -v
```

---

## 7. Testing with Real Sample Data

Execute the test script to send `data/raw/sample.json` into the API:
```bash
python scripts/test_real_sample.py
```
*(If the live server is running on port 8000, it sends an HTTP POST; otherwise, it executes through FastAPI's in-memory TestClient.)*

---

## 8. Clinical Ground Truth & ML Protocol

1. **Labels Reference:** `data/labels/labels.csv` is prepared for independent clinical gold-standard labels (`sampleId,label`).
2. **Valid Classes:** `NORMAL`, `EXOTROPIA`, `ESOTROPIA`, `VERTICAL_STRABISMUS`, `OTHER`.
3. **Data Splitting:** Training/test splits MUST be grouped by participant to prevent data leakage.
4. **No Synthetic Predictions:** The inference layer strictly outputs `SCREENING_INCONCLUSIVE` until a verified model trained against real clinical labels is placed in `app/models/`.
