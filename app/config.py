"""Application Configuration for RemiCare Strabismus AI Backend."""

import os
from pathlib import Path
from typing import List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment Variables
APP_ENV = os.getenv("APP_ENV", "development").lower()
APP_NAME = "remicare-strabismus-ai"
API_VERSION = "v1"

# Model Configuration
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "korean_shared_model.joblib")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
FEATURE_CONTRACT_VERSION = "shared-v1.0.0"

# CORS Configuration
# Comma-separated list of allowed frontend origins
# In production, set REMICARE_FRONTEND_ORIGINS=https://YOUR-VERCEL-DOMAIN.vercel.app
DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000"
)
FRONTEND_ORIGINS_RAW = os.getenv("REMICARE_FRONTEND_ORIGINS", DEFAULT_DEV_ORIGINS)


def get_allowed_origins() -> List[str]:
    """Parse comma-separated origin string into a clean list of allowed CORS origins."""
    origins = [o.strip() for o in FRONTEND_ORIGINS_RAW.split(",") if o.strip()]
    return origins
