"""Application Configuration for RemiCare Strabismus AI Backend."""

import os
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment Variables
APP_ENV = os.getenv("APP_ENV", "development").lower()
APP_NAME = "remicare-strabismus-ai"
API_VERSION = "v1"

# Model Configuration
# Phase 5: remicare_transfer_model.joblib (domain-adapted) takes priority.
# Phase 4: korean_shared_model.joblib (Korean baseline fallback).
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "korean_shared_model.joblib")
REMICARE_TRANSFER_MODEL_PATH = str(PROJECT_ROOT / "app" / "models" / "remicare_transfer_model.joblib")
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

# Database & Storage Configuration (Phase 3)
DATABASE_URL = os.getenv("DATABASE_URL", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_STORAGE_BUCKET_RAW = os.getenv("SUPABASE_STORAGE_BUCKET_RAW", "cover-test-raw")
ENABLE_AI_INFERENCE_ON_SAVE = os.getenv("ENABLE_AI_INFERENCE_ON_SAVE", "true").lower() in ("true", "1", "yes")

# Database Connection Pool Tuning
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "300"))  # Recycle every 5m to prevent Supabase idle drop
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# Keep-Alive & Self-Ping Configuration (Prevents Render spin-down & Supabase idle timeout)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()
SELF_PING_URL = os.getenv("SELF_PING_URL", "").strip()
KEEP_ALIVE_ENABLED = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() in ("true", "1", "yes")
KEEP_ALIVE_INTERVAL_SECONDS = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", "600"))  # 10 minutes default
KEEP_ALIVE_PING_DB = os.getenv("KEEP_ALIVE_PING_DB", "true").lower() in ("true", "1", "yes")


def get_allowed_origins() -> List[str]:
    """Parse comma-separated origin string into a clean list of allowed CORS origins."""
    origins = [o.strip() for o in FRONTEND_ORIGINS_RAW.split(",") if o.strip()]
    return origins

