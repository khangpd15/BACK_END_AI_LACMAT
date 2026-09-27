"""RemiCare Strabismus AI Backend - Main Application Entrypoint.

Provides FastAPI application initialization, CORS configuration,
health check endpoint, and API router registration.
"""

import logging
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.screening import router as screening_router

# Configure structured audit logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("remicare.main")

# Initialize FastAPI application
app = FastAPI(
    title="RemiCare Strabismus AI Screening Backend",
    description=(
        "Specialized AI backend for webcam-based Cover Test screening. "
        "Provides data integrity validation, time-series preprocessing, "
        "cycle-based movement feature extraction, and ML inference. "
        "Screening result only — not a clinical diagnosis."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for development frontend
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(screening_router)


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns backend service health status, name, and version.",
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint responding with operational status and version."""
    return {
        "status": "ok",
        "service": "remicare-strabismus-ai",
        "version": __version__,
    }
