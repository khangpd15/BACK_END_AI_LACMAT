"""RemiCare Strabismus AI Backend - Main Application Entrypoint.

Provides FastAPI application initialization, CORS configuration,
health check endpoint, and API router registration.
"""

from contextlib import asynccontextmanager
import logging
import math
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.screening import router as screening_router
from app.api.transfer import router as transfer_router
from app.config import get_allowed_origins
from app.services.korean_transfer import get_korean_transfer_service

# Configure structured audit logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("remicare.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: Preload machine learning models once at startup."""
    logger.info("Initializing RemiCare Strabismus AI Backend services...")
    try:
        transfer_svc = get_korean_transfer_service()
        logger.info(
            "Korean Shared Model successfully preloaded at startup (Features: %d, Model: %s)",
            len(transfer_svc.feature_names),
            transfer_svc.artifact.get("name") if transfer_svc.artifact else "Unknown",
        )
    except Exception as e:
        logger.error("Failed to preload Korean Shared Model at startup: %s", e, exc_info=True)
    yield
    logger.info("Shutting down RemiCare Strabismus AI Backend...")


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
    lifespan=lifespan,
)

# CORS configuration: configurable via REMICARE_FRONTEND_ORIGINS env var
# Never uses wildcard '*' in production
allowed_origins = get_allowed_origins()
logger.info("Configured CORS Allowed Origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _sanitize_validation_errors(obj: Any) -> Any:
    """Recursively replace non-finite float numbers with strings for JSON response safety."""
    if isinstance(obj, list):
        return [_sanitize_validation_errors(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_validation_errors(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors safely ensuring non-finite values don't break JSON encoding."""
    sanitized = _sanitize_validation_errors(exc.errors())
    return JSONResponse(status_code=422, content={"detail": sanitized})


# Register API routes
app.include_router(screening_router)
app.include_router(transfer_router)


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
