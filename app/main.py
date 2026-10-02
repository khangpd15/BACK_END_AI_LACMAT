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
from starlette.middleware.gzip import GZipMiddleware
import time
from fastapi.responses import JSONResponse

from app import __version__
from app.api.transfer import router as transfer_router
from app.api.strabismus import router as strabismus_router
from app.services.strabismus_inference_service import get_strabismus_inference_service
from app.api.cover_test_session import router as cover_test_session_router
from app.api.cover_test import router as cover_test_v1_router
from app.config import get_allowed_origins
from app.db.database import check_db_connection, init_db
from app.services.keep_alive import (
    get_keep_alive_service,
    start_keep_alive,
    stop_keep_alive,
)
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
    """Application lifespan: Preload machine learning models, init DB, and start keep-alive worker."""
    logger.info("Initializing RemiCare Strabismus AI Backend - Phase 5...")
    try:
        # Initialize database tables
        await init_db()
    except Exception as e:
        logger.warning("Database init check note: %s", e)

    try:
        transfer_svc = get_korean_transfer_service()
        logger.info(
            "Transfer model loaded: experiment=%s, version=%s, features=%d, path=%s",
            transfer_svc.experiment,
            transfer_svc.artifact.get("version") if transfer_svc.artifact else "Unknown",
            len(transfer_svc.feature_names),
            transfer_svc.model_path,
        )
    except Exception as e:
        logger.error("Failed to preload transfer model at startup: %s", e, exc_info=True)

    try:
        strabismus_svc = get_strabismus_inference_service()
        if strabismus_svc.is_loaded:
            logger.info(
                "Strabismus bilateral ONNX model preloaded: version=%s, path=%s, input=%s",
                strabismus_svc.model_version,
                strabismus_svc.resolved_model_path,
                strabismus_svc.input_name,
            )
        else:
            logger.warning("Strabismus bilateral ONNX model is not loaded yet.")
    except Exception as e:
        logger.error("Failed to preload strabismus ONNX model at startup: %s", e, exc_info=True)

    # Start automated keep-alive self-ping and DB pool warming worker
    start_keep_alive()

    yield

    # Clean graceful shutdown
    logger.info("Shutting down RemiCare Strabismus AI Backend...")
    await stop_keep_alive()


# Initialize FastAPI application
app = FastAPI(
    title="RemiCare Strabismus AI Screening Backend",
    description=(
        "Specialized AI backend for webcam-based Cover Test screening. "
        "Provides data integrity validation, time-series preprocessing, "
        "cycle-based movement feature extraction, and ML inference. "
        "Screening result only - not a clinical diagnosis."
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

# Enable HTTP response compression for payloads >= 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def performance_measurement_middleware(request: Request, call_next):
    """Development and production non-invasive performance measurement middleware."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Attach Server-Timing header for client-side DevTools and metrics inspection
    response.headers["Server-Timing"] = f"total;dur={duration_ms:.1f}"

    content_length = response.headers.get("content-length")
    size_str = f"{int(content_length)}B" if content_length else "chunked"

    if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
        logger.info(
            "[RemiCare API Perf] %s %s | status=%d | time=%.1fms | size=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            size_str,
        )
    return response


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
app.include_router(transfer_router)
# Legacy endpoint maintained for compatibility
app.include_router(cover_test_session_router, prefix="/api/cover-test")
# New Phase 3 persistent cloud storage router
app.include_router(cover_test_v1_router, prefix="/api/v1/cover-test")
# Strabismus Bilateral Screening Router
app.include_router(strabismus_router)


@app.get(
    "/ping",
    tags=["System"],
    summary="Fast keep-alive ping endpoint",
    description=(
        "Ultra-lightweight endpoint for automated keep-alive services, Render self-ping, "
        "and uptime monitoring. Returns immediately with minimal overhead to prevent spin-downs."
    ),
)
async def ping() -> Dict[str, Any]:
    """Fast keep-alive ping responding with alive status and timestamp."""
    return {
        "status": "alive",
        "service": "remicare-strabismus-ai",
        "timestamp": time.time(),
    }


@app.get(
    "/health/db",
    tags=["System"],
    summary="Database connectivity diagnostics",
    description="Tests live connection to PostgreSQL / Supabase with SELECT 1 and returns connection diagnostics.",
)
async def db_health() -> Dict[str, Any]:
    """Diagnostics endpoint testing database connection and checking for network/Errno 101 issues."""
    return await check_db_connection()


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns backend service health status, model info, keep-alive metrics, and optional DB status.",
)
async def health_check(check_db: bool = False) -> Dict[str, Any]:
    """Health check endpoint responding with operational status, version, and active model info."""
    try:
        svc = get_korean_transfer_service()
        model_info = {
            "experiment": svc.experiment,
            "version": svc.artifact.get("version") if svc.artifact else None,
            "featureCount": len(svc.feature_names),
            "coordinateRescaling": svc.coordinate_rescaling,
            "ipdScaleFactor": svc.ipd_scale_factor,
        }
    except Exception:
        model_info = {"status": "not_loaded"}

    keep_alive_info = get_keep_alive_service().get_status()

    result: Dict[str, Any] = {
        "status": "ok",
        "service": "remicare-strabismus-ai",
        "version": __version__,
        "model": model_info,
        "keepAlive": keep_alive_info,
    }

    if check_db:
        db_res = await check_db_connection()
        result["database"] = db_res
        if not db_res.get("connected", False):
            result["status"] = "degraded"

    return result

