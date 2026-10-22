"""RemiCare Hirschberg AI Backend - Main Application Entrypoint.

Provides FastAPI application initialization, CORS configuration,
health check endpoints, and Hirschberg inference routers.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import math
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.hirschberg import router as hirschberg_router
from app.api.research_measurement import router as research_measurement_router
from app.config import get_allowed_origins
from app.services.keep_alive import (
    get_keep_alive_service,
    start_keep_alive,
    stop_keep_alive,
)
from app.services.trained_hirschberg_service import (
    preload_trained_hirschberg,
    trained_hirschberg_status,
)

# Configure structured audit logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("remicare.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: Preload the Hirschberg model and start the optional HTTP keep-alive worker."""
    logger.info("Initializing RemiCare Hirschberg AI Backend...")
    # Uvicorn binds its HTTP socket only after lifespan reaches yield. The CNN
    # bundle can take minutes to verify/load on a small Render instance.
    # Its service lock already serializes warmup with incoming predictions.
    async def warm_hirschberg():
        logger.info("Loading Hirschberg CNN in background...")
        try:
            await asyncio.to_thread(preload_trained_hirschberg)
            logger.info("Hirschberg warmup finished; inspect /api/v1/hirschberg/health for readiness.")
        except Exception:
            logger.exception("Hirschberg background warmup failed")

    hirschberg_warmup = asyncio.create_task(warm_hirschberg(), name="hirschberg-warmup")

    # Start automated keep-alive HTTP self-ping worker
    start_keep_alive()

    try:
        yield
    finally:
        hirschberg_warmup.cancel()
        with suppress(asyncio.CancelledError):
            await hirschberg_warmup
        logger.info("Shutting down RemiCare Hirschberg AI Backend...")
        await stop_keep_alive()


# Initialize FastAPI application
app = FastAPI(
    title="RemiCare Hirschberg AI Screening Backend",
    description=(
        "Specialized AI backend for Hirschberg ocular alignment test, "
        "bilateral eye screening, and corneal light reflex (Purkinje-I) analysis. "
        "Screening result only - not a clinical diagnosis."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)



# Enable HTTP response compression for payloads >= 1KB
@app.middleware("http")
async def performance_measurement_middleware(request: Request, call_next):
    """Performance measurement middleware."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["Server-Timing"] = f"total;dur={duration_ms:.1f}"

    content_length = response.headers.get("content-length")
    size_str = f"{int(content_length)}B" if content_length else "chunked"

    if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
        logger.info(
            "[Hirschberg API Perf] %s %s | status=%d | time=%.1fms | size=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            size_str,
        )
    return response
app.add_middleware(GZipMiddleware, minimum_size=1000)


# CORS configuration
allowed_origins = get_allowed_origins()
logger.info("Configured CORS Allowed Origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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
app.include_router(hirschberg_router)
app.include_router(research_measurement_router)


@app.get(
    "/ping",
    tags=["System"],
    summary="Fast keep-alive ping endpoint",
    description="Lightweight endpoint for keep-alive services, Render self-ping, and uptime monitoring.",
)
async def ping() -> Dict[str, Any]:
    """Fast keep-alive ping responding with alive status and timestamp."""
    return {
        "status": "alive",
        "service": "remicare-hirschberg-ai",
        "timestamp": time.time(),
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns backend service health status, Hirschberg CNN readiness.",
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint responding with operational status, version, and active model info."""
    hirschberg_info = trained_hirschberg_status()
    keep_alive_info = get_keep_alive_service().get_status()

    result: Dict[str, Any] = {
        "status": "ok",
        "service": "remicare-hirschberg-ai",
        "version": __version__,
        "models": {
            "hirschberg": hirschberg_info,
        },
        "deploymentCommit": os.getenv("RENDER_GIT_COMMIT"),
        "keepAlive": keep_alive_info,
        "sessionStorage": "local",
    }

    return result


@app.get("/ready", tags=["System"], summary="Model readiness for deployment")
async def readiness():
    model = trained_hirschberg_status()
    return JSONResponse(status_code=200 if model["loaded"] else 503,
                        content={"ready": model["loaded"], "hirschberg": model})
