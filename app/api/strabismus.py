"""Strabismus screening API endpoints for RemiCare.

Provides:
- POST /api/v1/strabismus/predict: bilateral image screening with Quality Gate & ONNX inference.
- GET /api/v1/strabismus/health: dedicated health check for model readiness.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.db.repositories.strabismus_repository import StrabismusRepository
from app.services.strabismus_inference_service import get_strabismus_inference_service

logger = logging.getLogger("remicare.api.strabismus")

router = APIRouter(prefix="/api/v1/strabismus", tags=["Strabismus Screening"])

# Supported image MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.get(
    "/health",
    summary="Strabismus Screening Model Health Check",
    description="Returns the status and version of the bilateral strabismus ONNX model resident in RAM.",
)
async def strabismus_health() -> Dict[str, Any]:
    """Dedicated health check verifying model memory residency."""
    svc = get_strabismus_inference_service()
    if svc.is_loaded:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": svc.model_version,
            "threshold": svc.threshold,
        }
    return {
        "status": "error",
        "model_loaded": False,
    }


@router.post(
    "/predict",
    summary="Screen bilateral eye frame for strabismus",
    description=(
        "Accepts a single RGB frame containing both eyes captured during primary gaze (STRAIGHT). "
        "Applies Quality Gate validation, runs Deep Learning ONNX inference, and returns NORMAL, "
        "SUSPICIOUS, or INCONCLUSIVE. Zero image persistence: user images are strictly discarded from memory."
    ),
    responses={
        200: {"description": "Screening successful or inconclusive due to quality"},
        400: {"description": "Invalid file upload or empty buffer"},
        413: {"description": "Image file exceeds 10 MB limit"},
        415: {"description": "Unsupported media type. Only JPEG, PNG, and WebP are allowed"},
        500: {"description": "Internal screening engine error"},
    },
)
async def predict_strabismus(
    image: UploadFile = File(..., description="Multipart image file (JPEG, PNG, or WebP) containing both eyes"),
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Processes bilateral image screening with Quality Gate and ONNX model."""
    # 1. Validate MIME type
    content_type = (image.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME_TYPES:
        # Fallback check on filename extension if content_type is octet-stream
        ext = (image.filename or "").lower().split(".")[-1]
        if ext not in ("jpg", "jpeg", "png", "webp"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media type '{content_type}'. Allowed types: JPEG, PNG, WebP.",
            )

    image_bytes = b""
    try:
        # 2. Read image buffer with size cap
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received.",
            )

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({len(image_bytes)} bytes) exceeds the maximum allowed 10 MB limit.",
            )

        # 3. Perform Quality Gate & Deep Learning Inference via Singleton Service
        inference_service = get_strabismus_inference_service()
        try:
            result = await run_in_threadpool(inference_service.predict, image_bytes)
        except Exception as inf_err:
            logger.error("[PredictStrabismusError] Inference failed: %s", inf_err, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Strabismus inference engine encountered an internal error.",
            )

        # 4. Save metadata to Supabase / PostgreSQL (strictly metadata only, zero image bytes)
        try:
            repo = StrabismusRepository(session)
            await repo.save_screening(
                status=result["status"],
                strabismus_probability=result.get("strabismus_probability"),
                confidence=result.get("confidence"),
                quality_score=result.get("quality_score"),
                threshold=result["threshold"],
                model_version=result["model_version"],
                inference_latency_ms=result.get("inference_latency_ms"),
                failure_reason=result.get("reason"),
            )
        except Exception as db_err:
            # Non-blocking log if DB pool is temporarily unreachable
            logger.warning("[StrabismusDBWarning] Could not persist screening metadata: %s", db_err)

        # 5. Format response adhering to specification
        if result["status"] == "INCONCLUSIVE":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "INCONCLUSIVE",
                    "reason": result.get("reason", "IMAGE_QUALITY_FAILED"),
                    "quality_score": result.get("quality_score", 0.0),
                    "threshold": result["threshold"],
                    "model_version": result["model_version"],
                },
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": result["status"],
                "strabismus_probability": result["strabismus_probability"],
                "confidence": result["confidence"],
                "quality_score": result["quality_score"],
                "threshold": result["threshold"],
                "model_version": result["model_version"],
                "inference_latency_ms": result["inference_latency_ms"],
            },
        )

    finally:
        # 6. Guaranteed buffer and file descriptor cleanup
        if image:
            await image.close()
        del image_bytes
