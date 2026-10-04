"""Versioned research-only geometry measurement API."""

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.config import ENABLE_RESEARCH_MEASUREMENT_API
from app.schemas.research_measurement import (
    ResearchMeasurementRequest,
    ResearchMeasurementResponse,
)
from app.services.research_measurement_service import (
    ResearchMeasurementError,
    measure_research_request,
)

logger = logging.getLogger("remicare.api.research_measurement")

router = APIRouter(prefix="/api/v1/research", tags=["Research Measurement"])
_measurement_lock = asyncio.Lock()


@router.post(
    "/measurements",
    response_model=ResearchMeasurementResponse,
    summary="Research-only Hirschberg/Cover geometry measurement",
    description=(
        "Accepts a versioned research request for HIRSCHBERG or COVER. "
        "This endpoint does not replace /api/v1/strabismus/predict and does not produce a normal screening clearance."
    ),
)
async def create_research_measurement(request: ResearchMeasurementRequest) -> Dict[str, Any]:
    if not ENABLE_RESEARCH_MEASUREMENT_API:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "INVALID_REQUEST",
                "message": "Research measurement API is disabled on this backend.",
            },
        )

    try:
        await asyncio.wait_for(_measurement_lock.acquire(), timeout=2.0)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, headers={"Retry-After": "2"}, detail={
            "code": "SERVER_BUSY", "message": "Measurement engine is busy. Please try again."
        }) from exc
    try:
        return await run_in_threadpool(measure_research_request, request)
    except ResearchMeasurementError as exc:
        logger.info(
            "[ResearchMeasurementRejected] requestId=%s sessionId=%s code=%s",
            request.requestId,
            request.sessionId,
            exc.code,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except Exception as exc:
        logger.error("[ResearchMeasurementError] Unhandled error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INFERENCE_ERROR", "message": "Research measurement failed."},
        ) from exc
    finally:
        _measurement_lock.release()
