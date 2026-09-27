"""API endpoints for RemiCare Strabismus Screening."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, Request, status
from pydantic import ValidationError

from app.schemas import (
    AnalysisSummary,
    QualitySummary,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningStatus,
)
from app.services.screening import run_screening_pipeline

logger = logging.getLogger("remicare.api")

router = APIRouter(prefix="/api/v1/screening", tags=["Screening"])


@router.post(
    "/analyze",
    response_model=ScreeningResponse,
    summary="Analyze Cover Test time-series screening data",
    description=(
        "Receives raw Cover Test time-series JSON from React frontend, "
        "validates data integrity, performs preprocessing and feature extraction, "
        "and evaluates screening outcome. Screening result only — not a clinical diagnosis."
    ),
)
async def analyze_cover_test(raw_payload: Dict[str, Any] = Body(...)) -> ScreeningResponse:
    """Analyze Cover Test time-series payload with graceful data integrity handling."""
    # Attempt to parse into ScreeningRequest schema
    try:
        request_obj = ScreeningRequest.model_validate(raw_payload)
    except ValidationError as ve:
        sample_id = str(raw_payload.get("sampleId", "unknown")) if isinstance(raw_payload, dict) else "unknown"
        error_messages = [f"{err['loc']}: {err['msg']}" for err in ve.errors()]
        logger.warning(f"[SchemaValidationError] sampleId={sample_id} errors={error_messages}")

        return ScreeningResponse(
            sampleId=sample_id,
            status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
            modelVersion=None,
            quality=QualitySummary(
                status="FAIL",
                reason="INVALID_SCHEMA_PAYLOAD",
                issues=error_messages,
            ),
            analysis=AnalysisSummary(cyclesAnalyzed=0),
            reason="INVALID_SCHEMA_PAYLOAD",
            notice="Screening result only — not a diagnosis.",
        )
    except Exception as ex:
        sample_id = str(raw_payload.get("sampleId", "unknown")) if isinstance(raw_payload, dict) else "unknown"
        logger.error(f"[PayloadParseError] sampleId={sample_id} error={ex}")
        return ScreeningResponse(
            sampleId=sample_id,
            status=ScreeningStatus.SCREENING_INCONCLUSIVE.value,
            modelVersion=None,
            quality=QualitySummary(
                status="FAIL",
                reason="MALFORMED_REQUEST_BODY",
                issues=[str(ex)],
            ),
            analysis=AnalysisSummary(cyclesAnalyzed=0),
            reason="MALFORMED_REQUEST_BODY",
            notice="Screening result only — not a diagnosis.",
        )

    # Run complete pipeline
    response = run_screening_pipeline(request_obj)
    return response
