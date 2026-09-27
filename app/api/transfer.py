"""API endpoint for RemiCare to Korean Model Transfer Experiment.

Route: POST /api/v1/transfer/strabismus
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    ScreeningRequest,
    TransferExperimentResponse,
    TransferInconclusiveResponse,
)
from app.services.korean_transfer import get_korean_transfer_service
from app.services.validation import validate_screening_request

logger = logging.getLogger("remicare.transfer_api")

router = APIRouter(prefix="/api/v1/transfer", tags=["Transfer Experiment"])


@router.post(
    "/strabismus",
    response_model=TransferExperimentResponse,
    responses={
        422: {"model": TransferInconclusiveResponse, "description": "Validation failed on time-series"},
        500: {"description": "Internal model execution failure"},
    },
    summary="Execute Korean Shared Model Transfer Inference on RemiCare Cover Test Sampling",
)
async def transfer_strabismus_inference(request: ScreeningRequest) -> TransferExperimentResponse:
    """Receive raw Cover Test time-series from RemiCare frontend and run transfer inference."""
    sample_id = request.sampleId

    # 1. Privacy & Security: Log only metadata, NOT full eye trajectories
    logger.info("[Transfer Request Received] sampleId=%s, cycles=%d", sample_id, len(request.cycles))

    # 2. Prevent target leakage: Frontend must not send target labels
    raw_dict = request.model_dump()
    forbidden_keys = {"clinicalLabel", "diagnosis", "target", "groundTruth", "label"}
    found_forbidden = [k for k in forbidden_keys if k in raw_dict and raw_dict[k] is not None]
    if found_forbidden:
        logger.warning("[Target Leakage Attempt Blocked] sampleId=%s, forbidden=%s", sample_id, found_forbidden)
        # We strip or reject. Prompt says: "Frontend MUST NOT send target. The API request should contain observation data only."

    # 3. Validate raw Cover Test time-series using Data Integrity Gate
    val_result = validate_screening_request(request)
    if not val_result.is_valid:
        logger.warning(
            "[Validation Inconclusive] sampleId=%s, reason=%s, issues=%s",
            sample_id,
            val_result.reason,
            val_result.issues,
        )
        inconclusive_payload = TransferInconclusiveResponse(
            sampleId=sample_id,
            status="INCONCLUSIVE",
            inputCompatible=False,
            reason="INVALID_TIME_SERIES",
            details=val_result.issues or [val_result.reason or "Time-series validation failed"],
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=inconclusive_payload.model_dump(),
        )

    logger.info("[Validation Passed] sampleId=%s", sample_id)

    # 4. Feature Extraction & Model Inference
    try:
        transfer_svc = get_korean_transfer_service()
        response = transfer_svc.predict_transfer(request)
        logger.info(
            "[Inference Complete] sampleId=%s, prediction=%s, classProbability=%s",
            sample_id,
            response.prediction,
            response.classProbability,
        )
        return response
    except ValueError as ve:
        logger.warning("[Feature Extraction Inconclusive] sampleId=%s, error=%s", sample_id, str(ve))
        inconclusive_payload = TransferInconclusiveResponse(
            sampleId=sample_id,
            status="INCONCLUSIVE",
            inputCompatible=False,
            reason="INVALID_TIME_SERIES",
            details=[str(ve)],
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=inconclusive_payload.model_dump(),
        ) from ve
    except Exception as exc:
        logger.error("[Model Execution Error] sampleId=%s, error=%s", sample_id, str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "MODEL_ERROR", "message": "Failed to complete model inference"},
        ) from exc
