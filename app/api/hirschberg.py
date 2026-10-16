"""Image API and session management for the trained Hirschberg localization and screening pipeline."""

import asyncio
import logging
from typing import Any, Dict, Literal, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.hirschberg_session_service import HirschbergSessionService
from app.services.trained_hirschberg_service import (
    predict_trained_hirschberg,
    trained_hirschberg_status,
)

logger = logging.getLogger("remicare.api.hirschberg")

_prediction_lock = asyncio.Lock()

router = APIRouter(prefix="/api/v1/hirschberg", tags=["Hirschberg"])


@router.get("/health", summary="Hirschberg Model Health Check")
async def health():
    """Returns the status and checkpoint readiness of the trained Hirschberg CNN."""
    return trained_hirschberg_status()


@router.post(
    "/predict",
    summary="Run trained Hirschberg CNNs with landmark and capture QC",
    description=(
        "Accepts an image and runs trained Hirschberg CNN with pupil/iris landmark and capture QC. "
        "Optionally persists session locally to data/hirschberg/sessions/<session_id>/. "
        "Strict customer privacy: zero biometric user image persisted."
    ),
)
async def predict(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None, description="Optional custom session ID; auto-generated if omitted"),
    orientation: Literal["auto", "unmirrored", "mirrored"] = Form("auto"),
    kind: Literal["auto", "fullface", "bilateral"] = Form("auto"),
    fixation: Literal["unknown", "confirmed", "offaxis"] = Form("unknown"),
    save_session: bool = Form(True, description="Whether to persist session data locally"),
):
    try:
        raw = await image.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Image exceeds 10 MB")
        if not raw:
            raise HTTPException(400, "Empty image")

        filename = image.filename or "image.png"
        if _prediction_lock.locked():
            raise HTTPException(503, "Hirschberg is busy; retry shortly", headers={"Retry-After": "2"})
        async with _prediction_lock:
            prediction_result = await run_in_threadpool(
                predict_trained_hirschberg, raw, filename, orientation, kind, fixation,
            )

        if save_session:
            try:
                session_service = HirschbergSessionService()
                save_info = await session_service.persist_session(
                    session_id=session_id,
                    filename=filename,
                    orientation=orientation,
                    input_kind=kind,
                    fixation=fixation,
                    prediction_result=prediction_result,
                )
                prediction_result["sessionId"] = save_info["sessionId"]
                prediction_result["sessionPath"] = save_info["sessionPath"]
                prediction_result["saved"] = save_info["saved"]
            except Exception as save_err:
                logger.error("[HirschbergSessionSaveFailed] Failed to persist session: %s", save_err)
                prediction_result["saved"] = False
                prediction_result["saveError"] = str(save_err)
        else:
            prediction_result["saved"] = False

        return prediction_result
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, "Trained Hirschberg runtime unavailable") from exc
    finally:
        await image.close()


@router.get(
    "/sessions/{session_id}",
    summary="Get saved Hirschberg session details",
    description="Retrieves metadata and results for a stored Hirschberg session.",
)
async def get_session(
    session_id: str,
) -> Dict[str, Any]:
    service = HirschbergSessionService()
    try:
        record = await run_in_threadpool(service.get_session_local, session_id)
    except ValueError as exc:
        raise HTTPException(400, 'Invalid session ID') from exc
    if record is None:
        raise HTTPException(404, 'Hirschberg session not found')
    return record


@router.get('/sessions', summary='List local Hirschberg sessions')
async def list_sessions(limit: int = Query(50, ge=1, le=200),
                        offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    return await run_in_threadpool(HirschbergSessionService().list_sessions_local, limit, offset)
