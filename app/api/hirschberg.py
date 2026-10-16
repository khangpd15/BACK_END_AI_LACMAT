"""Image API and session management for the trained Hirschberg localization and screening pipeline."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import PROJECT_ROOT
from app.db.database import get_db_session
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
        "Optionally persists session locally to data/hirschberg/sessions/<session_id>/ and to PostgreSQL. "
        "Strict customer privacy: zero biometric user image persisted."
    ),
)
async def predict(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None, description="Optional custom session ID; auto-generated if omitted"),
    orientation: Literal["auto", "unmirrored", "mirrored"] = Form("auto"),
    kind: Literal["auto", "fullface", "bilateral"] = Form("auto"),
    fixation: Literal["unknown", "confirmed", "offaxis"] = Form("unknown"),
    save_session: bool = Form(True, description="Whether to persist session data locally and in DB"),
    db: AsyncSession = Depends(get_db_session),
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
                session_service = HirschbergSessionService(db_session=db)
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
    db: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    session_service = HirschbergSessionService(db_session=db)

    # 1. Try DB lookup first
    if session_service.repo:
        db_record = await session_service.repo.get_session_by_id(session_id)
        if db_record:
            return {
                "sessionId": db_record.session_id,
                "status": db_record.status,
                "predictedClass": db_record.predicted_class,
                "confidence": db_record.confidence,
                "reason": db_record.reason,
                "message": db_record.message_vi,
                "modelId": db_record.model_id,
                "featureContract": db_record.feature_contract,
                "orientation": db_record.orientation,
                "inputKind": db_record.input_kind,
                "measurements": db_record.measurements,
                "quality": db_record.quality,
                "uncalibratedModelScores": db_record.uncalibrated_scores,
                "latencyMs": db_record.latency_ms,
                "sessionPath": db_record.session_path,
                "createdAt": db_record.created_at.isoformat() if db_record.created_at else None,
                "metadata": db_record.metadata_json,
                "source": "database",
            }

    # 2. Fallback to local files
    local_dir = os.path.join(session_service.sessions_base_dir, session_id)
    result_path = os.path.join(local_dir, "result.json")
    metadata_path = os.path.join(local_dir, "metadata.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)
        meta_data = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        return {
            **res_data,
            "metadata": meta_data,
            "source": "filesystem",
        }

    raise HTTPException(status_code=404, detail=f"Hirschberg session '{session_id}' not found.")


@router.get(
    "/sessions",
    summary="List saved Hirschberg sessions",
    description="Lists recent Hirschberg screening sessions from the database.",
)
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    session_service = HirschbergSessionService(db_session=db)
    if not session_service.repo:
        return {"sessions": [], "count": 0}
    records = await session_service.repo.list_sessions(limit=limit, offset=offset)
    return {
        "count": len(records),
        "limit": limit,
        "offset": offset,
        "sessions": [
            {
                "sessionId": r.session_id,
                "status": r.status,
                "predictedClass": r.predicted_class,
                "confidence": r.confidence,
                "reason": r.reason,
                "modelId": r.model_id,
                "latencyMs": r.latency_ms,
                "createdAt": r.created_at.isoformat() if r.created_at else None,
                "sessionPath": r.session_path,
            }
            for r in records
        ],
    }
