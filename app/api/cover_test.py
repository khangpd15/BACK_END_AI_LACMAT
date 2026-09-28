"""New API endpoint for RemiCare Cover Test persistent cloud storage and research inference."""

import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.schemas.cover_test import (
    CoverTestSessionMetadataInput,
    CoverTestSessionResponse,
)
from app.services.cover_test.session_service import CoverTestSessionService

logger = logging.getLogger("remicare.api.cover_test")

router = APIRouter(prefix="/sessions", tags=["Cover Test Storage"])


@router.post(
    "",
    response_model=CoverTestSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Persist Cover Test 3-cycle session, trajectories, and eye crops to Supabase & PostgreSQL",
    description=(
        "Receives multipart/form-data containing session metadata, 3 raw JSON trajectories, "
        "and protocol-triggered eye-region crop images (up to 6 images). "
        "Guarantees raw data persistence in Supabase Storage and PostgreSQL before executing optional AI transfer inference."
    ),
)
async def persist_cover_test_session(
    session_metadata: str = Form(..., description="JSON string of CoverTestSessionMetadataInput"),
    cycle_1_raw: Optional[UploadFile] = File(None),
    cycle_1_left_eye: Optional[UploadFile] = File(None),
    cycle_1_right_eye: Optional[UploadFile] = File(None),
    cycle_2_raw: Optional[UploadFile] = File(None),
    cycle_2_left_eye: Optional[UploadFile] = File(None),
    cycle_2_right_eye: Optional[UploadFile] = File(None),
    cycle_3_raw: Optional[UploadFile] = File(None),
    cycle_3_left_eye: Optional[UploadFile] = File(None),
    cycle_3_right_eye: Optional[UploadFile] = File(None),
    run_inference: bool = Query(True, description="Whether to execute AI inference after storage"),
    db: AsyncSession = Depends(get_db_session),
) -> CoverTestSessionResponse:
    # 1. Parse session_metadata JSON
    try:
        meta_dict = json.loads(session_metadata)
        metadata_obj = CoverTestSessionMetadataInput(**meta_dict)
    except Exception as e:
        logger.warning("Failed to parse session_metadata Form field: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid session_metadata JSON: {str(e)}",
        )

    # 2. Extract and parse raw cycle trajectories
    raw_files = [
        (1, cycle_1_raw),
        (2, cycle_2_raw),
        (3, cycle_3_raw),
    ]
    raw_trajectories = {}
    for c_num, raw_file in raw_files:
        if raw_file is not None and raw_file.filename:
            try:
                content_bytes = await raw_file.read()
                if content_bytes:
                    raw_trajectories[c_num] = json.loads(content_bytes.decode("utf-8"))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cycle {c_num} raw trajectory JSON corrupted: {str(e)}",
                )

    if not raw_trajectories:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one cycle raw trajectory (cycle_1_raw) is required.",
        )

    # 3. Extract eye crop images
    image_files = [
        (1, "LEFT", cycle_1_left_eye),
        (1, "RIGHT", cycle_1_right_eye),
        (2, "LEFT", cycle_2_left_eye),
        (2, "RIGHT", cycle_2_right_eye),
        (3, "LEFT", cycle_3_left_eye),
        (3, "RIGHT", cycle_3_right_eye),
    ]
    images_data = {}
    for c_num, eye_side, img_file in image_files:
        if img_file is not None and img_file.filename:
            b = await img_file.read()
            ct = img_file.content_type or "image/jpeg"
            images_data[(c_num, eye_side)] = (b, ct)

    # 4. Delegate to Session Service
    session_service = CoverTestSessionService(db_session=db)
    return await session_service.persist_session(
        metadata_input=metadata_obj,
        raw_trajectories=raw_trajectories,
        images_data=images_data,
        run_inference=run_inference,
    )
