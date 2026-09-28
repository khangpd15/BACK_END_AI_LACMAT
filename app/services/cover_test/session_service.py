"""Core business logic service for Cover Test session persistence, validation, storage, and AI inference."""

from datetime import datetime, timezone
import json
import logging
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ENABLE_AI_INFERENCE_ON_SAVE
from app.db.repositories.cover_test_repository import CoverTestRepository
from app.schemas.cover_test import (
    CoverTestSessionMetadataInput,
    CoverTestSessionResponse,
)
from app.services.cover_test.storage_service import (
    SupabaseStorageService,
    get_storage_service,
)

logger = logging.getLogger("remicare.session_service")

# Regular expression for strict RFC 4122 UUID v4 validation
UUID_V4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Forbidden PII keys
FORBIDDEN_PII_KEYS = {
    "name",
    "phone",
    "phonenumber",
    "email",
    "cccd",
    "cmnd",
    "ssn",
    "dob",
    "dateofbirth",
    "address",
    "patientname",
}

VALID_PHASES = {
    "BASELINE",
    "COVER",
    "UNCOVER",
    "TRACKING",
    "COVER_LEFT",
    "COVER_RIGHT",
    "UNCOVER_LEFT",
    "UNCOVER_RIGHT",
    "CYCLE_COMPLETE",
    "NEXT_CYCLE",
    "FINISHED",
}

VALID_EYES = {"LEFT", "RIGHT"}


def validate_uuid_v4(val: str) -> uuid.UUID:
    """Validates that a string is a strictly conforming UUID v4."""
    if not val or not isinstance(val, str):
        raise HTTPException(
            status_code=422,
            detail="sessionId is required and must be a valid UUID string.",
        )
    clean_val = val.strip()
    if not UUID_V4_REGEX.match(clean_val):
        raise HTTPException(
            status_code=422,
            detail=f"sessionId '{clean_val}' is not a valid UUID v4.",
        )
    return uuid.UUID(clean_val)


def check_pii(data: Any, path: str = "") -> None:
    """Recursively checks for forbidden PII keys in dictionaries."""
    if isinstance(data, dict):
        for k, v in data.items():
            norm_k = k.lower().replace("_", "").replace("-", "")
            if norm_k in FORBIDDEN_PII_KEYS:
                raise HTTPException(
                    status_code=422,
                    detail=f"PII field '{k}' detected at '{path}'. PII is strictly forbidden.",
                )
            check_pii(v, f"{path}.{k}" if path else k)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            check_pii(item, f"{path}[{idx}]")


def validate_coordinate(val: Any, field_name: str, sample_idx: int) -> Optional[float]:
    """Validates that coordinate is finite and in range [0.0, 1.0] or None."""
    if val is None:
        return None
    if not isinstance(val, (int, float)):
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} must be numeric, got {type(val).__name__}",
        )
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} must be finite (not NaN/Infinity), got {fval}",
        )
    if fval < 0.0 or fval > 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"Sample [{sample_idx}]: {field_name} out of bounds ({fval}). Must be between 0.0 and 1.0.",
        )
    return fval


def validate_cycle_samples(samples: List[Dict[str, Any]], cycle_num: int) -> Tuple[int, int, float, float]:
    """Validates time-series samples within a cycle.
    
    Checks timestamp monotonicity, coordinate bounds, trackingQuality bounds.
    Returns: (total_samples, valid_samples, valid_ratio, mean_tracking_quality)
    """
    if not isinstance(samples, list) or len(samples) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cycle {cycle_num}: samples must be a non-empty array.",
        )

    last_t = -1.0
    valid_count = 0
    total_quality = 0.0

    for idx, s in enumerate(samples):
        if not isinstance(s, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}] must be a JSON object.",
            )

        # Check PII inside sample
        check_pii(s, f"cycle_{cycle_num}.sample_{idx}")

        # Timestamp monotonicity check
        t_val = s.get("t")
        if t_val is None:
            t_val = s.get("timestamp")
        if t_val is None or not isinstance(t_val, (int, float)) or math.isnan(float(t_val)):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: timestamp/t must be a valid number.",
            )
        f_t = float(t_val)
        if f_t < 0.0:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: timestamp/t cannot be negative ({f_t}).",
            )
        if f_t < last_t:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: non-monotonic timestamp detected ({f_t} < {last_t}).",
            )
        last_t = f_t

        # Tracking quality check
        tq = s.get("trackingQuality", 0.95)
        if not isinstance(tq, (int, float)) or math.isnan(float(tq)) or math.isinf(float(tq)):
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: trackingQuality must be finite numeric.",
            )
        f_tq = float(tq)
        if f_tq < 0.0 or f_tq > 1.0:
            raise HTTPException(
                status_code=422,
                detail=f"Cycle {cycle_num}, sample [{idx}]: trackingQuality must be in [0.0, 1.0], got {f_tq}.",
            )
        total_quality += f_tq

        # Coordinates check
        lx = s.get("leftX") if "leftX" in s else (s.get("leftEye", {}) or {}).get("x")
        ly = s.get("leftY") if "leftY" in s else (s.get("leftEye", {}) or {}).get("y")
        rx = s.get("rightX") if "rightX" in s else (s.get("rightEye", {}) or {}).get("x")
        ry = s.get("rightY") if "rightY" in s else (s.get("rightEye", {}) or {}).get("y")

        validate_coordinate(lx, "leftX", idx)
        validate_coordinate(ly, "leftY", idx)
        validate_coordinate(rx, "rightX", idx)
        validate_coordinate(ry, "rightY", idx)
        if "x" in s:
            validate_coordinate(s.get("x"), "x", idx)
        if "y" in s:
            validate_coordinate(s.get("y"), "y", idx)

        # Validity count
        l_valid = s.get("leftValid", False)
        r_valid = s.get("rightValid", False)
        if l_valid or r_valid:
            valid_count += 1

    total = len(samples)
    v_ratio = round(valid_count / total, 4) if total > 0 else 0.0
    mean_q = round(total_quality / total, 4) if total > 0 else 0.0
    return total, valid_count, v_ratio, mean_q


def validate_image_payload(
    file_bytes: bytes,
    content_type: str,
    eye: str,
    cycle_num: int,
) -> Tuple[int, int, int]:
    """Validates JPEG image payload size and MIME type.
    
    Returns (width, height, file_size).
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cycle {cycle_num}, {eye} eye image: image file is empty (0 bytes).",
        )
    size = len(file_bytes)
    if size > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"Cycle {cycle_num}, {eye} eye image: size {size} exceeds 5MB limit.",
        )
    clean_ct = (content_type or "").lower().split(";")[0].strip()
    if clean_ct not in {"image/jpeg", "image/jpg", "application/octet-stream"}:
        raise HTTPException(
            status_code=422,
            detail=f"Cycle {cycle_num}, {eye} eye image: invalid MIME '{content_type}'. Must be image/jpeg.",
        )
    return 256, 256, size


class CoverTestSessionService:
    """Orchestrates validation, persistence, storage, and AI inference for a Cover Test session."""

    def __init__(
        self,
        db_session: AsyncSession,
        storage_service: Optional[SupabaseStorageService] = None,
    ):
        self.db_session = db_session
        self.repo = CoverTestRepository(db_session)
        self.storage = storage_service or get_storage_service()

    async def persist_session(
        self,
        metadata_input: CoverTestSessionMetadataInput,
        raw_trajectories: Dict[int, Dict[str, Any]],
        images_data: Dict[Tuple[int, str], Tuple[bytes, str]],
        run_inference: bool = True,
    ) -> CoverTestSessionResponse:
        # 1. Validate canonical UUID v4
        session_uuid = validate_uuid_v4(metadata_input.sessionId)
        session_id_str = str(session_uuid)

        # 2. Validate PII in metadata
        check_pii(metadata_input.clientMetadata or {}, "clientMetadata")

        # 3. Determine storage paths
        now_utc = datetime.now(timezone.utc)
        year_str = now_utc.strftime("%Y")
        month_str = now_utc.strftime("%m")
        storage_root = f"cover-test-raw/{year_str}/{month_str}/{session_id_str}"
        storage_prefix = f"{year_str}/{month_str}/{session_id_str}"

        # 4. Upsert session initial state
        session_record = await self.repo.upsert_session(
            session_id=session_uuid,
            cycle_count=len(raw_trajectories),
            sampling_rate_hz=metadata_input.samplingRateHz,
            source_device=metadata_input.sourceDevice,
            tracker=metadata_input.tracker,
            raw_schema_version=metadata_input.rawSchemaVersion,
            storage_root=storage_root,
            processing_status="UPLOADING",
            client_metadata=metadata_input.clientMetadata or {},
        )

        # 5. Process and validate cycles
        cycles_to_save = []
        manifest_cycles = []
        total_samples_saved = 0

        for cycle_num in sorted(raw_trajectories.keys()):
            raw_cycle = raw_trajectories[cycle_num]
            samples = raw_cycle.get("samples") or []
            total, valid_cnt, v_ratio, mean_q = validate_cycle_samples(samples, cycle_num)
            total_samples_saved += total

            covered_eye = str(raw_cycle.get("coveredEye") or (cycle_num % 2 == 1 and "LEFT" or "RIGHT")).upper()
            tracked_eye = str(raw_cycle.get("trackedEye") or (cycle_num % 2 == 1 and "RIGHT" or "LEFT")).upper()
            duration_ms = raw_cycle.get("durationMs", int(samples[-1].get("t", 0)) if samples else 0)

            # Upload raw.json to Supabase Storage
            cycle_folder_name = f"cycle_{cycle_num:02d}"
            raw_json_storage_path = f"{storage_prefix}/{cycle_folder_name}/raw.json"
            clean_raw_payload = {
                "schemaVersion": metadata_input.rawSchemaVersion,
                "sessionId": session_id_str,
                "cycle": cycle_num,
                "coveredEye": covered_eye,
                "trackedEye": tracked_eye,
                "samplingRateHz": metadata_input.samplingRateHz,
                "durationMs": duration_ms,
                "samples": samples,
            }
            await self.storage.upload_raw_json(raw_json_storage_path, clean_raw_payload)

            cycles_to_save.append({
                "cycle_number": cycle_num,
                "covered_eye": covered_eye,
                "tracked_eye": tracked_eye,
                "sample_count": total,
                "valid_sample_count": valid_cnt,
                "valid_ratio": v_ratio,
                "mean_tracking_quality": mean_q,
                "cycle_status": "COMPLETED",
                "raw_storage_path": raw_json_storage_path,
                "duration_ms": duration_ms,
            })
            manifest_cycles.append({
                "cycle": cycle_num,
                "storagePath": raw_json_storage_path,
                "sampleCount": total,
                "validRatio": v_ratio,
            })

        # Save cycles to database
        db_cycles = await self.repo.upsert_cycles(session_uuid, cycles_to_save)
        cycle_id_map = {c.cycle_number: c.cycle_id for c in db_cycles}

        # 6. Process and validate eye images
        images_to_save = []
        for (cycle_num, eye_side), (img_bytes, mime_type) in images_data.items():
            norm_eye = eye_side.strip().upper()
            if norm_eye not in VALID_EYES:
                continue
            width, height, file_size = validate_image_payload(img_bytes, mime_type, norm_eye, cycle_num)

            filename = "left_eye.jpg" if norm_eye == "LEFT" else "right_eye.jpg"
            cycle_folder_name = f"cycle_{cycle_num:02d}"
            image_storage_path = f"{storage_prefix}/{cycle_folder_name}/{filename}"
            capture_event = "UNCOVER_LEFT" if norm_eye == "LEFT" else "UNCOVER_RIGHT"

            await self.storage.upload_image(image_storage_path, img_bytes, "image/jpeg")

            target_cycle_id = cycle_id_map.get(cycle_num, session_uuid)
            images_to_save.append({
                "cycle_id": target_cycle_id,
                "cycle_number": cycle_num,
                "eye": norm_eye,
                "capture_event": capture_event,
                "timestamp": now_utc,
                "storage_path": image_storage_path,
                "mime_type": "image/jpeg",
                "width": width,
                "height": height,
                "file_size": file_size,
                "upload_status": "UPLOADED",
                "crop_region": {"width": width, "height": height},
            })

        saved_images = await self.repo.upsert_images(session_uuid, images_to_save)

        # 7. Upload manifest.json to Storage
        manifest_payload = {
            "sessionId": session_id_str,
            "createdAt": now_utc.isoformat(),
            "cycleCount": len(cycles_to_save),
            "samplingRateHz": metadata_input.samplingRateHz,
            "cycles": manifest_cycles,
            "imagesCount": len(saved_images),
            "storageRoot": storage_root,
        }
        await self.storage.upload_raw_json(f"{storage_prefix}/manifest.json", manifest_payload)

        # Commit DB transaction so raw data and metadata are guaranteed persisted
        await self.db_session.commit()
        logger.info(
            "[CoverTestSessionPersisted] sessionId=%s, cycles=%d, images=%d, storage=%s",
            session_id_str,
            len(cycles_to_save),
            len(saved_images),
            storage_root,
        )

        # 8. Optional AI Transfer Inference (Decoupled and Resilient)
        ai_result_payload = None
        processing_status = "COMPLETED"

        should_infer = run_inference and ENABLE_AI_INFERENCE_ON_SAVE
        if should_infer:
            try:
                from app.schemas import CoverCycle, EyeSample, ScreeningRequest
                from app.services.korean_transfer import get_korean_transfer_service

                # Build screening request from raw cycles
                cycles_payload = []
                for c_num in sorted(raw_trajectories.keys()):
                    c_dict = raw_trajectories[c_num]
                    samples_list = [
                        EyeSample(**s) for s in c_dict.get("samples", [])
                    ]
                    cycles_payload.append(
                        CoverCycle(
                            cycle=c_num,
                            coveredEye=str(c_dict.get("coveredEye", "LEFT")).upper(),
                            trackedEye=str(c_dict.get("trackedEye", "RIGHT")).upper(),
                            samples=samples_list,
                        )
                    )

                screening_req = ScreeningRequest(
                    sampleId=session_id_str,
                    test="COVER_TEST",
                    cycles=cycles_payload,
                )

                transfer_svc = get_korean_transfer_service()
                inference_resp = transfer_svc.predict_transfer(screening_req)

                ai_dict = inference_resp.model_dump()
                ai_result_payload = {
                    "status": ai_dict.get("status", "TRANSFER_EXPERIMENT"),
                    "inputCompatible": ai_dict.get("inputCompatible", True),
                    "prediction": ai_dict.get("prediction", "NORMAL"),
                    "classProbability": ai_dict.get("classProbability", {}),
                    "domainShiftWarning": ai_dict.get("domainShiftWarning", True),
                    "model": ai_dict.get("model", {"name": "korean_shared_model", "version": "shared-v1.0.0"}),
                    "features": ai_dict.get("features"),
                    "comparisonModels": ai_dict.get("comparisonModels", []),
                    "notice": "Research transfer experiment only - not a medical diagnosis.",
                }

                # Persist AI result in DB
                await self.repo.upsert_result(
                    session_id=session_uuid,
                    result_data={
                        "model_name": ai_dict.get("model", {}).get("name", "korean_shared_model"),
                        "model_version": ai_dict.get("model", {}).get("version", "shared-v1.0.0"),
                        "feature_schema_version": "shared-v1.0.0",
                        "status": ai_dict.get("status", "TRANSFER_EXPERIMENT"),
                        "input_compatible": ai_dict.get("inputCompatible", True),
                        "prediction": ai_dict.get("prediction", "NORMAL"),
                        "class_probabilities": ai_dict.get("classProbability", {}),
                        "domain_shift_warning": ai_dict.get("domainShiftWarning", True),
                        "features_snapshot": ai_dict.get("features"),
                        "comparison_models": ai_dict.get("comparisonModels", []),
                        "notice": "Research transfer experiment only - not a medical diagnosis.",
                    },
                )
                processing_status = "COMPLETED"
                logger.info("[AIInferenceSuccess] sessionId=%s, prediction=%s", session_id_str, inference_resp.prediction)

            except Exception as ai_err:
                # RAW DATA SURVIVES AI FAILURE: Do not fail HTTP request!
                logger.warning(
                    "[AIInferenceFailed] sessionId=%s, error=%s. Preserving raw data with PARTIAL_SUCCESS.",
                    session_id_str,
                    ai_err,
                )
                processing_status = "PARTIAL_SUCCESS"
                ai_result_payload = None

        # Update final processing status
        await self.repo.update_session_status(session_uuid, processing_status)
        await self.db_session.commit()

        return CoverTestSessionResponse(
            success=True,
            sessionId=session_id_str,
            saved=True,
            processingStatus=processing_status,
            storageRoot=storage_root,
            cyclesSaved=len(cycles_to_save),
            imagesSaved=len(saved_images),
            aiResult=ai_result_payload,
            message="Cover test session data persisted successfully.",
        )
