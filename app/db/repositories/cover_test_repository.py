"""Repository layer for Cover Test Database operations."""

import logging
import uuid
import math
from typing import Any, Dict, List, Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import (
    CoverTestCycleModel,
    CoverTestImageModel,
    CoverTestResultModel,
    CoverTestSessionModel,
)

logger = logging.getLogger("remicare.repository")


def sanitize_json_data(obj: Any) -> Any:
    """Recursively replaces NaN and inf with None for PostgreSQL JSONB compliance."""
    if isinstance(obj, dict):
        return {k: sanitize_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json_data(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


class CoverTestRepository:
    """Handles all database persistence for Cover Test sessions, cycles, images, and AI results."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_session(self, session_id: uuid.UUID) -> Optional[CoverTestSessionModel]:
        stmt = select(CoverTestSessionModel).where(CoverTestSessionModel.session_id == session_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def upsert_session(
        self,
        session_id: uuid.UUID,
        cycle_count: int,
        sampling_rate_hz: float,
        source_device: str,
        tracker: str,
        raw_schema_version: str,
        storage_root: str,
        processing_status: str,
        client_metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> CoverTestSessionModel:
        clean_metadata = sanitize_json_data(client_metadata or {})
        existing = await self.get_session(session_id)
        if existing:
            existing.cycle_count = int(cycle_count)
            existing.sampling_rate_hz = float(sampling_rate_hz)
            existing.source_device = str(source_device)[:50]
            existing.tracker = str(tracker)[:50]
            existing.raw_schema_version = str(raw_schema_version)[:20]
            existing.storage_root = str(storage_root)
            existing.processing_status = str(processing_status)[:30]
            existing.error_message = error_message
            existing.client_metadata = clean_metadata
            await self.session.flush()
            return existing

        new_session = CoverTestSessionModel(
            session_id=session_id,
            cycle_count=int(cycle_count),
            sampling_rate_hz=float(sampling_rate_hz),
            source_device=str(source_device)[:50],
            tracker=str(tracker)[:50],
            raw_schema_version=str(raw_schema_version)[:20],
            storage_root=str(storage_root),
            processing_status=str(processing_status)[:30],
            client_metadata=clean_metadata,
            error_message=error_message,
        )
        self.session.add(new_session)
        await self.session.flush()
        return new_session

    async def update_session_status(
        self,
        session_id: uuid.UUID,
        processing_status: str,
        error_message: Optional[str] = None,
    ) -> None:
        stmt = (
            update(CoverTestSessionModel)
            .where(CoverTestSessionModel.session_id == session_id)
            .values(
                processing_status=str(processing_status)[:30],
                error_message=error_message,
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def upsert_cycles(
        self,
        session_id: uuid.UUID,
        cycles_data: List[Dict[str, Any]],
    ) -> List[CoverTestCycleModel]:
        saved_cycles = []
        # Batch select all existing cycles for this session (Eliminates N+1 queries)
        stmt = select(CoverTestCycleModel).where(CoverTestCycleModel.session_id == session_id)
        result = await self.session.execute(stmt)
        existing_cycles_map = {c.cycle_number: c for c in result.scalars().all()}

        for c in cycles_data:
            c_num = int(c["cycle_number"])
            existing = existing_cycles_map.get(c_num)

            s_count = max(0, int(c.get("sample_count", 0)))
            v_count = max(0, min(s_count, int(c.get("valid_sample_count", 0))))
            v_ratio = max(0.0, min(1.0, float(c.get("valid_ratio", 0.0))))
            mean_q = max(0.0, min(1.0, float(c.get("mean_tracking_quality", 0.0))))
            raw_dur = c.get("duration_ms")
            duration_val = int(round(float(raw_dur))) if raw_dur is not None else None

            if existing:
                existing.covered_eye = str(c["covered_eye"])[:20]
                existing.tracked_eye = str(c["tracked_eye"])[:20]
                existing.sample_count = s_count
                existing.valid_sample_count = v_count
                existing.valid_ratio = v_ratio
                existing.mean_tracking_quality = mean_q
                existing.cycle_status = str(c.get("cycle_status", "COMPLETED"))[:20]
                existing.raw_storage_path = str(c["raw_storage_path"])
                existing.duration_ms = duration_val
                saved_cycles.append(existing)
            else:
                cid = c.get("cycle_id")
                if cid is None or not isinstance(cid, uuid.UUID):
                    try:
                        cid = uuid.UUID(str(cid)) if cid else uuid.uuid4()
                    except Exception:
                        cid = uuid.uuid4()

                new_cycle = CoverTestCycleModel(
                    cycle_id=cid,
                    session_id=session_id,
                    cycle_number=c_num,
                    covered_eye=str(c["covered_eye"])[:20],
                    tracked_eye=str(c["tracked_eye"])[:20],
                    sample_count=s_count,
                    valid_sample_count=v_count,
                    valid_ratio=v_ratio,
                    mean_tracking_quality=mean_q,
                    cycle_status=str(c.get("cycle_status", "COMPLETED"))[:20],
                    raw_storage_path=str(c["raw_storage_path"]),
                    duration_ms=duration_val,
                )
                self.session.add(new_cycle)
                saved_cycles.append(new_cycle)

        await self.session.flush()
        return saved_cycles

    async def upsert_images(
        self,
        session_id: uuid.UUID,
        images_data: List[Dict[str, Any]],
    ) -> List[CoverTestImageModel]:
        saved_images = []
        # Batch select all existing images for this session (Eliminates N+1 queries)
        stmt = select(CoverTestImageModel).where(CoverTestImageModel.session_id == session_id)
        result = await self.session.execute(stmt)
        existing_images_map = {(img.cycle_number, img.eye): img for img in result.scalars().all()}

        for img in images_data:
            c_id = img["cycle_id"]
            if not isinstance(c_id, uuid.UUID):
                c_id = uuid.UUID(str(c_id))
            eye = str(img["eye"]).strip().upper()
            c_num = int(img["cycle_number"])
            existing = existing_images_map.get((c_num, eye))

            clean_crop = sanitize_json_data(img.get("crop_region"))

            if existing:
                existing.cycle_id = c_id
                existing.capture_event = str(img["capture_event"])[:30]
                existing.timestamp = img["timestamp"]
                existing.storage_path = str(img["storage_path"])
                existing.mime_type = str(img.get("mime_type", "image/jpeg"))[:30]
                existing.width = int(img["width"])
                existing.height = int(img["height"])
                existing.file_size = int(img["file_size"])
                existing.upload_status = str(img.get("upload_status", "UPLOADED"))[:20]
                existing.crop_region = clean_crop
                saved_images.append(existing)
            else:
                img_id = img.get("image_id")
                if img_id is None or not isinstance(img_id, uuid.UUID):
                    try:
                        img_id = uuid.UUID(str(img_id)) if img_id else uuid.uuid4()
                    except Exception:
                        img_id = uuid.uuid4()

                new_img = CoverTestImageModel(
                    image_id=img_id,
                    session_id=session_id,
                    cycle_id=c_id,
                    cycle_number=c_num,
                    eye=eye,
                    capture_event=str(img["capture_event"])[:30],
                    timestamp=img["timestamp"],
                    storage_path=str(img["storage_path"]),
                    mime_type=str(img.get("mime_type", "image/jpeg"))[:30],
                    width=int(img["width"]),
                    height=int(img["height"]),
                    file_size=int(img["file_size"]),
                    upload_status=str(img.get("upload_status", "UPLOADED"))[:20],
                    crop_region=clean_crop,
                )
                self.session.add(new_img)
                saved_images.append(new_img)

        await self.session.flush()
        return saved_images

    async def upsert_result(
        self,
        session_id: uuid.UUID,
        result_data: Dict[str, Any],
    ) -> CoverTestResultModel:
        model_name = str(result_data.get("model_name", "remicare-fps-10-15"))[:100]
        model_version = str(result_data.get("model_version", "10-15fps-v1.1.0"))[:50]
        model_source = str(result_data.get("model_source", "fps_10_15_model"))[:50]
        prediction_val = result_data.get("prediction", "NORMAL")
        if prediction_val not in ("NORMAL", "STRABISMUS", "INCONCLUSIVE"):
            prediction_val = "INCONCLUSIVE"

        stmt = select(CoverTestResultModel).where(
            CoverTestResultModel.session_id == session_id,
            CoverTestResultModel.model_name == model_name,
            CoverTestResultModel.model_version == model_version,
        )
        result = await self.session.execute(stmt)
        existing = result.scalars().first()

        clean_probs = sanitize_json_data(result_data.get("class_probabilities", {}))
        clean_features = sanitize_json_data(result_data.get("features_snapshot"))
        clean_comparisons = sanitize_json_data(result_data.get("comparison_models", []))
        raw_conf = result_data.get("confidence")
        conf_val = float(raw_conf) if raw_conf is not None and not (math.isnan(float(raw_conf)) or math.isinf(float(raw_conf))) else None
        image_url_val = result_data.get("image_url")
        cloudinary_pid_val = result_data.get("cloudinary_public_id")

        if existing:
            existing.model_source = model_source
            existing.feature_schema_version = str(result_data.get("feature_schema_version", "shared-v1.0.0"))[:50]
            existing.status = str(result_data.get("status", "COMPLETED"))[:30]
            existing.input_compatible = bool(result_data.get("input_compatible", True))
            existing.prediction = prediction_val
            existing.class_probabilities = clean_probs
            existing.confidence = conf_val
            if image_url_val is not None:
                existing.image_url = str(image_url_val)
            if cloudinary_pid_val is not None:
                existing.cloudinary_public_id = str(cloudinary_pid_val)[:255]
            existing.domain_shift_warning = bool(result_data.get("domain_shift_warning", False))
            existing.features_snapshot = clean_features
            existing.comparison_models = clean_comparisons
            existing.notice = str(
                result_data.get(
                    "notice",
                    "Screening result derived from 10-15 FPS model consensus aggregation.",
                )
            )
            await self.session.flush()
            return existing

        res_id = result_data.get("result_id")
        if res_id is None or not isinstance(res_id, uuid.UUID):
            try:
                res_id = uuid.UUID(str(res_id)) if res_id else uuid.uuid4()
            except Exception:
                res_id = uuid.uuid4()

        new_result = CoverTestResultModel(
            result_id=res_id,
            session_id=session_id,
            model_name=model_name,
            model_version=model_version,
            model_source=model_source,
            feature_schema_version=str(result_data.get("feature_schema_version", "shared-v1.0.0"))[:50],
            status=str(result_data.get("status", "COMPLETED"))[:30],
            input_compatible=bool(result_data.get("input_compatible", True)),
            prediction=prediction_val,
            class_probabilities=clean_probs,
            confidence=conf_val,
            image_url=str(image_url_val) if image_url_val is not None else None,
            cloudinary_public_id=str(cloudinary_pid_val)[:255] if cloudinary_pid_val is not None else None,
            domain_shift_warning=bool(result_data.get("domain_shift_warning", False)),
            features_snapshot=clean_features,
            comparison_models=clean_comparisons,
            notice=str(
                result_data.get(
                    "notice",
                    "Screening result derived from 10-15 FPS model consensus aggregation.",
                )
            ),
        )
        self.session.add(new_result)
        await self.session.flush()
        return new_result
