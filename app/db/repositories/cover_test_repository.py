"""Repository layer for Cover Test Database operations."""

import logging
import uuid
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
        existing = await self.get_session(session_id)
        if existing:
            existing.cycle_count = cycle_count
            existing.sampling_rate_hz = sampling_rate_hz
            existing.source_device = source_device
            existing.tracker = tracker
            existing.raw_schema_version = raw_schema_version
            existing.storage_root = storage_root
            existing.processing_status = processing_status
            existing.error_message = error_message
            if client_metadata is not None:
                existing.client_metadata = client_metadata
            await self.session.flush()
            return existing

        new_session = CoverTestSessionModel(
            session_id=session_id,
            cycle_count=cycle_count,
            sampling_rate_hz=sampling_rate_hz,
            source_device=source_device,
            tracker=tracker,
            raw_schema_version=raw_schema_version,
            storage_root=storage_root,
            processing_status=processing_status,
            client_metadata=client_metadata or {},
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
                processing_status=processing_status,
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
        for c in cycles_data:
            c_num = c["cycle_number"]
            stmt = select(CoverTestCycleModel).where(
                CoverTestCycleModel.session_id == session_id,
                CoverTestCycleModel.cycle_number == c_num,
            )
            result = await self.session.execute(stmt)
            existing = result.scalars().first()

            if existing:
                existing.covered_eye = c["covered_eye"]
                existing.tracked_eye = c["tracked_eye"]
                existing.sample_count = c.get("sample_count", 0)
                existing.valid_sample_count = c.get("valid_sample_count", 0)
                existing.valid_ratio = c.get("valid_ratio", 0.0)
                existing.mean_tracking_quality = c.get("mean_tracking_quality", 0.0)
                existing.cycle_status = c.get("cycle_status", "COMPLETED")
                existing.raw_storage_path = c["raw_storage_path"]
                existing.duration_ms = c.get("duration_ms")
                saved_cycles.append(existing)
            else:
                new_cycle = CoverTestCycleModel(
                    session_id=session_id,
                    cycle_number=c_num,
                    covered_eye=c["covered_eye"],
                    tracked_eye=c["tracked_eye"],
                    sample_count=c.get("sample_count", 0),
                    valid_sample_count=c.get("valid_sample_count", 0),
                    valid_ratio=c.get("valid_ratio", 0.0),
                    mean_tracking_quality=c.get("mean_tracking_quality", 0.0),
                    cycle_status=c.get("cycle_status", "COMPLETED"),
                    raw_storage_path=c["raw_storage_path"],
                    duration_ms=c.get("duration_ms"),
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
        for img in images_data:
            cycle_id = img["cycle_id"]
            eye = img["eye"]
            stmt = select(CoverTestImageModel).where(
                CoverTestImageModel.cycle_id == cycle_id,
                CoverTestImageModel.eye == eye,
            )
            result = await self.session.execute(stmt)
            existing = result.scalars().first()

            if existing:
                existing.capture_event = img["capture_event"]
                existing.timestamp = img["timestamp"]
                existing.storage_path = img["storage_path"]
                existing.mime_type = img.get("mime_type", "image/jpeg")
                existing.width = img["width"]
                existing.height = img["height"]
                existing.file_size = img["file_size"]
                existing.upload_status = img.get("upload_status", "UPLOADED")
                existing.crop_region = img.get("crop_region")
                saved_images.append(existing)
            else:
                new_img = CoverTestImageModel(
                    session_id=session_id,
                    cycle_id=cycle_id,
                    cycle_number=img["cycle_number"],
                    eye=eye,
                    capture_event=img["capture_event"],
                    timestamp=img["timestamp"],
                    storage_path=img["storage_path"],
                    mime_type=img.get("mime_type", "image/jpeg"),
                    width=img["width"],
                    height=img["height"],
                    file_size=img["file_size"],
                    upload_status=img.get("upload_status", "UPLOADED"),
                    crop_region=img.get("crop_region"),
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
        model_name = result_data.get("model_name", "korean_shared_model")
        model_version = result_data.get("model_version", "shared-v1.0.0")

        stmt = select(CoverTestResultModel).where(
            CoverTestResultModel.session_id == session_id,
            CoverTestResultModel.model_name == model_name,
            CoverTestResultModel.model_version == model_version,
        )
        result = await self.session.execute(stmt)
        existing = result.scalars().first()

        if existing:
            existing.feature_schema_version = result_data.get("feature_schema_version", "shared-v1.0.0")
            existing.status = result_data.get("status", "TRANSFER_EXPERIMENT")
            existing.input_compatible = result_data.get("input_compatible", True)
            existing.prediction = result_data["prediction"]
            existing.class_probabilities = result_data["class_probabilities"]
            existing.domain_shift_warning = result_data.get("domain_shift_warning", True)
            existing.features_snapshot = result_data.get("features_snapshot")
            existing.comparison_models = result_data.get("comparison_models", [])
            existing.notice = result_data.get(
                "notice",
                "Research transfer experiment only - not a medical diagnosis.",
            )
            await self.session.flush()
            return existing

        new_result = CoverTestResultModel(
            session_id=session_id,
            model_name=model_name,
            model_version=model_version,
            feature_schema_version=result_data.get("feature_schema_version", "shared-v1.0.0"),
            status=result_data.get("status", "TRANSFER_EXPERIMENT"),
            input_compatible=result_data.get("input_compatible", True),
            prediction=result_data["prediction"],
            class_probabilities=result_data["class_probabilities"],
            domain_shift_warning=result_data.get("domain_shift_warning", True),
            features_snapshot=result_data.get("features_snapshot"),
            comparison_models=result_data.get("comparison_models", []),
            notice=result_data.get(
                "notice",
                "Research transfer experiment only - not a medical diagnosis.",
            ),
        )
        self.session.add(new_result)
        await self.session.flush()
        return new_result
