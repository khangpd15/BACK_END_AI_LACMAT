"""Repository for persisting strabismus screening metadata."""

import logging
import uuid
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.strabismus_model import StrabismusScreeningModel

logger = logging.getLogger("remicare.db.strabismus")


class StrabismusRepository:
    """Handles persistence of strabismus screening metadata without storing user images."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_screening(
        self,
        status: str,
        strabismus_probability: Optional[float],
        confidence: Optional[float],
        quality_score: Optional[float],
        threshold: float,
        model_version: str,
        inference_latency_ms: Optional[float],
        failure_reason: Optional[str] = None,
        image_saved: bool = False,
        screening_id: Optional[uuid.UUID] = None,
    ) -> StrabismusScreeningModel:
        """Persists screening metadata to strabismus_screenings table."""
        record_id = screening_id or uuid.uuid4()
        record = StrabismusScreeningModel(
            id=record_id,
            status=status,
            strabismus_probability=strabismus_probability,
            confidence=confidence,
            quality_score=quality_score,
            threshold=threshold,
            model_version=model_version,
            inference_latency_ms=inference_latency_ms,
            image_saved=image_saved,
            failure_reason=failure_reason,
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        logger.info(
            "[StrabismusRepository] Saved screening metadata: id=%s, status=%s, prob=%s, latency_ms=%s",
            record.id,
            record.status,
            record.strabismus_probability,
            record.inference_latency_ms,
        )
        return record
