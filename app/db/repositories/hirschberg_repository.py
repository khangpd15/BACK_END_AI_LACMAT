"""Repository for persisting Hirschberg screening sessions and results."""

import logging
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.hirschberg_session import HirschbergSessionModel

logger = logging.getLogger("remicare.db.hirschberg")


class HirschbergRepository:
    """Handles persistence of Hirschberg screening sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_session_by_id(self, session_id: str) -> Optional[HirschbergSessionModel]:
        stmt = select(HirschbergSessionModel).where(HirschbergSessionModel.session_id == session_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[HirschbergSessionModel]:
        stmt = (
            select(HirschbergSessionModel)
            .order_by(HirschbergSessionModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def save_session(
        self,
        session_id: str,
        status: str,
        predicted_class: str,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
        message_vi: Optional[str] = None,
        model_id: Optional[str] = None,
        feature_contract: Optional[str] = None,
        orientation: Optional[str] = None,
        input_kind: Optional[str] = None,
        measurements: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
        uncalibrated_scores: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        session_path: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        record_id: Optional[uuid.UUID] = None,
    ) -> HirschbergSessionModel:
        """Saves or updates a Hirschberg session in the database."""
        existing = await self.get_session_by_id(session_id)
        if existing:
            existing.status = status
            existing.predicted_class = predicted_class
            existing.confidence = confidence
            existing.reason = reason
            existing.message_vi = message_vi
            existing.model_id = model_id
            existing.feature_contract = feature_contract
            existing.orientation = orientation
            existing.input_kind = input_kind
            existing.measurements = measurements
            existing.quality = quality
            existing.uncalibrated_scores = uncalibrated_scores
            existing.latency_ms = latency_ms
            existing.session_path = session_path
            existing.metadata_json = metadata_json
            record = existing
        else:
            record = HirschbergSessionModel(
                id=record_id or uuid.uuid4(),
                session_id=session_id,
                status=status,
                predicted_class=predicted_class,
                confidence=confidence,
                reason=reason,
                message_vi=message_vi,
                model_id=model_id,
                feature_contract=feature_contract,
                orientation=orientation,
                input_kind=input_kind,
                measurements=measurements,
                quality=quality,
                uncalibrated_scores=uncalibrated_scores,
                latency_ms=latency_ms,
                session_path=session_path,
                metadata_json=metadata_json,
            )
            self.session.add(record)

        await self.session.commit()
        await self.session.refresh(record)
        logger.info(
            "[HirschbergRepository] Saved session: id=%s, sessionId=%s, status=%s, class=%s",
            record.id,
            record.session_id,
            record.status,
            record.predicted_class,
        )
        return record
