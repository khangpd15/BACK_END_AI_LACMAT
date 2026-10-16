"""SQLAlchemy model for Hirschberg Screening Sessions and Results."""

import uuid
from typing import Optional
from sqlalchemy import DateTime, Float, JSON, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.guid import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")


class HirschbergSessionModel(Base):
    """Database model for hirschberg_sessions.
    Stores metadata, measurements, and screening results.
    Strictly preserves customer privacy: NO raw biometric image stored.
    """
    __tablename__ = "hirschberg_sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    predicted_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message_vi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    feature_contract: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    orientation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    input_kind: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    measurements: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    quality: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    uncalibrated_scores: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    session_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
