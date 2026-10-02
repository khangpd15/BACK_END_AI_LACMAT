"""SQLAlchemy model for Strabismus Screening metadata."""

import uuid
from typing import Optional
from sqlalchemy import Boolean, DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.models.cover_test_session import GUID


class StrabismusScreeningModel(Base):
    """Database model for strabismus_screenings.
    Stores metadata only: strictly no image binary, no image URL, no base64.
    """
    __tablename__ = "strabismus_screenings"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    strabismus_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[float] = mapped_column(Float, default=0.20, nullable=False)
    model_version: Mapped[str] = mapped_column(
        String(100), default="remicare-bilateral-resnet18-v1", nullable=False
    )
    inference_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_saved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failure_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
