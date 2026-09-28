"""SQLAlchemy model for Cover Test Protocol Images."""

import uuid
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID, JSONType


class CoverTestImageModel(Base):
    """Database model for cover_test_images."""
    __tablename__ = "cover_test_images"

    image_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cycle_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_cycles.cycle_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    eye: Mapped[str] = mapped_column(String(10), nullable=False)  # 'LEFT' or 'RIGHT'
    capture_event: Mapped[str] = mapped_column(String(30), nullable=False)  # 'UNCOVER_LEFT' or 'UNCOVER_RIGHT'
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(30), default="image/jpeg", nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    upload_status: Mapped[str] = mapped_column(String(20), default="UPLOADED", nullable=False)
    crop_region: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("cycle_id", "eye", name="uq_cycle_eye"),
        UniqueConstraint("session_id", "cycle_number", "eye", name="uq_session_cycle_eye"),
    )

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="images")
    cycle: Mapped["CoverTestCycleModel"] = relationship("CoverTestCycleModel", back_populates="images")
