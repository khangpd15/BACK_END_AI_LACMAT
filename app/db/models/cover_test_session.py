"""SQLAlchemy model for Cover Test Sessions."""

import uuid
from typing import List, Optional
from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, CHAR
from app.db.database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's native UUID type, otherwise uses CHAR(36).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class CoverTestSessionModel(Base):
    """Database model for cover_test_sessions."""
    __tablename__ = "cover_test_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cycle_count: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    sampling_rate_hz: Mapped[float] = mapped_column(Float, default=15.0, nullable=False)
    source_device: Mapped[str] = mapped_column(String(50), default="WEBCAM", nullable=False)
    tracker: Mapped[str] = mapped_column(String(50), default="MEDIAPIPE_IRIS", nullable=False)
    raw_schema_version: Mapped[str] = mapped_column(String(20), default="1.0.0", nullable=False)
    storage_root: Mapped[str] = mapped_column(Text, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(30), default="SESSION_CREATED", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    client_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    cycles: Mapped[List["CoverTestCycleModel"]] = relationship(
        "CoverTestCycleModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CoverTestCycleModel.cycle_number",
    )
    images: Mapped[List["CoverTestImageModel"]] = relationship(
        "CoverTestImageModel",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    result: Mapped[Optional["CoverTestResultModel"]] = relationship(
        "CoverTestResultModel",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )
