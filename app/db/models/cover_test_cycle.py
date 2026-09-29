"""SQLAlchemy model for Cover Test Cycles."""

import uuid
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from app.db.models.cover_test_session import CoverTestSessionModel
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID


class CoverTestCycleModel(Base):
    """Database model for cover_test_cycles."""
    __tablename__ = "cover_test_cycles"

    cycle_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    covered_eye: Mapped[str] = mapped_column(String(20), nullable=False)  # 'LEFT', 'RIGHT', 'ALTERNATING'
    tracked_eye: Mapped[str] = mapped_column(String(20), nullable=False)  # 'RIGHT', 'LEFT', 'BOTH'
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    valid_sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    valid_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_tracking_quality: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cycle_status: Mapped[str] = mapped_column(String(20), default="COMPLETED", nullable=False)
    raw_storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "cycle_number", name="uq_session_cycle"),
    )

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="cycles")
