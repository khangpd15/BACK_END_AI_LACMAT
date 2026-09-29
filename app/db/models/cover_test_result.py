"""SQLAlchemy model for Cover Test AI Transfer Experiment Results."""

import uuid
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from app.db.models.cover_test_session import CoverTestSessionModel
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID, JSONType


class CoverTestResultModel(Base):
    """Database model for cover_test_results.
    
    Stores primary screening results from 10-15 FPS model (no biometric/facial images stored to preserve customer privacy).
    Korean model results are kept for research comparison only (in comparison_models).
    """
    __tablename__ = "cover_test_results"

    result_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), default="remicare-fps-10-15", nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="10-15fps-v1.1.0", nullable=False)
    model_source: Mapped[str] = mapped_column(String(50), default="fps_10_15_model", nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(50), default="shared-v1.0.0", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="COMPLETED", nullable=False)
    input_compatible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    prediction: Mapped[str] = mapped_column(String(30), nullable=False)  # 'NORMAL', 'STRABISMUS', 'INCONCLUSIVE'
    class_probabilities: Mapped[dict] = mapped_column(JSONType, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    domain_shift_warning: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    features_snapshot: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    comparison_models: Mapped[list] = mapped_column(JSONType, default=list, nullable=False)
    notice: Mapped[str] = mapped_column(
        Text,
        default="Screening result derived from 10-15 FPS model consensus aggregation.",
        nullable=False,
    )
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "model_name", "model_version", name="uq_session_model_version"),
    )

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="result")
