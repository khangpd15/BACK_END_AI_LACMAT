"""SQLAlchemy model for Cover Test AI Transfer Experiment Results."""

from app.db.models import CoverTestSessionModel
import uuid
from typing import List, Optional
from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.db.models.cover_test_session import GUID, JSONType


class CoverTestResultModel(Base):
    """Database model for cover_test_results.
    
    STRICT RESEARCH NOTICE:
    AI Result is NOT a clinical diagnosis.
    Korean baseline is NOT ground truth.
    """
    __tablename__ = "cover_test_results"

    result_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("cover_test_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), default="korean_shared_model", nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="shared-v1.0.0", nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(50), default="shared-v1.0.0", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="TRANSFER_EXPERIMENT", nullable=False)
    input_compatible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    prediction: Mapped[str] = mapped_column(String(30), nullable=False)  # 'NORMAL', 'STRABISMUS', 'INCONCLUSIVE'
    class_probabilities: Mapped[dict] = mapped_column(JSONType, nullable=False)
    domain_shift_warning: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    features_snapshot: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    comparison_models: Mapped[list] = mapped_column(JSONType, default=list, nullable=False)
    notice: Mapped[str] = mapped_column(
        Text,
        default="Research transfer experiment only - not a medical diagnosis.",
        nullable=False,
    )
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "model_name", "model_version", name="uq_session_model_version"),
    )

    # Relationships
    session: Mapped["CoverTestSessionModel"] = relationship("CoverTestSessionModel", back_populates="result")
