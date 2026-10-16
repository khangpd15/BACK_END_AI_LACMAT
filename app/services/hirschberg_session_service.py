"""Hirschberg Session Service.

Handles persistence of Hirschberg screening sessions:
- Local storage under data/hirschberg/sessions/<session_id>/ (metadata.json, result.json).
- Database persistence via HirschbergRepository into hirschberg_sessions table.
- Customer privacy: strictly no raw biometric images stored on disk or DB.
"""

from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import PROJECT_ROOT
from app.db.repositories.hirschberg_repository import HirschbergRepository

logger = logging.getLogger("remicare.hirschberg_session_service")

# Regular expression for valid session IDs (alphanumeric, dash, underscore, dot)
SESSION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def sanitize_floats(obj: Any) -> Any:
    """Recursively replace non-finite float numbers with strings or None for JSON serialization."""
    if isinstance(obj, list):
        return [sanitize_floats(item) for item in obj]
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    return obj


class HirschbergSessionService:
    """Service for managing Hirschberg screening sessions."""

    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
        self.repo = HirschbergRepository(db_session) if db_session else None
        self.sessions_base_dir = os.path.join(str(PROJECT_ROOT), "data", "hirschberg", "sessions")
        os.makedirs(self.sessions_base_dir, exist_ok=True)

    def persist_session_local(
        self,
        session_id: str,
        filename: str,
        orientation: str,
        input_kind: str,
        fixation: str,
        prediction_result: Dict[str, Any],
        client_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persists metadata.json and result.json to data/hirschberg/sessions/<session_id>/."""
        target_dir = os.path.join(self.sessions_base_dir, session_id)
        if os.path.exists(target_dir):
            unique_suffix = f"{int(time.time() * 1000)}"
            target_dir = os.path.join(self.sessions_base_dir, f"{session_id}_{unique_suffix}")

        os.makedirs(target_dir, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat()
        metadata_payload = {
            "sessionId": session_id,
            "testType": "hirschberg",
            "createdAt": created_at,
            "filename": filename,
            "orientation": orientation,
            "inputKind": input_kind,
            "fixation": fixation,
            "clientMetadata": client_metadata or {},
        }

        # Clean non-finite floats
        safe_metadata = sanitize_floats(metadata_payload)
        safe_result = sanitize_floats(prediction_result)

        metadata_path = os.path.join(target_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(safe_metadata, f, indent=2, ensure_ascii=False)

        result_path = os.path.join(target_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(safe_result, f, indent=2, ensure_ascii=False)

        rel_path = os.path.relpath(target_dir, str(PROJECT_ROOT)).replace("\\", "/")
        logger.info(
            "[HirschbergSessionLocalSaved] sessionId=%s, path=%s",
            session_id,
            rel_path,
        )
        return rel_path

    async def persist_session(
        self,
        session_id: Optional[str],
        filename: str,
        orientation: str,
        input_kind: str,
        fixation: str,
        prediction_result: Dict[str, Any],
        client_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Saves session locally and in database."""
        sid = (session_id or "").strip()
        if not sid:
            sid = f"hirschberg_{uuid.uuid4()}"
        elif not SESSION_ID_REGEX.match(sid):
            sid = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", sid)

        # 1. Save local session files
        rel_path = self.persist_session_local(
            session_id=sid,
            filename=filename,
            orientation=orientation,
            input_kind=input_kind,
            fixation=fixation,
            prediction_result=prediction_result,
            client_metadata=client_metadata,
        )

        # 2. Save metadata in DB if session is available
        if self.repo:
            try:
                confidence_val = prediction_result.get("confidence")
                if confidence_val is not None:
                    try:
                        confidence_val = float(confidence_val)
                    except (ValueError, TypeError):
                        confidence_val = None

                latency_val = prediction_result.get("latencyMs")
                if latency_val is not None:
                    try:
                        latency_val = float(latency_val)
                    except (ValueError, TypeError):
                        latency_val = None

                await self.repo.save_session(
                    session_id=sid,
                    status=prediction_result.get("status", "UNKNOWN"),
                    predicted_class=prediction_result.get("predictedClass", "UNKNOWN"),
                    confidence=confidence_val,
                    reason=prediction_result.get("reason"),
                    message_vi=prediction_result.get("message"),
                    model_id=prediction_result.get("modelId"),
                    feature_contract=prediction_result.get("featureContract"),
                    orientation=prediction_result.get("orientation", orientation),
                    input_kind=prediction_result.get("inputKind", input_kind),
                    measurements=sanitize_floats(prediction_result.get("measurements")),
                    quality=sanitize_floats(prediction_result.get("quality")),
                    uncalibrated_scores=sanitize_floats(prediction_result.get("uncalibratedModelScores")),
                    latency_ms=latency_val,
                    session_path=rel_path,
                    metadata_json={
                        "filename": filename,
                        "fixation": fixation,
                        "clientMetadata": client_metadata or {},
                    },
                )
            except Exception as e:
                logger.error("[HirschbergSessionDBSaveError] Error saving to DB: %s", e, exc_info=True)

        return {
            "sessionId": sid,
            "sessionPath": rel_path,
            "saved": True,
        }
