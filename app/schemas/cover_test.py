"""Pydantic schemas for Cover Test cloud persistence and API contracts."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class CoverTestCycleMetaInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    cycleNumber: int = Field(..., ge=1, description="Cycle index (1-based)")
    coveredEye: str = Field(..., description="Eye covered: LEFT, RIGHT, ALTERNATING")
    trackedEye: str = Field(..., description="Eye tracked: RIGHT, LEFT, BOTH")
    sampleCount: Optional[int] = Field(None, ge=0)
    validSampleCount: Optional[int] = Field(None, ge=0)
    validRatio: Optional[float] = Field(None, ge=0.0, le=1.0)
    meanTrackingQuality: Optional[float] = Field(None, ge=0.0, le=1.0)
    durationMs: Optional[int] = Field(None, ge=0)


class CoverTestSessionMetadataInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(..., description="Canonical UUID v4 session ID")
    cycleCount: int = Field(3, ge=1, description="Number of completed cycles")
    samplingRateHz: float = Field(15.0, gt=0, description="Acquisition sampling frequency")
    sourceDevice: str = Field("WEBCAM", description="Device type")
    tracker: str = Field("MEDIAPIPE_IRIS", description="Ocular landmark tracking engine")
    rawSchemaVersion: str = Field("1.0.0", description="Raw dataset schema version")
    clientMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    cycles: Optional[List[CoverTestCycleMetaInput]] = Field(default_factory=list)


class CoverTestSessionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    sessionId: str
    saved: bool
    processingStatus: str
    storageRoot: str
    cyclesSaved: int
    imagesSaved: int
    aiResult: Optional[Dict[str, Any]] = None
    message: str = "Cover test session persisted successfully"
