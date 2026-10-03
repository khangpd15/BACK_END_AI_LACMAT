"""Schemas for versioned research-only geometry measurement requests."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchEligibility(BaseModel):
    model_config = ConfigDict(extra="allow")

    consent: bool = Field(False, description="Separate research/measurement consent.")
    ageYears: Optional[int] = Field(None, ge=0, le=120)
    redFlag: bool = Field(False, description="True when screening exclusion red flags are present.")


class ResearchCoverSample(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: Optional[float] = None
    realTimestampMs: Optional[float] = None
    t: Optional[float] = None
    phase: str
    iris_x: Optional[float] = None
    eye_corner: Optional[Dict[str, Any]] = None
    visibility: Optional[Dict[str, Any]] = None
    trackingQuality: Optional[float] = Field(None, ge=0.0, le=1.0)
    frameValid: Optional[bool] = None

    @field_validator("iris_x")
    @classmethod
    def validate_iris_x(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value < 0.0 or value > 1.0:
            raise ValueError("iris_x must be normalized to [0, 1]")
        return value


class ResearchMeasurementRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schemaVersion: str
    featureVersion: str
    protocolVersion: str = "research-measurement-v1"
    configVersion: str = "TODO_PILOT"
    testType: Literal["HIRSCHBERG", "COVER"]
    sessionId: str
    requestId: Optional[str] = None
    distance_bucket: str
    eligibility: ResearchEligibility
    quality: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # HIRSCHBERG payload.
    imageDataUrl: Optional[str] = None
    imageBase64: Optional[str] = None

    # COVER payload.
    samples: Optional[List[ResearchCoverSample]] = None


class ResearchMeasurementResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    requestId: str
    sessionId: str
    testType: str
    status: str
    result: str
    reasonCodes: List[str] = Field(default_factory=list)
    measurements: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    versions: Dict[str, Any] = Field(default_factory=dict)
    experimental: bool = True
