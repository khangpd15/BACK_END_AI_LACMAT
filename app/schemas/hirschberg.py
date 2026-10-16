"""Schemas for Hirschberg AI Backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HirschbergSessionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    sessionId: str
    saved: bool
    sessionPath: Optional[str] = None
    predictedClass: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    message: Optional[str] = None
