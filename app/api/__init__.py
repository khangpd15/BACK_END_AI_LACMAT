"""API routes package for Hirschberg & Strabismus AI Backend."""

from app.api.hirschberg import router as hirschberg_router
from app.api.research_measurement import router as research_measurement_router
from app.api.strabismus import router as strabismus_router

__all__ = [
    "hirschberg_router",
    "strabismus_router",
    "research_measurement_router",
]
