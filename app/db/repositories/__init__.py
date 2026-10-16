"""Repositories package for Hirschberg & Strabismus AI Backend."""

from app.db.repositories.hirschberg_repository import HirschbergRepository
from app.db.repositories.strabismus_repository import StrabismusRepository

__all__ = [
    "HirschbergRepository",
    "StrabismusRepository",
]
