"""
API Routes
Organized by version and resource
"""
from fastapi import APIRouter
from .v1 import router as v1_router

# Main router that includes all versions
api_router = APIRouter()
api_router.include_router(v1_router, prefix="/v1", tags=["v1"])

__all__ = ['api_router']

