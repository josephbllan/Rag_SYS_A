from fastapi import APIRouter
from .search import router as search_router
from .system import router as system_router
from .auth import router as auth_router

router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["authentication"])
router.include_router(search_router, prefix="/search", tags=["search"])
router.include_router(system_router, tags=["system"])
