"""
System Routes (V1)
Health checks and system information
"""
from fastapi import APIRouter, Depends

from api.controllers.health_controller import HealthController

# Router
router = APIRouter()


# Dependency: Get controller
def get_health_controller() -> HealthController:
    """Get health controller instance"""
    return HealthController()


# Routes
@router.get("/health")
async def health_check(
    controller: HealthController = Depends(get_health_controller)
):
    """
    Basic health check endpoint
    
    Returns service status and basic information.
    """
    result = await controller.check_health()
    return result


@router.get("/health/detailed")
async def detailed_health_check(
    controller: HealthController = Depends(get_health_controller)
):
    """
    Detailed health check with system metrics
    
    Returns service status plus CPU, memory, and disk metrics.
    """
    result = await controller.check_detailed_health()
    return result

