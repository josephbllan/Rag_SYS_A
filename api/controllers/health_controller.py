"""
Health Controller
System health and status checks
"""
from typing import Dict, Any
from datetime import datetime
import psutil
import platform

from .base_controller import BaseController


class HealthController(BaseController):
    """
    Health check controller
    
    Responsibilities:
    - System health monitoring
    - Component status checks
    - Performance metrics
    """
    
    def __init__(self):
        """Initialize health controller"""
        super().__init__(service=None)
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Perform basic health check
        
        Returns:
            Health status dictionary
        """
        return {
            'status': 'healthy',
            'service': 'RAG Image Search System',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0'
        }
    
    async def check_detailed_health(self) -> Dict[str, Any]:
        """
        Perform detailed health check with system metrics
        
        Returns:
            Detailed health status dictionary
        """
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'status': 'healthy',
                'service': 'RAG Image Search System',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0.0',
                'system': {
                    'platform': platform.system(),
                    'python_version': platform.python_version(),
                    'cpu_percent': cpu_percent,
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent
                    },
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'percent': disk.percent
                    }
                }
            }
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return {
                'status': 'degraded',
                'service': 'RAG Image Search System',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def __repr__(self) -> str:
        return "HealthController()"

