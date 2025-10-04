"""
Base Service Class
Provides common service functionality
"""
from abc import ABC
import logging
from typing import Any, Dict, Optional


class BaseService(ABC):
    """
    Abstract base class for all services
    Provides common functionality for service layer
    """
    
    def __init__(self, service_name: str):
        """
        Initialize base service
        
        Args:
            service_name: Name of the service for logging
        """
        self._service_name = service_name
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        self._metrics: Dict[str, Any] = {}
        
    @property
    def service_name(self) -> str:
        """Get service name"""
        return self._service_name
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._is_initialized
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return self._metrics.copy()
    
    def initialize(self) -> None:
        """Initialize service"""
        if self._is_initialized:
            self._logger.warning(f"{self._service_name} already initialized")
            return
            
        self._logger.info(f"Initializing {self._service_name}...")
        self._is_initialized = True
        self._logger.info(f"{self._service_name} initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown service"""
        self._logger.info(f"Shutting down {self._service_name}...")
        self._is_initialized = False
        self._metrics.clear()
        self._logger.info(f"{self._service_name} shutdown complete")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health
        
        Returns:
            Dictionary with health status
        """
        return {
            'service': self._service_name,
            'status': 'healthy' if self._is_initialized else 'not_initialized',
            'initialized': self._is_initialized
        }
    
    def _record_metric(self, metric_name: str, value: Any) -> None:
        """Record a metric"""
        self._metrics[metric_name] = value
        self._logger.debug(f"Metric recorded: {metric_name}={value}")
    
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized before operations"""
        if not self._is_initialized:
            raise RuntimeError(
                f"{self._service_name} not initialized. "
                f"Call initialize() first."
            )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self._service_name}', "
            f"initialized={self._is_initialized})"
        )
    
    def __enter__(self):
        """Context manager entry"""
        if not self._is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False

