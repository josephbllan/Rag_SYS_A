"""
Base Controller
Common functionality for all controllers
"""
from typing import Any, Dict, Optional
from datetime import datetime
import time
import logging


class BaseController:
    """
    Base controller implementing common patterns
    
    Design Patterns:
    - Template Method: Common request handling flow
    - Dependency Injection: Services injected via constructor
    """
    
    def __init__(self, service: Optional[Any] = None):
        """
        Initialize base controller
        
        Args:
            service: Optional service instance
        """
        self._service = service
        self._logger = logging.getLogger(self.__class__.__name__)
    
    async def handle_request(
        self,
        request_data: Any,
        handler_func,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Template method for handling requests
        
        Args:
            request_data: Request data
            handler_func: Function to handle the request
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        request_id = kwargs.get('request_id', 'unknown')
        
        try:
            self._logger.info(f"[{request_id}] Processing request")
            
            # Execute handler
            result = await handler_func(request_data, **kwargs)
            
            # Build success response
            execution_time = time.time() - start_time
            response = self._build_success_response(
                data=result,
                execution_time=execution_time,
                request_id=request_id
            )
            
            self._logger.info(
                f"[{request_id}] Request completed in {execution_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"[{request_id}] Request failed: {e}")
            execution_time = time.time() - start_time
            return self._build_error_response(
                error=e,
                execution_time=execution_time,
                request_id=request_id
            )
    
    def _build_success_response(
        self,
        data: Any,
        execution_time: float,
        request_id: str
    ) -> Dict[str, Any]:
        """Build standardized success response"""
        return {
            'success': True,
            'data': data,
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'execution_time': execution_time,
                'version': '2.0.0'
            }
        }
    
    def _build_error_response(
        self,
        error: Exception,
        execution_time: float,
        request_id: str
    ) -> Dict[str, Any]:
        """Build standardized error response"""
        return {
            'success': False,
            'error': {
                'code': error.__class__.__name__,
                'message': str(error),
                'type': 'application_error'
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'execution_time': execution_time,
                'version': '2.0.0'
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(service={self._service})"

