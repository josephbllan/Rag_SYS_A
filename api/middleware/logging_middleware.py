"""
Logging Middleware
Logs all requests and responses
"""
import time
import logging
from fastapi import Request, FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses
    
    Features:
    - Request logging
    - Response time tracking
    - Status code logging
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and log details
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
            
        Returns:
            Response
        """
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"-> {response.status_code} ({duration:.3f}s)"
            )
            
            # Add custom headers
            response.headers["X-Process-Time"] = f"{duration:.3f}"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {duration:.3f}s - {str(e)}"
            )
            raise


def setup_logging_middleware(app: FastAPI):
    """
    Setup logging middleware for FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(LoggingMiddleware)
    logger.info("Logging middleware configured")

