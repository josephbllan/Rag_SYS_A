"""
Error Handler Middleware
Global error handling for API
"""
import logging
from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors
    
    Args:
        request: Incoming request
        exc: Validation exception
        
    Returns:
        JSON error response
    """
    logger.warning(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            'success': False,
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': 'Request validation failed',
                'details': exc.errors()
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions
    
    Args:
        request: Incoming request
        exc: HTTP exception
        
    Returns:
        JSON error response
    """
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'success': False,
            'error': {
                'code': 'HTTP_ERROR',
                'message': exc.detail,
                'status_code': exc.status_code
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all other exceptions
    
    Args:
        request: Incoming request
        exc: Exception
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'An internal error occurred',
                'details': str(exc)
            }
        }
    )


def setup_error_handlers(app: FastAPI):
    """
    Setup error handlers for FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Error handlers configured")

