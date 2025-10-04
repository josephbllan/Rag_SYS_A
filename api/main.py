"""
FastAPI Application with MVC Architecture
Enhanced version with design patterns and best practices
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routes import api_router
from api.middleware import setup_logging_middleware, setup_error_handlers
from config.settings import API_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Design Patterns Applied:
    - Factory Pattern (application creation)
    - Middleware Pattern (cross-cutting concerns)
    - Dependency Injection (controllers and services)
    
    Returns:
        Configured FastAPI app
    """
    # Create app
    app = FastAPI(
        title="RAG Image Search API",
        description="Multi-modal image search system with advanced software engineering",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup middleware
    setup_logging_middleware(app)
    setup_error_handlers(app)
    
    # Include routers
    app.include_router(api_router, prefix="/api")
    
    logger.info("FastAPI application created successfully")
    
    return app


# Create app instance
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("=" * 60)
    logger.info("RAG Image Search API Starting...")
    logger.info("Version: 2.0.0")
    logger.info("Architecture: MVC with Design Patterns")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("RAG Image Search API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

