"""
REST API Endpoints for RAG System
FastAPI-based API for external agent integration
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path
import uuid
import shutil

from core.search_engine import SearchEngine, create_search_engine
from core.query_processor import QueryProcessor, process_natural_query
from config.settings import API_CONFIG
from config.database import get_db, SearchQuery, SearchResult, ShoeImage

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Shoe Image RAG API",
    description="Retrieval-Augmented Generation API for shoe image search and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = create_search_engine()

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: Optional[int] = Field(10, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score")

class ImageSearchRequest(BaseModel):
    image_path: str = Field(..., description="Path to reference image")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: Optional[int] = Field(10, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score")

class HybridSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Text query")
    image_path: Optional[str] = Field(None, description="Path to reference image")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: Optional[int] = Field(10, description="Maximum number of results")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query_type: str
    execution_time: float
    metadata: Dict[str, Any]

class RecommendationRequest(BaseModel):
    image_path: str = Field(..., description="Path to reference image")
    limit: Optional[int] = Field(10, description="Maximum number of recommendations")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "shoe-image-rag"}

# Search endpoints
@app.post("/search/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    """Search images using text query"""
    try:
        import time
        start_time = time.time()
        
        results = search_engine.text_to_image_search(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type="text",
            execution_time=execution_time,
            metadata={"query": request.query, "filters": request.filters}
        )
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", response_model=SearchResponse)
async def image_search(request: ImageSearchRequest):
    """Search similar images using reference image"""
    try:
        import time
        start_time = time.time()
        
        results = search_engine.image_to_image_search(
            image_path=request.image_path,
            filters=request.filters,
            limit=request.limit
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type="image",
            execution_time=execution_time,
            metadata={"image_path": request.image_path, "filters": request.filters}
        )
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """Hybrid search combining text, image, and metadata"""
    try:
        import time
        start_time = time.time()
        
        results = search_engine.hybrid_search(
            query=request.query,
            image_path=request.image_path,
            filters=request.filters,
            limit=request.limit
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type="hybrid",
            execution_time=execution_time,
            metadata={
                "query": request.query,
                "image_path": request.image_path,
                "filters": request.filters
            }
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Advanced semantic search with query expansion"""
    try:
        import time
        start_time = time.time()
        
        results = search_engine.semantic_search(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type="semantic",
            execution_time=execution_time,
            metadata={"query": request.query, "filters": request.filters}
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/natural", response_model=SearchResponse)
async def natural_language_search(query: str = Query(..., description="Natural language query")):
    """Process natural language queries"""
    try:
        import time
        start_time = time.time()
        
        # Process natural language query
        query_processor = QueryProcessor()
        intent = query_processor.process_query(query)
        
        # Execute search based on intent
        if intent.query_type == "text":
            results = search_engine.text_to_image_search(
                query=" ".join(intent.search_terms),
                filters=intent.filters,
                limit=intent.limit
            )
        elif intent.query_type == "image":
            if intent.image_path:
                results = search_engine.image_to_image_search(
                    image_path=intent.image_path,
                    filters=intent.filters,
                    limit=intent.limit
                )
            else:
                results = []
        elif intent.query_type == "metadata":
            results = search_engine.metadata_search(
                filters=intent.filters,
                limit=intent.limit
            )
        else:  # hybrid
            results = search_engine.hybrid_search(
                query=" ".join(intent.search_terms),
                image_path=intent.image_path,
                filters=intent.filters,
                limit=intent.limit
            )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type=intent.query_type,
            execution_time=execution_time,
            metadata={
                "original_query": query,
                "processed_terms": intent.search_terms,
                "filters": intent.filters,
                "intent": intent.query_type
            }
        )
        
    except Exception as e:
        logger.error(f"Natural language search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoints
@app.post("/recommendations", response_model=SearchResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations based on an image"""
    try:
        import time
        start_time = time.time()
        
        results = search_engine.get_recommendations(
            image_path=request.image_path,
            limit=request.limit
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query_type="recommendation",
            execution_time=execution_time,
            metadata={"reference_image": request.image_path}
        )
        
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Image upload endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for search"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = Path("uploads") / unique_filename
        
        # Create uploads directory
        upload_path.parent.mkdir(exist_ok=True)
        
        # Save file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": unique_filename,
            "path": str(upload_path),
            "size": upload_path.stat().st_size,
            "content_type": file.content_type
        }
        
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metadata endpoints
@app.get("/metadata/categories")
async def get_metadata_categories():
    """Get available metadata categories"""
    from config.settings import METADATA_CATEGORIES
    return METADATA_CATEGORIES

@app.get("/metadata/filters")
async def get_available_filters():
    """Get available filter options"""
    try:
        db = next(get_db())
        
        # Get unique values for each category
        filters = {}
        
        for category in ["pattern", "shape", "size", "brand"]:
            values = db.query(getattr(ShoeImage, category)).distinct().all()
            filters[category] = [value[0] for value in values if value[0]]
        
        return filters
        
    except Exception as e:
        logger.error(f"Failed to get filters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/stats")
async def get_search_stats():
    """Get search statistics"""
    try:
        stats = search_engine.get_search_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/recent")
async def get_recent_searches(limit: int = Query(10, description="Number of recent searches")):
    """Get recent search queries"""
    try:
        db = next(get_db())
        recent_searches = db.query(SearchQuery).order_by(
            SearchQuery.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": search.id,
                "query": search.query_text,
                "type": search.query_type,
                "results_count": search.results_count,
                "created_at": search.created_at.isoformat()
            }
            for search in recent_searches
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recent searches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get system status and health"""
    try:
        vector_stats = search_engine.vector_db.get_stats()
        
        return {
            "status": "healthy",
            "vector_database": vector_stats,
            "search_engine": "active",
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/system/rebuild-index")
async def rebuild_vector_index():
    """Rebuild the vector index"""
    try:
        search_engine.vector_db.rebuild_index()
        return {"status": "success", "message": "Vector index rebuilt"}
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@app.get("/export/results")
async def export_search_results(
    query: str = Query(..., description="Search query"),
    format: str = Query("json", description="Export format (json, csv)"),
    limit: int = Query(100, description="Maximum results to export")
):
    """Export search results"""
    try:
        results = search_engine.text_to_image_search(query, limit=limit)
        
        if format == "csv":
            import pandas as pd
            df = pd.DataFrame(results)
            csv_path = f"exports/results_{uuid.uuid4()}.csv"
            Path("exports").mkdir(exist_ok=True)
            df.to_csv(csv_path, index=False)
            return FileResponse(csv_path, media_type="text/csv")
        else:
            return JSONResponse(content=results)
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_CONFIG["host"], port=API_CONFIG["port"])


