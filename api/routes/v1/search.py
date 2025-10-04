"""
Search Routes (V1)
Text, image, and hybrid search endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from api.controllers.search_controller import SearchController
from api.dependencies import get_search_engine

# Router
router = APIRouter()


# Request models
class TextSearchRequest(BaseModel):
    """Text search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: int = Field(10, gt=0, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class ImageSearchRequest(BaseModel):
    """Image search request model"""
    image_path: str = Field(..., description="Path to reference image")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: int = Field(10, gt=0, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class HybridSearchRequest(BaseModel):
    """Hybrid search request model"""
    query: Optional[str] = Field(None, description="Text query")
    image_path: Optional[str] = Field(None, description="Path to reference image")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: int = Field(10, gt=0, le=100, description="Maximum number of results")


# Dependency: Get controller
def get_search_controller(
    search_engine=Depends(get_search_engine)
) -> SearchController:
    """Get search controller instance"""
    return SearchController(search_engine)


# Routes
@router.post("/text")
async def text_search(
    request: TextSearchRequest,
    controller: SearchController = Depends(get_search_controller)
):
    """
    Search images using text query
    
    This endpoint performs text-to-image search using CLIP embeddings.
    """
    try:
        result = await controller.text_search(
            query=request.query,
            filters=request.filters,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        return {'success': True, 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image")
async def image_search(
    request: ImageSearchRequest,
    controller: SearchController = Depends(get_search_controller)
):
    """
    Search images using reference image
    
    This endpoint performs image-to-image search using visual embeddings.
    """
    try:
        result = await controller.image_search(
            image_path=request.image_path,
            filters=request.filters,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        return {'success': True, 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid")
async def hybrid_search(
    request: HybridSearchRequest,
    controller: SearchController = Depends(get_search_controller)
):
    """
    Hybrid search combining text and image
    
    This endpoint performs multi-modal search using both text and image inputs.
    """
    try:
        result = await controller.hybrid_search(
            query=request.query,
            image_path=request.image_path,
            filters=request.filters,
            limit=request.limit
        )
        return {'success': True, 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

