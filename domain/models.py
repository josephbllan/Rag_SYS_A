"""
Domain Models with Pydantic validation
Strongly-typed, validated domain models for the RAG system
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, constr
from .enums import (
    QueryType, PatternType, ShapeType, SizeType, BrandType,
    ModelName, VectorBackend
)


class ImageMetadata(BaseModel):
    """Validated image metadata model"""
    filename: constr(min_length=1, max_length=255)
    original_path: str
    pattern: PatternType = PatternType.OTHER
    shape: ShapeType = ShapeType.ROUND
    size: SizeType = SizeType.MEDIUM
    brand: BrandType = BrandType.OTHER
    color: Optional[str] = None
    style: Optional[str] = None
    image_width: Optional[int] = Field(None, gt=0, le=10000)
    image_height: Optional[int] = Field(None, gt=0, le=10000)
    file_size: Optional[int] = Field(None, gt=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('filename')
    def validate_filename(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError('Filename cannot be empty or whitespace')
        return v.strip()
    
    @validator('image_width', 'image_height')
    def validate_dimensions(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 10000):
            raise ValueError('Image dimensions must be between 0 and 10000')
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = False


class SearchFilters(BaseModel):
    """Validated search filters"""
    brand: Optional[Union[BrandType, List[BrandType]]] = None
    pattern: Optional[Union[PatternType, List[PatternType]]] = None
    shape: Optional[Union[ShapeType, List[ShapeType]]] = None
    size: Optional[Union[SizeType, List[SizeType]]] = None
    color: Optional[Union[str, List[str]]] = None
    style: Optional[Union[str, List[str]]] = None
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_results: Optional[int] = Field(None, gt=0, le=1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.dict().items() if v is not None}
    
    class Config:
        use_enum_values = True


class SearchQuery(BaseModel):
    """Validated search query model"""
    query: str = Field(..., min_length=1, max_length=500)
    query_type: QueryType = QueryType.TEXT
    filters: Optional[SearchFilters] = None
    limit: int = Field(10, gt=0, le=100)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    image_path: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    @validator('query')
    def sanitize_query(cls, v: str) -> str:
        """Remove potentially malicious content"""
        import re
        malicious_patterns = [
            r'<script', r'javascript:', r'on\w+=', 
            r'eval\(', r'exec\('
        ]
        for pattern in malicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially malicious content')
        return v.strip()
    
    @validator('image_path')
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            import os
            # Basic path validation
            if not v or '..' in v:
                raise ValueError('Invalid image path')
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class SearchResultItem(BaseModel):
    """Validated search result item"""
    vector_id: str
    filename: str
    original_path: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., gt=0)
    metadata: ImageMetadata
    scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class SearchResponse(BaseModel):
    """Validated search response"""
    results: List[SearchResultItem]
    total_count: int = Field(..., ge=0)
    query_type: QueryType
    execution_time: float = Field(..., ge=0.0)
    query_metadata: Dict[str, Any] = Field(default_factory=dict)
    filters_applied: Optional[SearchFilters] = None
    
    class Config:
        use_enum_values = True


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: ModelName
    device: str = "cpu"
    batch_size: int = Field(32, gt=0, le=128)
    dimension: int = Field(512, gt=0)
    cache_enabled: bool = True
    
    @validator('device')
    def validate_device(cls, v: str) -> str:
        if v not in ['cpu', 'cuda', 'mps']:
            raise ValueError('Device must be cpu, cuda, or mps')
        return v
    
    class Config:
        use_enum_values = True


class VectorDatabaseConfig(BaseModel):
    """Vector database configuration"""
    backend: VectorBackend
    collection_name: str = "shoe_images"
    dimension: int = Field(512, gt=0)
    index_type: str = "IVFFlat"
    nlist: int = Field(1000, gt=0)
    nprobe: int = Field(10, gt=0)
    distance_metric: str = "cosine"
    
    class Config:
        use_enum_values = True


class SystemHealth(BaseModel):
    """System health status"""
    status: str = "healthy"
    vector_db_status: str = "connected"
    models_loaded: List[str] = Field(default_factory=list)
    total_vectors: int = Field(0, ge=0)
    uptime_seconds: float = Field(0.0, ge=0.0)
    last_check: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class IndexingResult(BaseModel):
    """Result of indexing operation"""
    total_processed: int = Field(0, ge=0)
    successful: int = Field(0, ge=0)
    failed: int = Field(0, ge=0)
    execution_time: float = Field(0.0, ge=0.0)
    errors: List[str] = Field(default_factory=list)
    
    @validator('successful', 'failed')
    def validate_counts(cls, v: int, values: Dict[str, Any]) -> int:
        if 'total_processed' in values:
            if v > values['total_processed']:
                raise ValueError('Count cannot exceed total processed')
        return v
    
    class Config:
        validate_assignment = True

