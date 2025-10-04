"""
Value Objects - Immutable domain objects
Implements value objects following DDD principles
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from .enums import QueryType, ModelName
from .types import VectorType


@dataclass(frozen=True)
class QueryIntent:
    """Immutable query intent value object"""
    query_type: QueryType
    search_terms: List[str]
    filters: Dict[str, Any]
    image_path: Optional[str] = None
    similarity_threshold: float = 0.7
    limit: int = 10
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate value object invariants"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0 and 1")
        if self.limit < 1:
            raise ValueError(f"limit must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query_type': self.query_type.value,
            'search_terms': self.search_terms,
            'filters': self.filters,
            'image_path': self.image_path,
            'similarity_threshold': self.similarity_threshold,
            'limit': self.limit,
            'confidence': self.confidence
        }


@dataclass(frozen=True)
class EmbeddingResult:
    """Immutable embedding result value object"""
    vector: VectorType
    dimension: int
    model_name: ModelName
    generated_at: datetime = field(default_factory=datetime.utcnow)
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Validate embedding result"""
        if self.dimension != len(self.vector):
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(self.vector)}")
        if self.processing_time < 0:
            raise ValueError(f"processing_time must be non-negative")


@dataclass(frozen=True)
class SearchScore:
    """Immutable search score value object"""
    visual_score: float = 0.0
    text_score: float = 0.0
    metadata_score: float = 0.0
    hybrid_score: float = 0.0
    
    def __post_init__(self):
        """Validate scores"""
        for score in [self.visual_score, self.text_score, 
                     self.metadata_score, self.hybrid_score]:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"All scores must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'visual': self.visual_score,
            'text': self.text_score,
            'metadata': self.metadata_score,
            'hybrid': self.hybrid_score
        }


@dataclass(frozen=True)
class VectorIdentifier:
    """Immutable vector identifier value object"""
    id: str
    collection: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate identifier"""
        if not self.id or not self.id.strip():
            raise ValueError("ID cannot be empty")
        if not self.collection or not self.collection.strip():
            raise ValueError("Collection cannot be empty")
    
    def to_string(self) -> str:
        """Convert to string representation"""
        return f"{self.collection}:{self.id}"


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key value object"""
    key: str
    namespace: str = "default"
    version: int = 1
    
    def __post_init__(self):
        """Validate cache key"""
        if not self.key or not self.key.strip():
            raise ValueError("Key cannot be empty")
        if self.version < 1:
            raise ValueError("Version must be positive")
    
    def to_string(self) -> str:
        """Convert to string representation"""
        return f"{self.namespace}:{self.key}:v{self.version}"


@dataclass(frozen=True)
class ModelVersion:
    """Immutable model version value object"""
    name: str
    version: str
    checksum: Optional[str] = None
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate model version"""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        if not self.version or not self.version.strip():
            raise ValueError("Version cannot be empty")

