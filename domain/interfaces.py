"""
Interfaces (Protocols) for dependency inversion
Defines contracts that implementations must follow
"""
from typing import Protocol, List, Dict, Any, Optional, Generic, runtime_checkable
from abc import abstractmethod
from .types import VectorType, T
from .models import SearchQuery, SearchResultItem, ImageMetadata
from .value_objects import QueryIntent, EmbeddingResult


@runtime_checkable
class IEmbeddingModel(Protocol):
    """Interface for embedding models"""
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        ...
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        ...
    
    def encode(self, input_data: Any) -> VectorType:
        """Encode input to embedding vector"""
        ...
    
    def encode_batch(self, inputs: List[Any], batch_size: int = 32) -> List[VectorType]:
        """Encode batch of inputs"""
        ...


@runtime_checkable
class IVectorDatabase(Protocol):
    """Interface for vector database operations"""
    
    def add_vectors(
        self, 
        vectors: VectorType, 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to database"""
        ...
    
    def search(
        self, 
        query_vector: VectorType, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Search for similar vectors"""
        ...
    
    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        ...
    
    def rebuild_index(self) -> None:
        """Rebuild the vector index"""
        ...


@runtime_checkable
class ISearchStrategy(Protocol):
    """Interface for search strategies"""
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        ...
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute search strategy"""
        ...
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate query for this strategy"""
        ...


@runtime_checkable
class IQueryProcessor(Protocol):
    """Interface for query processing"""
    
    def process(self, query: str) -> QueryIntent:
        """Process query and extract intent"""
        ...
    
    def validate(self, query: str) -> tuple[bool, str]:
        """Validate query"""
        ...
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations"""
        ...


@runtime_checkable
class IRepository(Protocol, Generic[T]):
    """Generic repository interface"""
    
    def add(self, entity: T) -> T:
        """Add entity"""
        ...
    
    def get_by_id(self, id: Any) -> Optional[T]:
        """Get entity by ID"""
        ...
    
    def update(self, entity: T) -> T:
        """Update entity"""
        ...
    
    def delete(self, id: Any) -> bool:
        """Delete entity"""
        ...
    
    def find(self, criteria: Dict[str, Any]) -> List[T]:
        """Find entities matching criteria"""
        ...
    
    def count(self) -> int:
        """Count total entities"""
        ...


@runtime_checkable
class ICache(Protocol):
    """Interface for caching operations"""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...
    
    def clear(self) -> None:
        """Clear all cache"""
        ...
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        ...


@runtime_checkable
class ILogger(Protocol):
    """Interface for logging operations"""
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message"""
        ...
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        ...
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        ...
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message"""
        ...
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message"""
        ...


@runtime_checkable
class IEventPublisher(Protocol):
    """Interface for event publishing"""
    
    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        ...
    
    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to an event"""
        ...
    
    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """Unsubscribe from an event"""
        ...


@runtime_checkable
class IValidator(Protocol, Generic[T]):
    """Interface for validation"""
    
    def validate(self, data: T) -> tuple[bool, List[str]]:
        """Validate data and return result with errors"""
        ...
    
    def is_valid(self, data: T) -> bool:
        """Check if data is valid"""
        ...


@runtime_checkable
class ISerializer(Protocol, Generic[T]):
    """Interface for serialization"""
    
    def serialize(self, obj: T) -> str:
        """Serialize object to string"""
        ...
    
    def deserialize(self, data: str) -> T:
        """Deserialize string to object"""
        ...


@runtime_checkable
class IMetadataExtractor(Protocol):
    """Interface for metadata extraction"""
    
    def extract(self, file_path: str) -> ImageMetadata:
        """Extract metadata from file"""
        ...
    
    def extract_batch(self, file_paths: List[str]) -> List[ImageMetadata]:
        """Extract metadata from multiple files"""
        ...


@runtime_checkable
class IScorer(Protocol):
    """Interface for scoring algorithms"""
    
    def calculate_score(
        self, 
        query_vector: VectorType, 
        result_vector: VectorType
    ) -> float:
        """Calculate similarity score"""
        ...
    
    def calculate_hybrid_score(self, scores: Dict[str, float]) -> float:
        """Calculate hybrid score from multiple scores"""
        ...


@runtime_checkable
class IModelFactory(Protocol):
    """Interface for model factory"""
    
    def create_model(self, model_type: str, **kwargs: Any) -> IEmbeddingModel:
        """Create embedding model"""
        ...
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        ...


@runtime_checkable
class ISearchService(Protocol):
    """Interface for search service"""
    
    def search(self, query: SearchQuery) -> List[SearchResultItem]:
        """Execute search"""
        ...
    
    def search_text(self, query_text: str, **kwargs: Any) -> List[SearchResultItem]:
        """Text search"""
        ...
    
    def search_image(self, image_path: str, **kwargs: Any) -> List[SearchResultItem]:
        """Image search"""
        ...
    
    def search_hybrid(
        self, 
        query_text: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs: Any
    ) -> List[SearchResultItem]:
        """Hybrid search"""
        ...

