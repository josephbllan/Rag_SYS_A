"""
Abstract Base Classes for OOP hierarchy
Provides base implementations and defines contracts for inheritance
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generic
from datetime import datetime
import logging

from .types import VectorType, T
from .models import SearchQuery, SearchResultItem, ImageMetadata
from .value_objects import EmbeddingResult


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._dimension: int = 0
        self._model: Any = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_loaded = False
        
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self._model_name
    
    @property
    def device(self) -> str:
        """Get device"""
        return self._device
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def encode(self, input_data: Any) -> VectorType:
        """Encode input to embedding"""
        pass
    
    def encode_batch(
        self, 
        inputs: List[Any], 
        batch_size: int = 32
    ) -> List[VectorType]:
        """Default batch encoding implementation"""
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            self._logger.debug(f"Encoding batch {i//batch_size + 1}")
            batch_results = [self.encode(item) for item in batch]
            results.extend(batch_results)
        return results
    
    def warmup(self) -> None:
        """Warmup model by running a dummy inference"""
        try:
            self._logger.info("Warming up model...")
            dummy_input = self._get_dummy_input()
            _ = self.encode(dummy_input)
            self._logger.info("Model warmup completed")
        except Exception as e:
            self._logger.warning(f"Model warmup failed: {e}")
    
    @abstractmethod
    def _get_dummy_input(self) -> Any:
        """Get dummy input for warmup"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self._model_name}', device='{self._device}')"


class BaseVectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    def __init__(self, dimension: int, collection_name: str = "default"):
        self._dimension = dimension
        self._collection_name = collection_name
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        
    @property
    def dimension(self) -> int:
        """Get vector dimension"""
        return self._dimension
    
    @property
    def collection_name(self) -> str:
        """Get collection name"""
        return self._collection_name
    
    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._is_initialized
    
    @abstractmethod
    def add_vectors(
        self, 
        vectors: VectorType, 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> None:
        """Add vectors to database"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query_vector: VectorType, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass
    
    def rebuild_index(self) -> None:
        """Rebuild the vector index - optional operation"""
        self._logger.warning(f"{self.__class__.__name__} does not support index rebuilding")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self._dimension}, collection='{self._collection_name}')"


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self._name
    
    @abstractmethod
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute search strategy"""
        pass
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Default validation"""
        if not query.query or not query.query.strip():
            return False, "Query is empty"
        return True, "Valid"
    
    def _log_search(self, query: SearchQuery, result_count: int) -> None:
        """Log search execution"""
        self._logger.info(
            f"Strategy '{self._name}' executed: query='{query.query[:50]}...', "
            f"results={result_count}"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repositories"""
    
    def __init__(self, entity_type: type):
        self._entity_type = entity_type
        self._logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def add(self, entity: T) -> T:
        """Add entity"""
        pass
    
    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """Update entity"""
        pass
    
    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity"""
        pass
    
    @abstractmethod
    def find(self, criteria: Dict[str, Any]) -> List[T]:
        """Find entities matching criteria"""
        pass
    
    def count(self) -> int:
        """Count total entities - default implementation"""
        return len(self.find({}))
    
    def exists(self, id: Any) -> bool:
        """Check if entity exists"""
        return self.get_by_id(id) is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(entity_type={self._entity_type.__name__})"


class BaseService(ABC):
    """Abstract base class for services"""
    
    def __init__(self, service_name: str):
        self._service_name = service_name
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        
    @property
    def service_name(self) -> str:
        """Get service name"""
        return self._service_name
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._is_initialized
    
    def initialize(self) -> None:
        """Initialize service - can be overridden"""
        self._logger.info(f"Initializing {self._service_name} service...")
        self._is_initialized = True
    
    def shutdown(self) -> None:
        """Shutdown service - can be overridden"""
        self._logger.info(f"Shutting down {self._service_name} service...")
        self._is_initialized = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._service_name}', initialized={self._is_initialized})"


class BaseValidator(ABC, Generic[T]):
    """Abstract base class for validators"""
    
    def __init__(self, validator_name: str):
        self._validator_name = validator_name
        self._logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def validator_name(self) -> str:
        """Get validator name"""
        return self._validator_name
    
    @abstractmethod
    def validate(self, data: T) -> tuple[bool, List[str]]:
        """Validate data and return result with errors"""
        pass
    
    def is_valid(self, data: T) -> bool:
        """Check if data is valid"""
        valid, _ = self.validate(data)
        return valid
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._validator_name}')"


class BaseEventHandler(ABC):
    """Abstract base class for event handlers"""
    
    def __init__(self, event_type: str):
        self._event_type = event_type
        self._logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def event_type(self) -> str:
        """Get event type"""
        return self._event_type
    
    @abstractmethod
    def handle(self, event_data: Dict[str, Any]) -> None:
        """Handle event"""
        pass
    
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the event"""
        return event_type == self._event_type
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(event_type='{self._event_type}')"

