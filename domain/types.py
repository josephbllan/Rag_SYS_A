"""
Type Definitions for RAG System
Provides strong typing, type aliases, and type hints throughout the system
"""
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, List, Dict, Any,
    Tuple, Callable, Literal, Final, runtime_checkable
)
import numpy as np
from numpy.typing import NDArray

# Type Aliases
VectorType = NDArray[np.float32]
EmbeddingDimension = Literal[384, 512, 768, 1024, 2048]
SearchType = Literal["text", "image", "hybrid", "semantic", "metadata"]
ModelType = Literal["clip", "resnet", "sentence_transformer"]
VectorBackendType = Literal["faiss", "chroma"]
DeviceType = Literal["cpu", "cuda"]

# Generic Type Variables
T = TypeVar('T')
TModel = TypeVar('TModel')
TResult = TypeVar('TResult')
TEntity = TypeVar('TEntity')

# Constants
MAX_QUERY_LENGTH: Final[int] = 500
MIN_QUERY_LENGTH: Final[int] = 1
MAX_RESULTS: Final[int] = 100
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_EMBEDDING_DIM: Final[int] = 512

# Type Guards
def is_valid_vector(vector: Any) -> bool:
    """Type guard for valid vectors"""
    return (
        isinstance(vector, np.ndarray) and 
        vector.ndim in [1, 2] and 
        vector.dtype == np.float32
    )

def is_valid_metadata(metadata: Any) -> bool:
    """Type guard for valid metadata"""
    return isinstance(metadata, dict) and 'filename' in metadata

def is_valid_embedding_dimension(dim: int) -> bool:
    """Type guard for valid embedding dimensions"""
    return dim in [384, 512, 768, 1024, 2048]

