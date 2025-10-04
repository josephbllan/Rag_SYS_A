"""
Domain Layer - Core business entities and types
Contains type definitions, enums, and domain models
"""

from .types import *
from .enums import *
from .models import *
from .value_objects import *

__all__ = [
    # Types
    'VectorType', 'EmbeddingDimension', 'SearchType', 'ModelType',
    # Enums
    'QueryType', 'ModelName', 'VectorBackend', 'PatternType', 'ShapeType', 'SizeType', 'BrandType',
    # Models
    'ImageMetadata', 'SearchQuery', 'SearchResult', 'EmbeddingResult',
    # Value Objects
    'QueryIntent', 'SearchFilters'
]

