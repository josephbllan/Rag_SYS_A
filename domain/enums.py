"""
Enumerations for type safety
Defines all enum types used across the RAG system
"""
from enum import Enum, auto


class QueryType(str, Enum):
    """Enumeration of supported query types"""
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    METADATA = "metadata"
    NATURAL_LANGUAGE = "natural_language"


class ModelName(str, Enum):
    """Enumeration of supported AI models"""
    CLIP_VIT_B32 = "ViT-B/32"
    CLIP_VIT_L14 = "ViT-L/14"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    SENTENCE_MINI_LM = "all-MiniLM-L6-v2"


class VectorBackend(str, Enum):
    """Enumeration of vector database backends"""
    FAISS = "faiss"
    CHROMA = "chroma"


class PatternType(str, Enum):
    """Shoe pattern types"""
    ZIGZAG = "zigzag"
    CIRCULAR = "circular"
    SQUARE = "square"
    DIAMOND = "diamond"
    BRAND_LOGO = "brand_logo"
    OTHER = "other"


class ShapeType(str, Enum):
    """Shoe shape types"""
    ROUND = "round"
    SQUARE = "square"
    OVAL = "oval"
    IRREGULAR = "irregular"
    ELONGATED = "elongated"


class SizeType(str, Enum):
    """Shoe size categories"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"


class BrandType(str, Enum):
    """Supported shoe brands"""
    NIKE = "nike"
    ADIDAS = "adidas"
    PUMA = "puma"
    CONVERSE = "converse"
    VANS = "vans"
    REEBOK = "reebok"
    NEW_BALANCE = "new_balance"
    ASICS = "asics"
    UNDER_ARMOUR = "under_armour"
    JORDAN = "jordan"
    OTHER = "other"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExportFormat(str, Enum):
    """Export format types"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"


class CacheStrategy(str, Enum):
    """Cache invalidation strategies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


class SearchStatus(str, Enum):
    """Search operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

