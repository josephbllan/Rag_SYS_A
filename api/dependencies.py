"""
Dependency Injection for FastAPI
Provides dependency injection for controllers and services
"""
from typing import Optional
from functools import lru_cache

# Import services
from services.base_service import BaseService
from services.indexing_service import IndexingService

# Import existing components
from core.search_engine import SearchEngine, create_search_engine
from core.query_processor import QueryProcessor


# Singleton instances
_search_engine_instance: Optional[SearchEngine] = None
_query_processor_instance: Optional[QueryProcessor] = None


def get_search_engine() -> SearchEngine:
    """Get or create search engine instance (Singleton)"""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = create_search_engine()
    return _search_engine_instance


def get_query_processor() -> QueryProcessor:
    """Get or create query processor instance (Singleton)"""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance


def get_indexing_service() -> IndexingService:
    """Get indexing service instance"""
    search_engine = get_search_engine()
    return IndexingService(search_engine)


# You can add more dependency providers as needed

