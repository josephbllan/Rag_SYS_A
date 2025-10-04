"""
Search Controller
Handles search-related business logic
"""
from typing import Dict, Any, List, Optional
import time

from .base_controller import BaseController
from core.search_engine import SearchEngine


class SearchController(BaseController):
    """
    Search Controller implementing MVC pattern
    
    Responsibilities:
    - Handle search requests
    - Coordinate with search engine
    - Format responses
    - Error handling
    """
    
    def __init__(self, search_engine: SearchEngine):
        """
        Initialize search controller
        
        Args:
            search_engine: SearchEngine instance (dependency injection)
        """
        super().__init__(service=search_engine)
        self._search_engine = search_engine
    
    async def text_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Execute text-to-image search
        
        Args:
            query: Search query text
            filters: Optional metadata filters
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results dictionary
        """
        start_time = time.time()
        
        try:
            self._logger.info(f"Text search: '{query}' (limit={limit})")
            
            # Execute search using search engine
            results = self._search_engine.text_to_image_search(
                query=query,
                filters=filters,
                limit=limit
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results
                if r.get('score', 0) >= similarity_threshold
            ]
            
            execution_time = time.time() - start_time
            
            return {
                'results': filtered_results,
                'total_count': len(filtered_results),
                'query_type': 'text',
                'execution_time': execution_time,
                'query': query,
                'filters': filters
            }
            
        except Exception as e:
            self._logger.error(f"Text search failed: {e}")
            raise
    
    async def image_search(
        self,
        image_path: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Execute image-to-image search
        
        Args:
            image_path: Path to query image
            filters: Optional metadata filters
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results dictionary
        """
        start_time = time.time()
        
        try:
            self._logger.info(f"Image search: {image_path} (limit={limit})")
            
            # Execute search
            results = self._search_engine.image_to_image_search(
                image_path=image_path,
                filters=filters,
                limit=limit
            )
            
            # Filter by similarity
            filtered_results = [
                r for r in results
                if r.get('score', 0) >= similarity_threshold
            ]
            
            execution_time = time.time() - start_time
            
            return {
                'results': filtered_results,
                'total_count': len(filtered_results),
                'query_type': 'image',
                'execution_time': execution_time,
                'image_path': image_path,
                'filters': filters
            }
            
        except Exception as e:
            self._logger.error(f"Image search failed: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: Optional[str] = None,
        image_path: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Execute hybrid search (text + image)
        
        Args:
            query: Optional text query
            image_path: Optional image path
            filters: Optional metadata filters
            limit: Maximum number of results
            
        Returns:
            Search results dictionary
        """
        start_time = time.time()
        
        try:
            self._logger.info(
                f"Hybrid search: query='{query}', image={image_path} (limit={limit})"
            )
            
            # Execute hybrid search
            results = self._search_engine.hybrid_search(
                query=query,
                image_path=image_path,
                filters=filters,
                limit=limit
            )
            
            execution_time = time.time() - start_time
            
            return {
                'results': results,
                'total_count': len(results),
                'query_type': 'hybrid',
                'execution_time': execution_time,
                'query': query,
                'image_path': image_path,
                'filters': filters
            }
            
        except Exception as e:
            self._logger.error(f"Hybrid search failed: {e}")
            raise
    
    def __repr__(self) -> str:
        return f"SearchController(engine={self._search_engine})"

