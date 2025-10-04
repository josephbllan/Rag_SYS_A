"""
Strategy Pattern Implementation
Defines a family of algorithms and makes them interchangeable
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from ..domain.models import SearchQuery, SearchResultItem
from ..domain.base_classes import BaseSearchStrategy


class SearchContext:
    """
    Context for search strategies
    Maintains reference to strategy objects and delegates work to them
    """
    
    def __init__(self, strategy: Optional[BaseSearchStrategy] = None):
        self._strategy = strategy
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def strategy(self) -> Optional[BaseSearchStrategy]:
        """Get current strategy"""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: BaseSearchStrategy) -> None:
        """Set strategy"""
        self._logger.info(f"Switching strategy to: {strategy.name}")
        self._strategy = strategy
    
    def execute_search(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """
        Execute search using current strategy
        """
        if self._strategy is None:
            raise ValueError("No search strategy set")
        
        self._logger.info(f"Executing search with strategy: {self._strategy.name}")
        return self._strategy.execute(query, context)
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate query using current strategy"""
        if self._strategy is None:
            raise ValueError("No search strategy set")
        
        return self._strategy.validate_query(query)


class TextSearchStrategy(BaseSearchStrategy):
    """
    Concrete strategy for text-to-image search
    """
    
    def __init__(self):
        super().__init__(name="text_search")
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute text search"""
        self._logger.info(f"Executing text search: {query.query}")
        
        # Get dependencies from context
        embedding_manager = context.get('embedding_manager')
        vector_db = context.get('vector_db')
        
        if not embedding_manager or not vector_db:
            raise ValueError("Missing required dependencies in context")
        
        # Generate text embedding
        text_embedding = embedding_manager.get_text_embedding(query.query, "clip")
        
        # Search vector database
        filters = query.filters.to_dict() if query.filters else None
        results = vector_db.search(text_embedding, k=query.limit, filters=filters)
        
        self._log_search(query, len(results))
        return results
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate text search query"""
        if not query.query or not query.query.strip():
            return False, "Text query is empty"
        if len(query.query) > 500:
            return False, "Text query too long (max 500 characters)"
        return True, "Valid"


class ImageSearchStrategy(BaseSearchStrategy):
    """
    Concrete strategy for image-to-image search
    """
    
    def __init__(self):
        super().__init__(name="image_search")
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute image search"""
        self._logger.info(f"Executing image search: {query.image_path}")
        
        # Get dependencies from context
        embedding_manager = context.get('embedding_manager')
        vector_db = context.get('vector_db')
        
        if not embedding_manager or not vector_db:
            raise ValueError("Missing required dependencies in context")
        
        if not query.image_path:
            raise ValueError("Image path is required for image search")
        
        # Generate image embedding
        image_embedding = embedding_manager.get_image_embedding(query.image_path, "clip")
        
        # Search vector database
        filters = query.filters.to_dict() if query.filters else None
        results = vector_db.search(image_embedding, k=query.limit, filters=filters)
        
        self._log_search(query, len(results))
        return results
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate image search query"""
        if not query.image_path:
            return False, "Image path is required"
        
        import os
        if not os.path.exists(query.image_path):
            return False, f"Image file not found: {query.image_path}"
        
        return True, "Valid"


class HybridSearchStrategy(BaseSearchStrategy):
    """
    Concrete strategy for hybrid search (text + image + metadata)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__(name="hybrid_search")
        self._weights = weights or {"visual": 0.4, "text": 0.3, "metadata": 0.3}
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute hybrid search"""
        self._logger.info(f"Executing hybrid search")
        
        # Get dependencies from context
        embedding_manager = context.get('embedding_manager')
        vector_db = context.get('vector_db')
        
        if not embedding_manager or not vector_db:
            raise ValueError("Missing required dependencies in context")
        
        all_results: Dict[str, SearchResultItem] = {}
        
        # Text search component
        if query.query:
            text_embedding = embedding_manager.get_text_embedding(query.query, "clip")
            text_results = vector_db.search(text_embedding, k=query.limit * 2)
            
            for result in text_results:
                result_id = result.vector_id
                if result_id not in all_results:
                    all_results[result_id] = result
                    all_results[result_id].scores = {}
                all_results[result_id].scores['text'] = result.similarity_score
        
        # Image search component
        if query.image_path:
            image_embedding = embedding_manager.get_image_embedding(query.image_path, "clip")
            image_results = vector_db.search(image_embedding, k=query.limit * 2)
            
            for result in image_results:
                result_id = result.vector_id
                if result_id not in all_results:
                    all_results[result_id] = result
                    all_results[result_id].scores = {}
                all_results[result_id].scores['visual'] = result.similarity_score
        
        # Metadata filtering
        if query.filters:
            for result in all_results.values():
                result.scores['metadata'] = 1.0  # Perfect match for metadata
        
        # Calculate hybrid scores
        hybrid_results = []
        for result in all_results.values():
            hybrid_score = self._calculate_hybrid_score(result.scores)
            result.similarity_score = hybrid_score
            hybrid_results.append(result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        final_results = hybrid_results[:query.limit]
        
        self._log_search(query, len(final_results))
        return final_results
    
    def _calculate_hybrid_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted hybrid score"""
        total_score = 0.0
        total_weight = 0.0
        
        for score_type, weight in self._weights.items():
            if score_type in scores:
                total_score += scores[score_type] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate hybrid search query"""
        if not query.query and not query.image_path:
            return False, "Either text query or image path is required"
        
        if query.image_path:
            import os
            if not os.path.exists(query.image_path):
                return False, f"Image file not found: {query.image_path}"
        
        return True, "Valid"


class SemanticSearchStrategy(BaseSearchStrategy):
    """
    Concrete strategy for semantic search with query expansion
    """
    
    def __init__(self):
        super().__init__(name="semantic_search")
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute semantic search"""
        self._logger.info(f"Executing semantic search: {query.query}")
        
        # Get dependencies from context
        embedding_manager = context.get('embedding_manager')
        vector_db = context.get('vector_db')
        query_processor = context.get('query_processor')
        
        if not all([embedding_manager, vector_db, query_processor]):
            raise ValueError("Missing required dependencies in context")
        
        # Expand query
        expanded_queries = query_processor.expand_query(query.query)
        
        all_results: Dict[str, SearchResultItem] = {}
        
        # Search with each expanded query
        for expanded_query in expanded_queries:
            embedding = embedding_manager.get_text_embedding(expanded_query, "clip")
            results = vector_db.search(embedding, k=query.limit * 2)
            
            for result in results:
                result_id = result.vector_id
                if result_id not in all_results:
                    all_results[result_id] = result
                    all_results[result_id].scores = {'matches': 0}
                all_results[result_id].scores['matches'] += 1
        
        # Boost scores based on number of matches
        semantic_results = []
        for result in all_results.values():
            match_count = result.scores.get('matches', 0)
            boost = min(match_count * 0.1, 0.3)  # Max 30% boost
            result.similarity_score = min(result.similarity_score + boost, 1.0)
            semantic_results.append(result)
        
        # Sort and limit
        semantic_results.sort(key=lambda x: x.similarity_score, reverse=True)
        final_results = semantic_results[:query.limit]
        
        self._log_search(query, len(final_results))
        return final_results
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate semantic search query"""
        if not query.query or not query.query.strip():
            return False, "Query is empty"
        return True, "Valid"


class MetadataSearchStrategy(BaseSearchStrategy):
    """
    Concrete strategy for metadata-only search
    """
    
    def __init__(self):
        super().__init__(name="metadata_search")
    
    def execute(
        self, 
        query: SearchQuery, 
        context: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Execute metadata search"""
        self._logger.info(f"Executing metadata search")
        
        # Get dependencies from context
        vector_db = context.get('vector_db')
        
        if not vector_db:
            raise ValueError("Missing required dependencies in context")
        
        if not query.filters:
            raise ValueError("Filters are required for metadata search")
        
        # Use dummy query for metadata-only search
        import numpy as np
        dummy_query = np.zeros(512, dtype=np.float32)
        
        # Search with filters
        filters = query.filters.to_dict()
        results = vector_db.search(dummy_query, k=query.limit * 10, filters=filters)
        
        # Limit results
        final_results = results[:query.limit]
        
        self._log_search(query, len(final_results))
        return final_results
    
    def validate_query(self, query: SearchQuery) -> tuple[bool, str]:
        """Validate metadata search query"""
        if not query.filters:
            return False, "Filters are required for metadata search"
        return True, "Valid"

