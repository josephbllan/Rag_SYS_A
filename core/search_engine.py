"""
Main Search Engine for RAG System
Combines vector search, metadata filtering, and hybrid search
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json

from .vector_db import VectorDatabase, create_vector_db 
from .embeddings import EmbeddingManager, MultiModalEmbedder, cosine_similarity
from config.settings import SEARCH_CONFIG, METADATA_CATEGORIES
from config.database import get_db, SearchQuery, SearchResult, ShoeImage

logger = logging.getLogger(__name__)

class SearchEngine:
    """Main search engine for the RAG system"""
    
    def __init__(self, vector_backend: str = "faiss"):
        self.vector_db = create_vector_db(vector_backend)
        self.embedding_manager = EmbeddingManager()
        self.multimodal_embedder = MultiModalEmbedder()
        
        # Search configuration
        self.max_results = SEARCH_CONFIG["max_results"]
        self.similarity_threshold = SEARCH_CONFIG["similarity_threshold"]
        self.hybrid_weights = SEARCH_CONFIG["hybrid_weights"]
    
    def text_to_image_search(self, query: str, filters: Dict[str, Any] = None, 
                           limit: int = None) -> List[Dict[str, Any]]:
        """Search images using text query"""
        try:
            # Generate text embedding
            text_embedding = self.embedding_manager.get_text_embedding(query, "clip")
            
            # Search vector database
            limit = limit or self.max_results
            results = self.vector_db.search(text_embedding, k=limit, filters=filters)
            
            # Apply similarity threshold
            filtered_results = [
                r for r in results 
                if r.get('similarity_score', 0) >= self.similarity_threshold
            ]
            
            # Log search query
            self._log_search_query(query, "text", filters, len(filtered_results))
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Text-to-image search failed: {e}")
            return []
    
    def image_to_image_search(self, image_path: str, filters: Dict[str, Any] = None,
                            limit: int = None) -> List[Dict[str, Any]]:
        """Search similar images using an input image"""
        try:
            # Generate image embedding
            image_embedding = self.embedding_manager.get_image_embedding(image_path, "clip")
            
            # Search vector database
            limit = limit or self.max_results
            results = self.vector_db.search(image_embedding, k=limit, filters=filters)
            
            # Apply similarity threshold
            filtered_results = [
                r for r in results 
                if r.get('similarity_score', 0) >= self.similarity_threshold
            ]
            
            # Log search query
            self._log_search_query(f"Image: {image_path}", "image", filters, len(filtered_results))
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Image-to-image search failed: {e}")
            return []
    
    def metadata_search(self, filters: Dict[str, Any], limit: int = None) -> List[Dict[str, Any]]:
        """Search using only metadata filters"""
        try:
            # Get all vectors and filter by metadata
            # This is a simplified implementation - in practice, you'd use a proper database query
            limit = limit or self.max_results
            
            # For now, we'll use the vector database with empty query
            dummy_query = np.zeros(512)  # CLIP dimension
            results = self.vector_db.search(dummy_query, k=1000, filters=filters)
            
            # Limit results
            limited_results = results[:limit]
            
            # Log search query
            self._log_search_query("Metadata filter", "metadata", filters, len(limited_results))
            
            return limited_results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def hybrid_search(self, query: str = None, image_path: str = None, 
                     filters: Dict[str, Any] = None, limit: int = None) -> List[Dict[str, Any]]:
        """Hybrid search combining text, image, and metadata"""
        try:
            limit = limit or self.max_results
            all_results = {}
            
            # Text search
            if query:
                text_results = self.text_to_image_search(query, filters, limit)
                for result in text_results:
                    result_id = result.get('vector_id', result.get('index_id'))
                    if result_id not in all_results:
                        all_results[result_id] = result
                        all_results[result_id]['scores'] = {}
                    all_results[result_id]['scores']['text'] = result.get('similarity_score', 0)
            
            # Image search
            if image_path:
                image_results = self.image_to_image_search(image_path, filters, limit)
                for result in image_results:
                    result_id = result.get('vector_id', result.get('index_id'))
                    if result_id not in all_results:
                        all_results[result_id] = result
                        all_results[result_id]['scores'] = {}
                    all_results[result_id]['scores']['visual'] = result.get('similarity_score', 0)
            
            # Metadata search
            if filters:
                metadata_results = self.metadata_search(filters, limit)
                for result in metadata_results:
                    result_id = result.get('vector_id', result.get('index_id'))
                    if result_id not in all_results:
                        all_results[result_id] = result
                        all_results[result_id]['scores'] = {}
                    all_results[result_id]['scores']['metadata'] = 1.0  # Perfect match for metadata
            
            # Calculate hybrid scores
            hybrid_results = []
            for result in all_results.values():
                hybrid_score = self._calculate_hybrid_score(result.get('scores', {}))
                result['hybrid_score'] = hybrid_score
                hybrid_results.append(result)
            
            # Sort by hybrid score
            hybrid_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            
            # Limit results
            final_results = hybrid_results[:limit]
            
            # Log search query
            search_type = "hybrid"
            if query and image_path:
                search_type = "text+image"
            elif query:
                search_type = "text+metadata"
            elif image_path:
                search_type = "image+metadata"
            
            self._log_search_query(
                f"Query: {query}, Image: {image_path}", 
                search_type, 
                filters, 
                len(final_results)
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def semantic_search(self, query: str, filters: Dict[str, Any] = None, 
                       limit: int = None) -> List[Dict[str, Any]]:
        """Advanced semantic search with query expansion"""
        try:
            # Query expansion
            expanded_queries = self._expand_query(query)
            
            all_results = {}
            
            # Search with each expanded query
            for expanded_query in expanded_queries:
                results = self.text_to_image_search(expanded_query, filters, limit)
                for result in results:
                    result_id = result.get('vector_id', result.get('index_id'))
                    if result_id not in all_results:
                        all_results[result_id] = result
                        all_results[result_id]['query_matches'] = []
                    all_results[result_id]['query_matches'].append(expanded_query)
            
            # Calculate semantic scores
            semantic_results = []
            for result in all_results.values():
                # Boost score based on number of query matches
                match_boost = len(result.get('query_matches', [])) * 0.1
                result['semantic_score'] = result.get('similarity_score', 0) + match_boost
                semantic_results.append(result)
            
            # Sort by semantic score
            semantic_results.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
            
            # Limit results
            limit = limit or self.max_results
            final_results = semantic_results[:limit]
            
            # Log search query
            self._log_search_query(f"Semantic: {query}", "semantic", filters, len(final_results))
            
            return final_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_recommendations(self, image_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations based on an image"""
        try:
            # Find similar images
            similar_results = self.image_to_image_search(image_path, limit=limit * 2)
            
            # Filter out the exact same image
            recommendations = []
            for result in similar_results:
                if result.get('original_path') != image_path:
                    recommendations.append(result)
                    if len(recommendations) >= limit:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            return []
    
    def search_by_similarity(self, reference_image_path: str, similarity_threshold: float = 0.8,
                           limit: int = None) -> List[Dict[str, Any]]:
        """Find images with high similarity to reference"""
        try:
            # Generate reference embedding
            reference_embedding = self.embedding_manager.get_image_embedding(reference_image_path, "clip")
            
            # Search with custom threshold
            limit = limit or self.max_results
            results = self.vector_db.search(reference_embedding, k=limit)
            
            # Filter by similarity threshold
            similar_results = [
                r for r in results 
                if r.get('similarity_score', 0) >= similarity_threshold
            ]
            
            return similar_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _calculate_hybrid_score(self, scores: Dict[str, float]) -> float:
        """Calculate hybrid score from individual scores"""
        weights = self.hybrid_weights
        total_score = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type in scores:
                total_score += scores[score_type] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        # Simple query expansion - in practice, you'd use more sophisticated methods
        expansions = [query]
        
        # Add variations
        if "shoe" in query.lower():
            expansions.extend([query.replace("shoe", "sneaker"), query.replace("shoe", "footwear")])
        
        if "red" in query.lower():
            expansions.append(query.replace("red", "crimson"))
        
        if "nike" in query.lower():
            expansions.append(query.replace("nike", "nike air"))
        
        return expansions
    
    def _log_search_query(self, query: str, query_type: str, filters: Dict[str, Any], 
                         result_count: int):
        """Log search query to database"""
        try:
            db = next(get_db())
            search_query = SearchQuery(
                query_text=query,
                query_type=query_type,
                filters=json.dumps(filters) if filters else None,
                results_count=result_count,
                execution_time=0.0  # Would be calculated in practice
            )
            db.add(search_query)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log search query: {e}")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            db = next(get_db())
            
            # Get recent searches
            recent_searches = db.query(SearchQuery).order_by(SearchQuery.created_at.desc()).limit(100).all()
            
            # Calculate stats
            total_searches = len(recent_searches)
            query_types = {}
            avg_results = 0
            
            for search in recent_searches:
                query_type = search.query_type
                query_types[query_type] = query_types.get(query_type, 0) + 1
                avg_results += search.results_count
            
            avg_results = avg_results / total_searches if total_searches > 0 else 0
            
            return {
                "total_searches": total_searches,
                "query_types": query_types,
                "avg_results_per_search": avg_results,
                "vector_db_stats": self.vector_db.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}

# Factory function
def create_search_engine(vector_backend: str = "faiss") -> SearchEngine:
    """Create a search engine instance"""
    return SearchEngine(vector_backend)

if __name__ == "__main__":
    # Test search engine
    engine = create_search_engine()
    print("Search engine initialized successfully!")
    print(f"Vector DB stats: {engine.vector_db.get_stats()}")

