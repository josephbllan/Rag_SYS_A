"""
Adapter Pattern Implementation
Allows incompatible interfaces to work together
"""
from typing import List, Dict, Any, Optional
import logging

from ..domain.types import VectorType
from ..domain.models import SearchResultItem
from ..domain.base_classes import BaseVectorDatabase


class VectorDatabaseAdapter:
    """
    Adapter that provides a uniform interface for different vector databases
    """
    
    def __init__(self, database: BaseVectorDatabase):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def add(self, vectors: VectorType, metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Unified interface for adding vectors"""
        try:
            self._database.add_vectors(vectors, metadata, ids)
        except Exception as e:
            self._logger.error(f"Failed to add vectors: {e}")
            raise
    
    def search(self, query: VectorType, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResultItem]:
        """Unified interface for searching"""
        try:
            return self._database.search(query, k=limit, filters=filters)
        except Exception as e:
            self._logger.error(f"Search failed: {e}")
            raise
    
    def remove(self, vector_id: str) -> None:
        """Unified interface for deletion"""
        try:
            self._database.delete_vector(vector_id)
        except Exception as e:
            self._logger.error(f"Failed to remove vector: {e}")
            raise
    
    def statistics(self) -> Dict[str, Any]:
        """Unified interface for statistics"""
        try:
            return self._database.get_stats()
        except Exception as e:
            self._logger.error(f"Failed to get statistics: {e}")
            return {}


class FAISSAdapter(BaseVectorDatabase):
    """Adapter for FAISS vector database"""
    
    def __init__(self, dimension: int, collection_name: str = "default"):
        super().__init__(dimension, collection_name)
        import faiss
        import numpy as np
        self._faiss = faiss
        self._np = np
        self._index = None
        self._metadata = []
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize FAISS index"""
        self._index = self._faiss.IndexFlatL2(self._dimension)
        self._is_initialized = True
        self._logger.info(f"FAISS index initialized with dimension {self._dimension}")
    
    def add_vectors(self, vectors: VectorType, metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Add vectors to FAISS index"""
        self._index.add(vectors.astype('float32'))
        self._metadata.extend(metadata)
        self._logger.debug(f"Added {len(vectors)} vectors to FAISS")
    
    def search(self, query_vector: VectorType, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResultItem]:
        """Search FAISS index"""
        distances, indices = self._index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self._metadata):
                meta = self._metadata[idx]
                result = SearchResultItem(
                    vector_id=meta.get('vector_id', f"idx_{idx}"),
                    filename=meta.get('filename', ''),
                    original_path=meta.get('original_path', ''),
                    similarity_score=1.0 / (1.0 + float(distance)),
                    rank=i + 1,
                    metadata=meta
                )
                results.append(result)
        
        return results
    
    def delete_vector(self, vector_id: str) -> None:
        """FAISS doesn't support deletion easily, mark as deleted"""
        for meta in self._metadata:
            if meta.get('vector_id') == vector_id:
                meta['_deleted'] = True
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        return {
            'total_vectors': self._index.ntotal,
            'dimension': self._dimension,
            'backend': 'faiss'
        }


class ChromaDBAdapter(BaseVectorDatabase):
    """Adapter for ChromaDB"""
    
    def __init__(self, dimension: int, collection_name: str = "default"):
        super().__init__(dimension, collection_name)
        import chromadb
        self._chromadb = chromadb
        self._client = None
        self._collection = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize ChromaDB client"""
        self._client = self._chromadb.Client()
        self._collection = self._client.get_or_create_collection(self._collection_name)
        self._is_initialized = True
        self._logger.info(f"ChromaDB collection initialized: {self._collection_name}")
    
    def add_vectors(self, vectors: VectorType, metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """Add vectors to ChromaDB"""
        if ids is None:
            ids = [f"id_{i}" for i in range(len(vectors))]
        
        self._collection.add(
            embeddings=vectors.tolist(),
            metadatas=metadata,
            ids=ids
        )
        self._logger.debug(f"Added {len(vectors)} vectors to ChromaDB")
    
    def search(self, query_vector: VectorType, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResultItem]:
        """Search ChromaDB"""
        results_data = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k
        )
        
        results = []
        for i, (id, distance, meta) in enumerate(zip(
            results_data['ids'][0],
            results_data['distances'][0],
            results_data['metadatas'][0]
        )):
            result = SearchResultItem(
                vector_id=id,
                filename=meta.get('filename', ''),
                original_path=meta.get('original_path', ''),
                similarity_score=1.0 - float(distance),
                rank=i + 1,
                metadata=meta
            )
            results.append(result)
        
        return results
    
    def delete_vector(self, vector_id: str) -> None:
        """Delete vector from ChromaDB"""
        self._collection.delete(ids=[vector_id])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        return {
            'total_vectors': self._collection.count(),
            'dimension': self._dimension,
            'backend': 'chroma'
        }

