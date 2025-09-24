"""
Vector Database Management for RAG System
Supports both FAISS and ChromaDB backends
"""
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import faiss
import chromadb
from chromadb.config import Settings
from config.settings import VECTOR_DB_DIR, VECTOR_DB_CONFIG, MODEL_CONFIG

class VectorDatabase: 
    """Unified vector database interface supporting FAISS and ChromaDB"""
    
    def __init__(self, backend: str = "faiss", collection_name: str = "shoe_images"):
        self.backend = backend
        self.collection_name = collection_name
        self.dimension = MODEL_CONFIG["clip"]["dimension"] if "dimension" in MODEL_CONFIG["clip"] else 512
        
        if backend == "faiss":
            self._init_faiss()
        elif backend == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _init_faiss(self):
        """Initialize FAISS vector database"""
        self.faiss_index_path = VECTOR_DB_DIR / f"{self.collection_name}.faiss"
        self.metadata_path = VECTOR_DB_DIR / f"{self.collection_name}_metadata.pkl"
        
        # Load or create FAISS index
        if self.faiss_index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.faiss_index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Create new index
            config = VECTOR_DB_CONFIG["faiss"]
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.dimension),
                self.dimension,
                config["nlist"]
            )
            self.metadata = []
    
    def _init_chroma(self):
        """Initialize ChromaDB vector database"""
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(VECTOR_DB_DIR / "chroma_db")
        ))
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": VECTOR_DB_CONFIG["chroma"]["distance_metric"]}
            )
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str] = None):
        """Add vectors to the database"""
        if self.backend == "faiss":
            self._add_vectors_faiss(vectors, metadata, ids)
        elif self.backend == "chroma":
            self._add_vectors_chroma(vectors, metadata, ids)
    
    def _add_vectors_faiss(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str] = None):
        """Add vectors to FAISS index"""
        if not self.index.is_trained:
            # Train the index with a subset of data
            training_data = vectors[:min(1000, len(vectors))]
            self.index.train(training_data.astype('float32'))
        
        # Add vectors to index
        self.index.add(vectors.astype('float32'))
        
        # Update metadata
        if ids is None:
            ids = [f"item_{len(self.metadata) + i}" for i in range(len(vectors))]
        
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            meta['vector_id'] = vector_id
            meta['index_id'] = len(self.metadata) + i
            self.metadata.append(meta)
        
        # Save index and metadata
        self._save_faiss()
    
    def _add_vectors_chroma(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str] = None):
        """Add vectors to ChromaDB"""
        if ids is None:
            ids = [f"item_{i}" for i in range(len(vectors))]
        
        # Convert vectors to list format for ChromaDB
        vectors_list = vectors.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=vectors_list,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query_vector: np.ndarray, k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.backend == "faiss":
            return self._search_faiss(query_vector, k, filters)
        elif self.backend == "chroma":
            return self._search_chroma(query_vector, k, filters)
    
    def _search_faiss(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        # Set search parameters
        self.index.nprobe = VECTOR_DB_CONFIG["faiss"]["nprobe"]
        
        # Search
        distances, indices = self.index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        # Get results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = 1.0 / (1.0 + distance)  # Convert distance to similarity
                result['rank'] = i + 1
                results.append(result)
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    def _search_chroma(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search ChromaDB"""
        # Convert query vector to list
        query_list = query_vector.tolist()
        
        # Prepare where clause for filters
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        for i, (id, distance, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            result = metadata.copy()
            result['vector_id'] = id
            result['similarity_score'] = 1.0 - distance  # ChromaDB uses cosine distance
            result['rank'] = i + 1
            formatted_results.append(result)
        
        return formatted_results
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to search results"""
        filtered_results = []
        
        for result in results:
            match = True
            for key, value in filters.items():
                if key in result:
                    if isinstance(value, list):
                        if result[key] not in value:
                            match = False
                            break
                    else:
                        if result[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_vector_by_id(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID (ChromaDB only)"""
        if self.backend == "chroma":
            result = self.collection.get(ids=[vector_id])
            if result['embeddings']:
                return np.array(result['embeddings'][0])
        return None
    
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]):
        """Update metadata for a vector"""
        if self.backend == "chroma":
            self.collection.update(
                ids=[vector_id],
                metadatas=[metadata]
            )
        elif self.backend == "faiss":
            # Find and update metadata
            for i, meta in enumerate(self.metadata):
                if meta.get('vector_id') == vector_id:
                    self.metadata[i].update(metadata)
                    self._save_faiss()
                    break
    
    def delete_vector(self, vector_id: str):
        """Delete a vector from the database"""
        if self.backend == "chroma":
            self.collection.delete(ids=[vector_id])
        elif self.backend == "faiss":
            # FAISS doesn't support deletion, so we mark as deleted
            for meta in self.metadata:
                if meta.get('vector_id') == vector_id:
                    meta['deleted'] = True
                    self._save_faiss()
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.backend == "faiss":
            return {
                "total_vectors": self.index.ntotal,
                "dimension": self.index.d,
                "is_trained": self.index.is_trained,
                "metadata_count": len(self.metadata)
            }
        elif self.backend == "chroma":
            count = self.collection.count()
            return {
                "total_vectors": count,
                "collection_name": self.collection_name
            }
    
    def _save_faiss(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, str(self.faiss_index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def rebuild_index(self):
        """Rebuild the entire index (FAISS only)"""
        if self.backend == "faiss":
            # Extract all vectors and metadata
            vectors = []
            valid_metadata = []
            
            for meta in self.metadata:
                if not meta.get('deleted', False):
                    # Get vector from ChromaDB or reconstruct
                    vector = self.get_vector_by_id(meta.get('vector_id', ''))
                    if vector is not None:
                        vectors.append(vector)
                        valid_metadata.append(meta)
            
            if vectors:
                # Create new index
                vectors_array = np.array(vectors)
                config = VECTOR_DB_CONFIG["faiss"]
                self.index = faiss.IndexIVFFlat(
                    faiss.IndexFlatL2(self.dimension),
                    self.dimension,
                    config["nlist"]
                )
                
                # Train and add vectors
                self.index.train(vectors_array.astype('float32'))
                self.index.add(vectors_array.astype('float32'))
                
                # Update metadata
                self.metadata = valid_metadata
                self._save_faiss()
    
    def clear_database(self):
        """Clear all vectors from the database"""
        if self.backend == "faiss":
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.dimension),
                self.dimension,
                VECTOR_DB_CONFIG["faiss"]["nlist"]
            )
            self.metadata = []
            self._save_faiss()
        elif self.backend == "chroma":
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": VECTOR_DB_CONFIG["chroma"]["distance_metric"]}
            )

# Factory function for creating vector database instances
def create_vector_db(backend: str = "faiss", collection_name: str = "shoe_images") -> VectorDatabase:
    """Create a vector database instance"""
    return VectorDatabase(backend=backend, collection_name=collection_name)

# Utility functions
def get_embedding_dimension(model_name: str) -> int:
    """Get embedding dimension for a given model"""
    dimensions = {
        "ViT-B/32": 512,
        "ViT-L/14": 768,
        "resnet50": 2048,
        "all-MiniLM-L6-v2": 384
    }
    return dimensions.get(model_name, 512)

def validate_vector(vector: np.ndarray, expected_dim: int) -> bool:
    """Validate vector dimensions"""
    return vector.shape[-1] == expected_dim

if __name__ == "__main__":
    # Test vector database
    db = create_vector_db("faiss")
    print(f"Vector database created: {db.get_stats()}")

