"""
Main RAG System Entry Point
Orchestrates the entire RAG system for shoe image search and analysis
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.search_engine import create_search_engine
from core.embeddings import EmbeddingManager
from core.query_processor import QueryProcessor
from config.database import create_tables, test_connection
from config.settings import DATA_DIR, VECTOR_DB_DIR, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"]),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG System orchestrator"""
    
    def __init__(self, vector_backend: str = "faiss"):
        self.vector_backend = vector_backend
        self.search_engine = None
        self.embedding_manager = None
        self.query_processor = None
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all RAG system components"""
        try:
            logger.info("Initializing RAG System...")
            
            # Test database connection
            if not test_connection():
                logger.error("Database connection failed")
                raise Exception("Database connection failed")
            
            # Create database tables
            create_tables()
            logger.info("Database tables created/verified")
            
            # Initialize search engine
            self.search_engine = create_search_engine(self.vector_backend)
            logger.info("Search engine initialized")
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager()
            logger.info("Embedding manager initialized")
            
            # Initialize query processor
            self.query_processor = QueryProcessor()
            logger.info("Query processor initialized")
            
            logger.info("RAG System initialized successfully!")
            
        except Exception as e:
            logger.error(f"RAG System initialization failed: {e}")
            raise
    
    def index_images(self, image_directory: str = None, batch_size: int = 32) -> Dict[str, Any]:
        """Index all images in the specified directory"""
        try:
            if image_directory is None:
                image_directory = str(DATA_DIR)
            
            logger.info(f"Starting image indexing from: {image_directory}")
            
            # Find all image files
            image_paths = self._find_image_files(image_directory)
            logger.info(f"Found {len(image_paths)} images to index")
            
            if not image_paths:
                return {"status": "no_images", "count": 0}
            
            # Process images in batches
            indexed_count = 0
            failed_count = 0
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
                
                for image_path in batch_paths:
                    try:
                        # Generate embeddings
                        clip_embedding = self.embedding_manager.get_image_embedding(image_path, "clip")
                        resnet_features = self.embedding_manager.get_image_embedding(image_path, "resnet")
                        
                        # Extract metadata from filename (assuming format from eBay scraper)
                        metadata = self._extract_metadata_from_path(image_path)
                        
                        # Add to vector database
                        self.search_engine.vector_db.add_vectors(
                            vectors=clip_embedding.reshape(1, -1),
                            metadata=[metadata],
                            ids=[f"img_{indexed_count}"]
                        )
                        
                        indexed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to index {image_path}: {e}")
                        failed_count += 1
                        continue
            
            logger.info(f"Indexing completed: {indexed_count} successful, {failed_count} failed")
            
            return {
                "status": "completed",
                "indexed_count": indexed_count,
                "failed_count": failed_count,
                "total_found": len(image_paths)
            }
            
        except Exception as e:
            logger.error(f"Image indexing failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def search(self, query: str, search_type: str = "text", **kwargs) -> List[Dict[str, Any]]:
        """Perform search using the RAG system"""
        try:
            if search_type == "text":
                return self.search_engine.text_to_image_search(query, **kwargs)
            elif search_type == "image":
                return self.search_engine.image_to_image_search(query, **kwargs)
            elif search_type == "hybrid":
                return self.search_engine.hybrid_search(query=query, **kwargs)
            elif search_type == "semantic":
                return self.search_engine.semantic_search(query, **kwargs)
            elif search_type == "natural":
                intent = self.query_processor.process_query(query)
                return self.search_engine.hybrid_search(
                    query=" ".join(intent.search_terms),
                    filters=intent.filters,
                    limit=intent.limit
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_recommendations(self, image_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations for an image"""
        try:
            return self.search_engine.get_recommendations(image_path, limit)
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            return {
                "search_engine": self.search_engine.get_search_stats(),
                "vector_db": self.search_engine.vector_db.get_stats(),
                "system": {
                    "vector_backend": self.vector_backend,
                    "data_directory": str(DATA_DIR),
                    "vector_db_directory": str(VECTOR_DB_DIR)
                }
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}
    
    def _find_image_files(self, directory: str) -> List[str]:
        """Find all image files in directory"""
        from config.settings import IMAGE_CONFIG
        
        image_paths = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return image_paths
        
        for ext in IMAGE_CONFIG["supported_formats"]:
            pattern = f"**/*{ext}"
            image_paths.extend([str(p) for p in directory_path.glob(pattern)])
        
        return sorted(image_paths)
    
    def _extract_metadata_from_path(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from image file path"""
        path = Path(image_path)
        filename = path.stem
        #------------------------------------NEED TO BE CHANGED
        # Try to extract metadata from filename (assuming eBay scraper format)
        # Format: shoe_{index}_{pattern}_{shape}_{size}_{brand}.jpg
        parts = filename.split('_')
        
        metadata = {
            "filename": path.name,
            "original_path": str(image_path),
            "pattern": "other",
            "shape": "other",
            "size": "medium",
            "brand": "other"
        }
        
        if len(parts) >= 6:
            try:
                metadata.update({
                    "pattern": parts[2] if parts[2] in ["zigzag", "circular", "square", "diamond", "brand_logo", "other"] else "other",
                    "shape": parts[3] if parts[3] in ["round", "square", "oval", "irregular", "elongated"] else "other",
                    "size": parts[4] if parts[4] in ["small", "medium", "large", "extra_large"] else "medium",
                    "brand": parts[5] if parts[5] in ["nike", "adidas", "puma", "converse", "vans", "reebok", "new_balance", "asics", "under_armour", "jordan", "other"] else "other"
                })
            except IndexError:
                pass
        
        return metadata

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG System for Shoe Image Search")
    parser.add_argument("--mode", choices=["index", "search", "serve", "stats"], 
                       default="serve", help="Operation mode")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--search-type", choices=["text", "image", "hybrid", "semantic", "natural"],
                       default="text", help="Search type")
    parser.add_argument("--image-dir", type=str, help="Image directory for indexing")
    parser.add_argument("--vector-backend", choices=["faiss", "chroma"], 
                       default="faiss", help="Vector database backend")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag = RAGSystem(vector_backend=args.vector_backend)
        
        if args.mode == "index":
            # Index images
            result = rag.index_images(args.image_dir)
            print(json.dumps(result, indent=2))
            
        elif args.mode == "search":
            # Perform search
            if not args.query:
                print("Error: Query is required for search mode")
                return
            
            results = rag.search(args.query, args.search_type, limit=args.limit)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results[:5]):  # Show first 5 results
                print(f"{i+1}. {result.get('filename', 'Unknown')} (Score: {result.get('similarity_score', 0):.3f})")
                
        elif args.mode == "stats":
            # Show statistics
            stats = rag.get_stats()
            print(json.dumps(stats, indent=2))
            
        elif args.mode == "serve":
            # Start web server
            print("Starting RAG System web interface...")
            print("Web interface will be available at: http://localhost:5000")
            print("API will be available at: http://localhost:8000")
            
            # Start web interface in background
            import subprocess
            import threading
            
            def start_web():
                from web.app import app
                app.run(host='127.0.0.1', port=5000, debug=False)
            
            def start_api():
                import uvicorn
                from api.endpoints import app as api_app
                uvicorn.run(api_app, host="0.0.0.0", port=8000)
            
            # Start both servers
            web_thread = threading.Thread(target=start_web)
            api_thread = threading.Thread(target=start_api)
            
            web_thread.daemon = True
            api_thread.daemon = True
            
            web_thread.start()
            api_thread.start()
            
            print("Both servers started. Press Ctrl+C to stop.")
            
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                
    except Exception as e:
        logger.error(f"RAG System failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

