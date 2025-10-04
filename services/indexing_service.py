"""
Indexing Service - Business logic for image indexing operations
Handles batch processing and metadata extraction
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from .base_service import BaseService
from ..domain.models import IndexingResult, ImageMetadata
from ..domain.interfaces import IVectorDatabase, IEmbeddingModel
from ..patterns.observer import EventPublisher


class IndexingService(BaseService):
    """
    Service for indexing images and extracting embeddings
    Handles batch processing and error recovery
    """
    
    def __init__(
        self,
        embedding_model: Optional[IEmbeddingModel] = None,
        vector_db: Optional[IVectorDatabase] = None,
        event_publisher: Optional[EventPublisher] = None
    ):
        """
        Initialize indexing service
        
        Args:
            embedding_model: Model for generating embeddings
            vector_db: Vector database for storing embeddings
            event_publisher: Event publisher for monitoring
        """
        super().__init__("IndexingService")
        self._embedding_model = embedding_model
        self._vector_db = vector_db
        self._event_publisher = event_publisher
        self._total_indexed = 0
        self._total_failed = 0
    
    def index_directory(
        self,
        directory_path: str,
        batch_size: int = 32,
        recursive: bool = True
    ) -> IndexingResult:
        """
        Index all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            batch_size: Number of images to process at once
            recursive: Whether to process subdirectories
            
        Returns:
            IndexingResult with statistics
            
        Raises:
            RuntimeError: If service not initialized
            ValueError: If directory doesn't exist
        """
        self._ensure_initialized()
        
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        self._logger.info(f"Starting indexing: {directory_path}")
        start_time = time.time()
        
        # Find all image files
        image_files = self._find_images(dir_path, recursive)
        total_files = len(image_files)
        
        if total_files == 0:
            self._logger.warning(f"No images found in {directory_path}")
            return IndexingResult(
                total_processed=0,
                successful=0,
                failed=0,
                execution_time=0.0
            )
        
        self._logger.info(f"Found {total_files} images to index")
        
        # Process in batches
        successful = 0
        failed = 0
        errors = []
        
        for i in range(0, total_files, batch_size):
            batch = image_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_files - 1) // batch_size + 1
            
            self._logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for image_path in batch:
                try:
                    self._index_single_image(image_path)
                    successful += 1
                    
                    # Publish event
                    if self._event_publisher:
                        self._event_publisher.publish('image_indexed', {
                            'filename': image_path.name,
                            'path': str(image_path)
                        })
                    
                except Exception as e:
                    failed += 1
                    error_msg = f"{image_path.name}: {str(e)}"
                    errors.append(error_msg)
                    self._logger.warning(f"Failed to index {image_path.name}: {e}")
                    
                    # Publish failure event
                    if self._event_publisher:
                        self._event_publisher.publish('indexing_failed', {
                            'filename': image_path.name,
                            'error': str(e)
                        })
        
        execution_time = time.time() - start_time
        
        # Update metrics
        self._total_indexed += successful
        self._total_failed += failed
        self._record_metric('total_indexed', self._total_indexed)
        self._record_metric('total_failed', self._total_failed)
        
        # Publish completion event
        if self._event_publisher:
            self._event_publisher.publish('indexing_complete', {
                'total': total_files,
                'successful': successful,
                'failed': failed,
                'execution_time': execution_time
            })
        
        result = IndexingResult(
            total_processed=total_files,
            successful=successful,
            failed=failed,
            execution_time=execution_time,
            errors=errors[:10]  # Limit to first 10 errors
        )
        
        self._logger.info(
            f"Indexing completed: {successful}/{total_files} successful "
            f"in {execution_time:.2f}s"
        )
        
        return result
    
    def _index_single_image(self, image_path: Path) -> None:
        """Index a single image"""
        if not self._embedding_model or not self._vector_db:
            raise RuntimeError("Embedding model and vector DB required")
        
        # Generate embedding
        embedding = self._embedding_model.encode(str(image_path))
        
        # Extract metadata
        metadata = self._extract_metadata(image_path)
        
        # Add to vector database
        self._vector_db.add_vectors(
            vectors=embedding.reshape(1, -1),
            metadata=[metadata.dict()],
            ids=[f"img_{image_path.stem}"]
        )
    
    def _find_images(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all image files in directory"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        if recursive:
            for ext in supported_extensions:
                image_files.extend(directory.rglob(f'*{ext}'))
        else:
            for ext in supported_extensions:
                image_files.extend(directory.glob(f'*{ext}'))
        
        return sorted(image_files)
    
    def _extract_metadata(self, image_path: Path) -> ImageMetadata:
        """Extract metadata from image file"""
        # Basic metadata extraction
        # In full implementation, would extract from EXIF, analyze filename patterns, etc.
        
        return ImageMetadata(
            filename=image_path.name,
            original_path=str(image_path)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return {
            'total_indexed': self._total_indexed,
            'total_failed': self._total_failed,
            'success_rate': (
                self._total_indexed / (self._total_indexed + self._total_failed)
                if (self._total_indexed + self._total_failed) > 0 else 0.0
            ),
            'service_status': 'active' if self._is_initialized else 'inactive'
        }

