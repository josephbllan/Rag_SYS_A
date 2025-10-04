"""
Image and Text Embedding Generation for RAG System
Supports CLIP, ResNet, and Sentence Transformers
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Union, Optional
import clip
from sentence_transformers import SentenceTransformer 
import torchvision.models as models
from config.settings import MODEL_CONFIG, IMAGE_CONFIG
import logging

logger = logging.getLogger(__name__)

class ImageEmbedder:
    """Generate embeddings for images using various models"""
    
    def __init__(self, model_type: str = "clip"):
        self.model_type = model_type
        self.device = torch.device(MODEL_CONFIG["clip"]["device"])
        self._load_model()
    
    def _load_model(self):
        """Load the specified model"""
        if self.model_type == "clip":
            self._load_clip()
        elif self.model_type == "resnet":
            self._load_resnet()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_clip(self):
        """Load CLIP model"""
        try:
            model_name = MODEL_CONFIG["clip"]["model_name"]
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            self.dimension = 512 if "ViT-B" in model_name else 768
            logger.info(f"CLIP model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_resnet(self):
        """Load ResNet model"""
        try:
            model_name = MODEL_CONFIG["resnet"]["model_name"]
            self.model = models.__dict__[model_name](pretrained=MODEL_CONFIG["resnet"]["pretrained"])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.dimension = 2048  # ResNet50 feature dimension
            
            # Define preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"ResNet model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load ResNet model: {e}")
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                if self.model_type == "clip":
                    image_features = self.model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                elif self.model_type == "resnet":
                    image_features = self.model(image)
                    # Use global average pooling
                    image_features = torch.nn.functional.adaptive_avg_pool2d(image_features, (1, 1))
                    image_features = image_features.view(image_features.size(0), -1)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return np.zeros(self.dimension)
    
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode multiple images in batches"""
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            
            for path in batch_paths:
                embedding = self.encode_image(path)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP (only available for CLIP model)"""
        if self.model_type != "clip":
            raise ValueError("Text encoding only available for CLIP model")
        
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}")
            return np.zeros(self.dimension)

class TextEmbedder:
    """Generate embeddings for text using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = MODEL_CONFIG["sentence_transformer"]["device"]
        self._load_model()
    
    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Sentence Transformer model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text"""
        try:
            embedding = self.model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}")
            return np.zeros(self.dimension)
    
    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Encode multiple texts in batches"""
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size)
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to encode texts batch: {e}")
            return [np.zeros(self.dimension) for _ in texts]

class MultiModalEmbedder:
    """Combined image and text embedding system"""
    
    def __init__(self):
        self.image_embedder = ImageEmbedder("clip")
        self.text_embedder = TextEmbedder()
        self.clip_embedder = ImageEmbedder("clip")  # For CLIP text encoding
    
    def encode_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Encode image with multiple models"""
        return {
            "clip": self.image_embedder.encode_image(image_path),
            "resnet": ImageEmbedder("resnet").encode_image(image_path)
        }
    
    def encode_text(self, text: str) -> Dict[str, np.ndarray]:
        """Encode text with multiple models"""
        return {
            "clip": self.clip_embedder.encode_text(text),
            "sentence_transformer": self.text_embedder.encode_text(text)
        }
    
    def encode_image_text_pair(self, image_path: str, text: str) -> Dict[str, np.ndarray]:
        """Encode both image and text"""
        return {
            "image_clip": self.image_embedder.encode_image(image_path),
            "image_resnet": ImageEmbedder("resnet").encode_image(image_path),
            "text_clip": self.clip_embedder.encode_text(text),
            "text_sentence_transformer": self.text_embedder.encode_text(text)
        }

class EmbeddingManager:
    """Manage and cache embeddings"""
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.multimodal_embedder = MultiModalEmbedder()
    
    def get_image_embedding(self, image_path: str, model_type: str = "clip", use_cache: bool = True) -> np.ndarray:
        """Get image embedding with optional caching"""
        if use_cache:
            cache_path = self.cache_dir / f"{Path(image_path).stem}_{model_type}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Generate embedding
        if model_type == "clip":
            embedding = self.multimodal_embedder.image_embedder.encode_image(image_path)
        elif model_type == "resnet":
            embedding = ImageEmbedder("resnet").encode_image(image_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cache embedding
        if use_cache:
            np.save(cache_path, embedding)
        
        return embedding
    
    def get_text_embedding(self, text: str, model_type: str = "sentence_transformer", use_cache: bool = True) -> np.ndarray:
        """Get text embedding with optional caching"""
        if use_cache:
            cache_key = f"{hash(text)}_{model_type}"
            cache_path = self.cache_dir / f"text_{cache_key}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Generate embedding
        if model_type == "clip":
            embedding = self.multimodal_embedder.clip_embedder.encode_text(text)
        elif model_type == "sentence_transformer":
            embedding = self.multimodal_embedder.text_embedder.encode_text(text)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cache embedding
        if use_cache:
            np.save(cache_path, embedding)
        
        return embedding
    
    def batch_process_images(self, image_paths: List[str], model_type: str = "clip") -> List[np.ndarray]:
        """Process multiple images in batch"""
        if model_type == "clip":
            embedder = self.multimodal_embedder.image_embedder
        elif model_type == "resnet":
            embedder = ImageEmbedder("resnet")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return embedder.encode_images_batch(image_paths)
    
    def batch_process_texts(self, texts: List[str], model_type: str = "sentence_transformer") -> List[np.ndarray]:
        """Process multiple texts in batch"""
        if model_type == "clip":
            embedder = self.multimodal_embedder.clip_embedder
        elif model_type == "sentence_transformer":
            embedder = self.multimodal_embedder.text_embedder
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return embedder.encode_texts_batch(texts)
    
    def clear_cache(self):
        """Clear embedding cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

# Utility functions
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit vector"""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    return np.dot(a_norm, b_norm)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate euclidean distance between two embeddings"""
    return np.linalg.norm(a - b)

def create_embedding_from_metadata(metadata: Dict[str, Any]) -> np.ndarray:
    """Create a simple embedding from metadata for hybrid search"""
    # Simple one-hot encoding of metadata
    features = []
    
    # Pattern features
    patterns = ["zigzag", "circular", "square", "diamond", "brand_logo", "other"]
    pattern_vector = [1 if metadata.get("pattern") == p else 0 for p in patterns]
    features.extend(pattern_vector)
    
    # Shape features
    shapes = ["round", "square", "oval", "irregular", "elongated"]
    shape_vector = [1 if metadata.get("shape") == s else 0 for s in shapes]
    features.extend(shape_vector)
    
    # Size features
    sizes = ["small", "medium", "large", "extra_large"]
    size_vector = [1 if metadata.get("size") == s else 0 for s in sizes]
    features.extend(size_vector)
    
    # Brand features
    brands = ["nike", "adidas", "puma", "converse", "vans", "reebok", "new_balance", "asics", "under_armour", "jordan", "other"]
    brand_vector = [1 if metadata.get("brand") == b else 0 for b in brands]
    features.extend(brand_vector)
    
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    # Test embedding system
    embedder = EmbeddingManager()
    print("Embedding system initialized successfully!")

