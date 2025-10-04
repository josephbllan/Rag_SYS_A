"""
RAG System Configuration Settings
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "ebay_shoes"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Database settings -----------------------to be changed to AWS
#will be created in the rag/ directory when the RAG system runs for the first time. 
DATABASE_URL = "sqlite:///rag/shoe_metadata.db"

# Vector database settings
VECTOR_DB_CONFIG = {
    "faiss": {
        "index_type": "IVFFlat",
        "nlist": 1000,
        "nprobe": 10,
        "dimension": 512  # CLIP embedding dimension
    },
    "chroma": {
        "collection_name": "shoe_images",
        "distance_metric": "cosine"
    }
}

# Model settings
MODEL_CONFIG = {
    "clip": {
        "model_name": "ViT-B/32",
        "device": "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu",
        "batch_size": 32
    },
    "sentence_transformer": {
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu"
    },
    "resnet": {
        "model_name": "resnet50",
        "pretrained": True,
        "device": "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    }
}

# Search settings
SEARCH_CONFIG = {
    "max_results": 50,
    "similarity_threshold": 0.7,
    "hybrid_weights": {
        "visual": 0.4,
        "text": 0.3,
        "metadata": 0.3
    },
    "cache_ttl": 3600  # 1 hour
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "cors_origins": ["*"],
    "max_file_size": 10 * 1024 * 1024  # 10MB
}

# Web interface settings
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "template_dir": "templates",
    "static_dir": "static"
}

# Image processing settings
IMAGE_CONFIG = {
    "max_size": (512, 512),
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "quality": 95
}

# Metadata categories (from eBay scraper)
METADATA_CATEGORIES = {
    "patterns": ["zigzag", "circular", "square", "diamond", "brand_logo", "other"],
    "shapes": ["round", "square", "oval", "irregular", "elongated"],
    "sizes": ["small", "medium", "large", "extra_large"],
    "brands": ["nike", "adidas", "puma", "converse", "vans", "reebok", 
              "new_balance", "asics", "under_armour", "jordan", "other"]
}

# Search query templates
QUERY_TEMPLATES = {
    "text_to_image": [
        "Find {brand} shoes with {pattern} pattern",
        "Show me {shape} shaped {size} shoes",
        "Search for {color} {brand} sneakers",
        "Find athletic shoes for {activity}",
        "Show me {style} shoes in {size} size"
    ],
    "image_similarity": [
        "Find shoes similar to this image",
        "Show me shoes with similar pattern",
        "Find shoes with similar shape and color"
    ],
    "metadata_filter": [
        "Filter by brand: {brand}",
        "Filter by pattern: {pattern}",
        "Filter by size: {size}",
        "Filter by shape: {shape}"
    ]
}

# Analytics settings
ANALYTICS_CONFIG = {
    "track_searches": True,
    "track_user_behavior": True,
    "retention_days": 30,
    "export_formats": ["json", "csv", "excel"]
}

# Cache settings
CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379/0",
    "memory_cache_size": 1000,
    "default_ttl": 3600
}

# Export settings
EXPORT_CONFIG = {
    "formats": ["json", "csv", "excel", "zip"],
    "max_results": 10000,
    "include_embeddings": False,
    "include_metadata": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "rag/logs/rag_system.log"
}

# Environment variables
ENV_VARS = {
    "CUDA_AVAILABLE": os.getenv("CUDA_AVAILABLE", "false"),
    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "DATABASE_URL": os.getenv("DATABASE_URL", DATABASE_URL),
    "DEBUG": os.getenv("DEBUG", "true").lower() == "true"
}

# JWT Authentication settings
JWT_CONFIG = {
    'secret_key': 'your-secret-key-change-this-in-production-use-openssl-rand-hex-32',
    'algorithm': 'HS256',
    'access_token_expire_minutes': 30,
    'refresh_token_expire_days': 7
}

# User database (for demo - replace with real database in production)
DEMO_USERS = {
    'admin': {
        'username': 'admin',
        'email': 'admin@rag.com',
        'hashed_password': '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',  # 'secret'
        'full_name': 'Admin User',
        'disabled': False,
        'roles': ['admin', 'user']
    },
    'demo': {
        'username': 'demo',
        'email': 'demo@rag.com',
        'hashed_password': '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',  # 'secret'
        'full_name': 'Demo User',
        'disabled': False,
        'roles': ['user']
    }
}
