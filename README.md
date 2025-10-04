# Multi-Modal Image Search System
###### By Joseph Ballan | ballan.joseph@gmail.com

A Retrieval-Augmented Generation (RAG) system for intelligent image search and analysis. This system combines computer vision, natural language processing, and vector search to provide multi-modal search capabilities for image datasets.

## 1. Features

### Core Capabilities
- Multi-Modal Search: Text-to-image, image-to-image, and hybrid search
- Advanced Embeddings: CLIP, ResNet, and Sentence Transformers
- Vector Database: FAISS and ChromaDB support
- Natural Language Processing: Intelligent query understanding
- Metadata Filtering: Attribute-based filtering
- Real-time Search: Fast similarity search with configurable thresholds

### Search Types
- Text Search: Find images using natural language descriptions
- Image Search: Find similar images using a reference image
- Hybrid Search: Combine text, image, and metadata for optimal results
- Semantic Search: Advanced query expansion and semantic understanding
- Metadata Search: Filter by specific attributes

### Web Interface
- Interactive Search: User-friendly web interface
- Image Upload: Upload images for similarity search
- Real-time Results: Instant search results with similarity scores
- Analytics Dashboard: Search statistics and insights
- Browse Mode: Explore all indexed images

### API Integration
- REST API: FastAPI-based RESTful endpoints with MVC architecture
- JWT Authentication: Token-based security with OAuth2 flow
- Role-Based Access: User roles and permissions
- CORS Support: Cross-origin resource sharing enabled
- File Upload: Image upload endpoints
- Export Options: JSON, CSV, and Excel export formats
- Health Monitoring: System status and health checks
- API Versioning: Versioned endpoints
- Auto-Documentation: Interactive Swagger UI and ReDoc

## 2. Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   REST API      │    │   Main System   │
│   (Flask)       │    │   (FastAPI)     │    │   (main.py)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Search Engine         │
                    │   (core/search_engine.py) │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────┐    ┌───────────▼───────────┐    ┌───────▼───────┐
│  Embeddings   │    │   Vector Database     │    │  Query        │
│  (CLIP/ResNet)│    │   (FAISS/ChromaDB)    │    │  Processor    │
└───────────────┘    └───────────────────────┘    └───────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     SQLite Database       │
                    │   (Metadata & Logs)       │
                    └───────────────────────────┘
```

## 3. Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Redis (optional, for caching)

## 4. Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional)
```bash
export CUDA_AVAILABLE=true  # Enable GPU acceleration
export REDIS_URL=redis://localhost:6379/0  # Redis cache URL
export DEBUG=true  # Enable debug mode
```

### 4. Initialize the System
```bash
python main.py --mode index --image-dir /path/to/shoe/images
```

## 5. Quick Start

### Start the Web Interface
```bash
python main.py --mode serve
```

This will start:
- Web interface at: http://localhost:5000
- REST API at: http://localhost:8000

### Index Images
```bash
python main.py --mode index --image-dir /path/to/images
```

### Search via Command Line
```bash
# Text search
python main.py --mode search --query "red nike sneakers" --search-type text

# Image search
python main.py --mode search --query /path/to/image.jpg --search-type image

# Hybrid search
python main.py --mode search --query "athletic shoes" --search-type hybrid
```

## 6. Usage Examples

### Web Interface
1. Open http://localhost:5000 in your browser
2. Enter a search query or upload an image
3. Select search type (text, image, hybrid, semantic)
4. Apply filters if needed
5. View results with similarity scores

### REST API

#### Text Search
```bash
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "red nike sneakers",
    "filters": {"brand": "nike", "size": "medium"},
    "limit": 10
  }'
```

#### Image Search
```bash
curl -X POST "http://localhost:8000/search/image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/reference/image.jpg",
    "limit": 10
  }'
```

#### Hybrid Search
```bash
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "athletic shoes",
    "image_path": "/path/to/image.jpg",
    "filters": {"brand": "adidas"},
    "limit": 10
  }'
```

#### Natural Language Search
```bash
curl -X POST "http://localhost:8000/search/natural?query=find%20me%20some%20running%20shoes%20in%20size%20large"
```

### Python API
```python
from main import RAGSystem

# Initialize system
rag = RAGSystem(vector_backend="faiss")

# Index images
result = rag.index_images("/path/to/images")
print(f"Indexed {result['indexed_count']} images")

# Text search
results = rag.search("red nike sneakers", search_type="text", limit=10)
for result in results:
    print(f"Image: {result['filename']}, Score: {result['similarity_score']:.3f}")

# Image search
results = rag.search("/path/to/image.jpg", search_type="image", limit=10)

# Hybrid search
results = rag.search("athletic shoes", search_type="hybrid", limit=10)
```

## 7. Configuration

### Vector Database Settings
```python
# config/settings.py
VECTOR_DB_CONFIG = {
    "faiss": {
        "index_type": "IVFFlat",
        "nlist": 1000,
        "nprobe": 10,
        "dimension": 512
    },
    "chroma": {
        "collection_name": "shoe_images",
        "distance_metric": "cosine"
    }
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    "clip": {
        "model_name": "ViT-B/32",
        "device": "cuda" if CUDA_AVAILABLE else "cpu",
        "batch_size": 32
    },
    "sentence_transformer": {
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu"
    }
}
```

### Search Configuration
```python
SEARCH_CONFIG = {
    "max_results": 50,
    "similarity_threshold": 0.7,
    "hybrid_weights": {
        "visual": 0.4,
        "text": 0.3,
        "metadata": 0.3
    }
}
```

## 8. API Endpoints

### 8.1 Authentication Endpoints [NEW - SECURE]
- \POST /api/v1/auth/login\ - Login and get JWT token
- \GET /api/v1/auth/me\ - Get current user information (requires auth)
- \POST /api/v1/auth/logout\ - Logout (validates token)

**Demo Credentials:**
- Username: \dmin\ / Password: \secret\ (admin, user roles)
- Username: \demo\ / Password: \secret\ (user role)

### Search Endpoints
- `POST /search/text` - Text-to-image search
- `POST /search/image` - Image-to-image search
- `POST /search/hybrid` - Hybrid search
- `POST /search/semantic` - Semantic search
- `POST /search/natural` - Natural language search

### Utility Endpoints
- `GET /health` - Health check
- `POST /upload` - Image upload
- `GET /metadata/categories` - Available metadata categories
- `GET /analytics/stats` - Search statistics
- `GET /system/status` - System status
- `POST /system/rebuild-index` - Rebuild vector index

### Export Endpoints
- `GET /export/results` - Export search results (JSON/CSV)

## 9. Metadata Categories

The system supports filtering by:

### Patterns
- zigzag, circular, square, diamond, brand_logo, other

### Shapes
- round, square, oval, irregular, elongated

### Sizes
- small, medium, large, extra_large

### Brands
- nike, adidas, puma, converse, vans, reebok, new_balance, asics, under_armour, jordan, other

## 10. Advanced Features

### Query Processing
- Natural language understanding
- Query expansion and synonyms
- Intent recognition
- Filter extraction

### Hybrid Scoring
- Weighted combination of visual, text, and metadata scores
- Configurable weights for different search types
- Similarity threshold filtering

### Caching
- Embedding cache for faster processing
- Redis support for distributed caching
- Configurable TTL settings

### Analytics
- Search query logging
- Performance metrics
- User behavior tracking
- Export capabilities

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   export CUDA_AVAILABLE=false
   ```

2. **Database Connection Issues**
   - Check SQLite file permissions
   - Ensure database directory exists

3. **Vector Index Issues**
   - Rebuild index: `python main.py --mode stats`
   - Check vector database directory permissions

4. **Model Loading Issues**
   - Ensure all dependencies are installed
   - Check model download permissions

### Logs
Check the log file at `rag/logs/rag_system.log` for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI CLIP for multi-modal embeddings
- Facebook AI Research for FAISS
- Hugging Face for transformers
- The open-source community for various dependencies

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for error details

---

**Note**: This system is designed for image search and analysis. The metadata extraction assumes a specific filename format from scraper data. Modify the metadata extraction logic in `main.py` for different data sources.
