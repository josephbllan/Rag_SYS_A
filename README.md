# Multi-Modal Image RAG System
### By Joseph Ballan | ballan.joseph@gmail.com

A sophisticated **generic** Retrieval-Augmented Generation (RAG) system for intelligent image search and analysis. This system combines computer vision, natural language processing, and vector search to provide powerful multi-modal search capabilities for **any type of images**.

> **🎯 Current Implementation**: Shoe Image Search (fully configurable for any domain)
> 
> **🔧 Generic Architecture**: Easily adaptable for clothing, cars, food, products, or any image domain

## 🚀 Features

### Core Capabilities
- **Multi-Modal Search**: Text-to-image, image-to-image, and hybrid search
- **Advanced Embeddings**: CLIP, ResNet, and Sentence Transformers
- **Vector Database**: FAISS and ChromaDB support
- **Natural Language Processing**: Intelligent query understanding
- **Metadata Filtering**: Customizable domain-specific filtering
- **Real-time Search**: Fast similarity search with configurable thresholds
- **Domain Agnostic**: Works with any type of images (shoes, clothing, cars, food, etc.)

### Search Types
- **Text Search**: Find images using natural language descriptions
- **Image Search**: Find similar images using a reference image
- **Hybrid Search**: Combine text, image, and metadata for optimal results
- **Semantic Search**: Advanced query expansion and semantic understanding
- **Metadata Search**: Filter by specific attributes (customizable per domain)

### Web Interface
- **Interactive Search**: User-friendly web interface
- **Image Upload**: Upload images for similarity search
- **Real-time Results**: Instant search results with similarity scores
- **Analytics Dashboard**: Search statistics and insights
- **Browse Mode**: Explore all indexed images

### API Integration
- **REST API**: FastAPI-based RESTful endpoints
- **CORS Support**: Cross-origin resource sharing enabled
- **File Upload**: Image upload endpoints
- **Export Options**: JSON, CSV, and Excel export formats
- **Health Monitoring**: System status and health checks

## 🏗️ Architecture

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
                    │   (Domain Metadata)       │
                    └───────────────────────────┘
```

### 🎯 **Domain Adaptability**

The system is designed with **domain-agnostic** components that can be easily configured for any image type:

| Component | Generic | Domain-Specific |
|-----------|---------|-----------------|
| **Vector Search** | ✅ CLIP/ResNet embeddings | ❌ |
| **Search Engine** | ✅ Similarity algorithms | ❌ |
| **Web Interface** | ✅ Generic image display | ❌ |
| **REST API** | ✅ Standard endpoints | ❌ |
| **Database Schema** | ❌ | ✅ Custom metadata |
| **Query Processing** | ❌ | ✅ Domain terms |
| **Metadata Categories** | ❌ | ✅ Domain attributes |

## 🎯 Domain Configuration

### **Current Implementation: Shoe Images**
The system is currently configured for shoe image search as an example. The metadata categories are **fully customizable** for any domain:

```python
# Example: Shoe-specific metadata categories (current implementation)
METADATA_CATEGORIES = {
    "patterns": ["zigzag", "circular", "square", "diamond", "brand_logo", "other"],
    "shapes": ["round", "square", "oval", "irregular", "elongated"],
    "sizes": ["small", "medium", "large", "extra_large"],
    "brands": ["nike", "adidas", "puma", "converse", "vans", "reebok", "new_balance", "asics", "under_armour", "jordan", "other"]
}

# This is just ONE example - you can define ANY categories for your domain!
```

### **Easy Domain Adaptation**

The system can be easily adapted for any image domain by modifying configuration files:

#### **🛍️ Clothing Images**
```python
METADATA_CATEGORIES = {
    "types": ["shirt", "pants", "dress", "jacket", "hat", "shoes"],
    "colors": ["red", "blue", "green", "black", "white", "yellow", "pink"],
    "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
    "brands": ["nike", "adidas", "zara", "h&m", "uniqlo", "other"],
    "seasons": ["spring", "summer", "fall", "winter", "all-season"]
}
```

#### **🚗 Car Images**
```python
METADATA_CATEGORIES = {
    "makes": ["toyota", "honda", "ford", "bmw", "mercedes", "audi", "tesla"],
    "types": ["sedan", "suv", "truck", "coupe", "hatchback", "convertible"],
    "colors": ["red", "blue", "black", "white", "silver", "gray"],
    "years": ["2020", "2021", "2022", "2023", "2024"],
    "fuel_types": ["gasoline", "electric", "hybrid", "diesel"]
}
```

#### **🍕 Food Images**
```python
METADATA_CATEGORIES = {
    "cuisines": ["italian", "chinese", "mexican", "indian", "japanese", "american"],
    "types": ["pizza", "pasta", "sushi", "burger", "salad", "dessert"],
    "dietary": ["vegetarian", "vegan", "gluten-free", "keto", "halal"],
    "price_range": ["budget", "moderate", "expensive"],
    "spice_level": ["mild", "medium", "hot", "extra-hot"]
}
```

#### **🏠 Real Estate Images**
```python
METADATA_CATEGORIES = {
    "property_types": ["house", "apartment", "condo", "townhouse", "studio"],
    "bedrooms": ["1", "2", "3", "4", "5+"],
    "bathrooms": ["1", "1.5", "2", "2.5", "3+"],
    "price_range": ["under_500k", "500k_1m", "1m_2m", "2m_5m", "5m+"],
    "locations": ["downtown", "suburbs", "rural", "waterfront", "mountain"]
}
```

### **Configuration Steps for New Domains**

1. **Update Metadata Categories** in `config/settings.py`
2. **Modify Database Schema** in `config/database.py`
3. **Update Query Processing** in `core/query_processor.py`
4. **Customize Filename Parsing** in `main.py`
5. **Update Collection Name** in vector database settings

## 📋 Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Redis (optional, for caching)

## 🛠️ Installation

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
# For shoe images (current implementation)
python main.py --mode index --image-dir /path/to/shoe/images

# For any other domain (after configuration)
python main.py --mode index --image-dir /path/to/your/images
```

## 🚀 Quick Start

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
# Shoe-specific examples (current implementation)
python main.py --mode search --query "red nike sneakers" --search-type text
python main.py --mode search --query "athletic shoes" --search-type hybrid

# Generic image search (works with any domain)
python main.py --mode search --query /path/to/image.jpg --search-type image
python main.py --mode search --query "red clothing" --search-type text
python main.py --mode search --query "sports car" --search-type text
```

## 📖 Usage Examples

### Web Interface
1. Open http://localhost:5000 in your browser
2. Enter a search query or upload an image
3. Select search type (text, image, hybrid, semantic)
4. Apply filters if needed
5. View results with similarity scores

### REST API

#### Text Search
```bash
# Shoe-specific example (current implementation)
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "red nike sneakers",
    "filters": {"brand": "nike", "size": "medium"},
    "limit": 10
  }'

# Generic examples (after domain configuration)
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "red sports car",
    "filters": {"make": "ferrari", "type": "coupe"},
    "limit": 10
  }'
```

#### Image Search
```bash
# Works with any image domain
curl -X POST "http://localhost:8000/search/image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/reference/image.jpg",
    "limit": 10
  }'
```

#### Hybrid Search
```bash
# Shoe-specific example
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "athletic shoes",
    "image_path": "/path/to/image.jpg",
    "filters": {"brand": "adidas"},
    "limit": 10
  }'

# Generic example
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "luxury car",
    "image_path": "/path/to/car.jpg",
    "filters": {"make": "bmw", "price_range": "expensive"},
    "limit": 10
  }'
```

#### Natural Language Search
```bash
# Shoe-specific
curl -X POST "http://localhost:8000/search/natural?query=find%20me%20some%20running%20shoes%20in%20size%20large"

# Generic (after configuration)
curl -X POST "http://localhost:8000/search/natural?query=show%20me%20red%20sports%20cars%20under%2050k"
```

### Python API
```python
from main import RAGSystem

# Initialize system
rag = RAGSystem(vector_backend="faiss")

# Index images
result = rag.index_images("/path/to/images")
print(f"Indexed {result['indexed_count']} images")

# Shoe-specific examples (current implementation)
results = rag.search("red nike sneakers", search_type="text", limit=10)
results = rag.search("athletic shoes", search_type="hybrid", limit=10)

# Generic examples (works with any domain after configuration)
results = rag.search("red sports car", search_type="text", limit=10)
results = rag.search("luxury sedan", search_type="hybrid", limit=10)
results = rag.search("/path/to/image.jpg", search_type="image", limit=10)

# Process results
for result in results:
    print(f"Image: {result['filename']}, Score: {result['similarity_score']:.3f}")
```

## ⚙️ Configuration

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

## 🔧 API Endpoints

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

## 📊 Metadata Categories

### **Fully Customizable for Any Domain**

The system supports **any metadata categories** you define. Here are examples for different domains:

#### **🛍️ Clothing Images**
- **Types**: shirt, pants, dress, jacket, hat, shoes
- **Colors**: red, blue, green, black, white, yellow, pink
- **Sizes**: XS, S, M, L, XL, XXL
- **Brands**: nike, adidas, zara, h&m, uniqlo, other

#### **🚗 Car Images**
- **Makes**: toyota, honda, ford, bmw, mercedes, audi, tesla
- **Types**: sedan, suv, truck, coupe, hatchback, convertible
- **Colors**: red, blue, black, white, silver, gray
- **Years**: 2020, 2021, 2022, 2023, 2024

#### **🍕 Food Images**
- **Cuisines**: italian, chinese, mexican, indian, japanese, american
- **Types**: pizza, pasta, sushi, burger, salad, dessert
- **Dietary**: vegetarian, vegan, gluten-free, keto, halal
- **Price Range**: budget, moderate, expensive

#### **👟 Shoe Images (Current Example)**
- **Patterns**: zigzag, circular, square, diamond, brand_logo, other
- **Shapes**: round, square, oval, irregular, elongated
- **Sizes**: small, medium, large, extra_large
- **Brands**: nike, adidas, puma, converse, vans, reebok, new_balance, asics, under_armour, jordan, other

### **How to Customize**
Simply update the `METADATA_CATEGORIES` in `config/settings.py` with your domain-specific attributes. The system will automatically adapt to your new categories!

## 🎯 Advanced Features

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

## 🐛 Troubleshooting

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI CLIP for multi-modal embeddings
- Facebook AI Research for FAISS
- Hugging Face for transformers
- The open-source community for various dependencies

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for error details

---

## 🎯 **Key Takeaways**

- **✅ Generic RAG System**: Works with any type of images (shoes, clothing, cars, food, etc.)
- **✅ Current Implementation**: Shoe image search (fully configurable)
- **✅ Easy Adaptation**: Modify configuration files for any domain
- **✅ Production Ready**: Web interface, REST API, and comprehensive features
- **✅ Scalable**: Vector database support with FAISS/ChromaDB

## 🔧 **Quick Domain Switch**

To adapt for a new domain:
1. Update `METADATA_CATEGORIES` in `config/settings.py`
2. Modify database schema in `config/database.py`
3. Update query processing in `core/query_processor.py`
4. Customize filename parsing in `main.py`
5. Rebuild the vector index

**Ready to use with any image domain!** 🚀
