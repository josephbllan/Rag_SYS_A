# RAG System Directory Structure Guide
## Comprehensive Academic Documentation

**Author:** Research Software Engineer  
**Date:** October 2025  
**Version:** 2.0  
**Purpose:** Academic documentation of RAG system architecture for Oxford University

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Architecture](#2-overall-architecture)
3. [Core System Layer](#3-core-system-layer)
4. [Domain Layer](#4-domain-layer)
5. [Application Layer](#5-application-layer)
6. [Infrastructure Layer](#6-infrastructure-layer)
7. [Presentation Layer](#7-presentation-layer)
8. [Utility Layer](#8-utility-layer)
9. [Configuration Layer](#9-configuration-layer)
10. [File Inventory](#10-file-inventory)
11. [Dependencies Map](#11-dependencies-map)
12. [Future Roadmap](#12-future-roadmap)
13. [Naming Conventions](#13-naming-conventions)
14. [Design Patterns Reference](#14-design-patterns-reference)

---

## 1. Introduction

This document provides a comprehensive academic analysis of the Retrieval-Augmented Generation (RAG) system directory structure. The system implements a multi-modal search platform that fuses text and image understanding to support high-quality retrieval and downstream reasoning. The guide aims to be both descriptive (what exists today) and prescriptive (how to extend it systematically), mapping concrete modules to architectural concepts so that contributors can navigate, refactor, and extend the codebase confidently.

### 1.1 System Overview

The codebase follows a layered architecture informed by Domain-Driven Design (DDD) and Clean Architecture. Conceptually, the Core layer implements the essential competencies of the system—embeddings, query understanding, vector-store integration, and search orchestration. The Domain layer captures the ubiquitous language of the problem space through enums, value objects, models, and interfaces that are framework-agnostic and test-friendly. The Application layer coordinates use cases via services, delegating business rules to the Domain and technical concerns to Infrastructure. The Presentation layer exposes HTTP endpoints (FastAPI) and a small web interface (Flask) to demonstrate and operate the system. Configuration centralizes environment and database setup, while Utilities and Templates provide cross-cutting scaffolding for future enhancements.

### 1.2 Architectural Principles Applied

- **Separation of Concerns**: Each module owns a single, explicit responsibility (e.g., `core/embeddings.py` only deals with representation learning strategies).
- **Dependency Inversion**: High-level policy (services, controllers) depend on abstractions defined in Domain (interfaces, base classes), not on concrete infrastructure.
- **Interface Segregation**: Narrow interfaces and protocols keep components independently testable and swappable (e.g., vector DB backends).
- **Single Responsibility**: Classes and functions are designed to have one reason to change, which simplifies maintenance and refactoring.
- **Open/Closed Principle**: The Strategy and Factory patterns enable extension by composition without modifying existing code paths.

### 1.3 Technology Stack

- **Language**: Python 3.12+
- **API Framework**: FastAPI (REST API)
- **Web Framework**: Flask (Web interface for demos/tools)
- **ML/AI**: PyTorch; multi-modal embeddings via CLIP/ResNet; Sentence Transformers for text
- **Vector Databases**: FAISS, ChromaDB (plug-and-play via adapter-like abstractions)
- **Auth/Security**: JWT for stateless auth; password handling utilities
- **Data Validation**: Pydantic models for request/response and domain entities
- **Design Patterns**: Factory, Strategy, Observer, Adapter, Singleton (see `patterns/`)

### 1.4 Document Scope and Outcomes

This guide inventories all files and directories under `rag/`, explains their roles within each architectural layer, and provides an import/dependency map to clarify coupling and data flow. It also identifies extension points and empty/planned folders—such as `core/models/`, `repositories/`, `infrastructure/adapters/`, `presentation/`, `utils/decorators`, `utils/validators`, and `templates/`—to make the roadmap concrete. Readers should leave with:

- A precise mental model of how search is orchestrated end-to-end
- Knowledge of where to implement new features or integrations
- Conventions for naming, layering, and pattern usage to maintain consistency

While the focus is the `rag/` package, the broader repository includes scripts and datasets that are out of scope for this document. The emphasis here is on maintainable, extensible architecture for a multi-modal RAG system that can evolve to new data types, retrieval strategies, and serving environments.

---

## 2. Overall Architecture

### 2.1 Layered Architecture Overview

At a high level, the system is decomposed into seven cooperating layers:

1. **Core**: Embedding strategies, query parsing/understanding, vector database integration, and search orchestration.
2. **Domain**: Types, enums, value objects, interfaces, and domain models that embody business meaning without framework dependencies.
3. **Application**: Use case coordination via services (e.g., indexing) that orchestrate domain logic and infrastructure.
4. **Infrastructure**: Cross-cutting patterns and (planned) adapters for external systems or pluggable backends.
5. **Presentation**: HTTP API (FastAPI app factory, controllers, routes, middleware, security) and a lightweight web UI.
6. **Configuration**: Settings and database initialization, consolidating environment-specific configuration.
7. **Utilities/Templates**: Currently placeholders for decorators, validators, and HTML templates.

The layering enforces that dependencies point inward: Presentation → Application → Domain, with Core straddling domain logic and algorithmic functionality. Infrastructure remains swappable and referenced through abstractions.

### 2.2 Component Interaction

- A client issues a request to the FastAPI app (`api/main.py`).
- Routing dispatches to a versioned endpoint (`api/routes/v1/*.py`) and controller (`api/controllers/*`).
- Controllers rely on `dependencies.py` to resolve services and domain abstractions.
- Services (e.g., `services/indexing_service.py`) orchestrate workflows by calling Core capabilities: embeddings, vector DB operations, and search engine coordination.
- The search engine (`core/search_engine.py`) consults the query processor (`core/query_processor.py`) to normalize queries, select embedding strategies, and determine retrieval parameters.
- Embedding backends (`core/embeddings.py`) produce vector representations using CLIP/ResNet for images and Sentence Transformers for text.
- Vector operations (`core/vector_db.py`) read/write to FAISS/ChromaDB through a unified interface.
- Responses are shaped via Domain models and returned through controllers to the client.

### 2.3 Design Patterns in Context

- **Factory** (in `patterns/factory.py`) constructs strategies or services without leaking creation logic into callers.
- **Strategy** (in `patterns/strategy.py`) encapsulates interchangeable algorithms, such as embedding selection or ranking policies.
- **Observer** (in `patterns/observer.py`) enables event-driven hooks (e.g., indexing completed) without tight coupling.
- **Adapter** (in `patterns/adapter.py` and planned `infrastructure/adapters/`) normalizes external APIs (e.g., vector DBs).
- **Singleton** (in `patterns/singleton.py`) restricts certain global coordinators or caches to a single process-wide instance when appropriate.

### 2.4 Cross-Cutting Concerns

- **Security**: `api/security/jwt_handler.py` and `password_handler.py` centralize authentication and credential handling.
- **Middleware**: `api/middleware/` provides structured logging and error handling for consistent observability.
- **Configuration**: `config/settings.py` and `config/database.py` encapsulate environment and DB lifecycle.

### 2.5 Evolution and Extensibility

The architecture anticipates growth:

- New embedding models can be added under `core/embeddings.py` and selected through Strategy/Factory wiring.
- Additional vector backends integrate via `core/vector_db.py` and (planned) `infrastructure/adapters/`.
- Domain models can evolve independently of the API surfaces; controllers remain thin and delegating.
- Repositories (planned) will abstract persistence, further decoupling Application from Infrastructure.

These choices allow the system to scale in complexity while preserving maintainability and testability.

---

## 3. Core System Layer

The Core layer implements the multi-modal representation learning, semantic query understanding, vector database abstraction, and the orchestration logic that fuses these into coherent search functionality.

### 3.1 `core/embeddings.py`

Responsibilities:
- Image embeddings via CLIP and ResNet (`ImageEmbedder`)
- Text embeddings via Sentence-Transformers and CLIP text encoder (`TextEmbedder`)
- Unified multi-modal interface (`MultiModalEmbedder`)
- Caching and batch APIs (`EmbeddingManager`), plus utility metrics (cosine similarity, Euclidean distance) and simple metadata-derived embeddings for hybrid search

Key classes/functions:
- `ImageEmbedder(model_type: str)`: Loads CLIP or ResNet, exposes `encode_image(path)` and `encode_images_batch(paths)`; normalizes CLIP embeddings; applies global average pooling for ResNet features.
- `TextEmbedder(model_name: str)`: Wraps SentenceTransformer with `encode_text` and `encode_texts_batch`; returns dense vectors sized by the model.
- `MultiModalEmbedder`: Convenience wrapper producing dictionaries of embeddings for both modalities and both model families (CLIP/ResNet for images; CLIP/SBERT for text).
- `EmbeddingManager(cache_dir)`: High-level façade that adds on-disk caching with `get_image_embedding`, `get_text_embedding`, and batch processing helpers. Supports switching model types and toggling caching.
- Utilities: `normalize_embedding`, `cosine_similarity`, `euclidean_distance`, `create_embedding_from_metadata` for simple hybrid signals from categorical metadata.

Notes on design:
- Strategy selection for embeddings is parameterized (`model_type`, `model_name`) and can be grown without breaking call sites. Normalization is enforced for cosine metrics.
- Device selection and dimensions are centralized via `config.settings.MODEL_CONFIG`.

### 3.2 `core/vector_db.py`

Responsibilities:
- Unified abstraction over FAISS and ChromaDB backends (`VectorDatabase`)
- Add/search/update/delete operations, metadata handling, and statistics reporting
- Factory method `create_vector_db(backend, collection_name)`

Key behaviors:
- FAISS: Builds an `IndexIVFFlat` over `IndexFlatL2`. Trains on up to 1,000 vectors, supports `nprobe` for recall/latency tuning. Converts L2 distances to similarity scores via `1/(1+d)` for ranking uniformity. Metadata is persisted to pickle alongside `.faiss` index.
- ChromaDB: Uses persistent client with collection per dataset. Supports where-clauses (exact or `$in` filtering), returns cosine distances converted to similarity (`1 - distance`).
- Filtering: `_apply_filters` runs post-search metadata filtering for FAISS. Chroma uses the collection’s `where` directly.
- Maintenance: `rebuild_index` and `clear_database` provide operational controls. `get_stats` returns essential observability metrics.

Design considerations:
- The adapter-like `VectorDatabase` enables swapping vector backends without touching callers. Validation helpers (`validate_vector`, `get_embedding_dimension`) keep dimensions consistent with model configs.

### 3.3 `core/query_processor.py`

Responsibilities:
- Transform natural language into structured `QueryIntent` with `query_type`, `filters`, `limit`, and optional `image_path`
- Regex-driven extraction for brand, pattern, shape, size, color, style; brand alias normalization; synonym expansion for broader recall
- Query validation and variation generation for coverage

Key components:
- `QueryIntent` dataclass: canonical representation of parsed intent.
- `QueryProcessor`:
  - Normalization: case-folding, whitespace compaction, contraction expansion
  - Extraction: `_extract_search_terms`, `_extract_filters`, `_extract_image_path`, thresholds and limits
  - Heuristics: `_determine_query_type` to choose among text/image/hybrid/metadata
  - Synonyms and aliases: richer matching without external services
- Convenience helpers: `create_query_processor`, `process_natural_query`.

Limitations and extensions:
- Current parsing is rule-based; can be augmented with lightweight NLP (e.g., spaCy) or learned intent classifiers. Color/style are optional filters and can be expanded from `config.settings`.

### 3.4 `core/search_engine.py`

Responsibilities:
- End-to-end search orchestration across modalities with thresholding, ranking, hybrid scoring, and logging
- Text→image search, image→image search, metadata-only search, hybrid fusion, semantic search with naive query expansion, recommendations, and similarity-based retrieval

Key flows:
- Text search: `get_text_embedding(query, "clip")` → `vector_db.search` → thresholding → logging.
- Image search: `get_image_embedding(image_path, "clip")` → `vector_db.search` → thresholding → logging.
- Hybrid: merges per-modality scores into a weighted `hybrid_score` using `SEARCH_CONFIG["hybrid_weights"]`, then sorts and truncates.
- Semantic: simple expansion (`_expand_query`) to cover synonyms/variants; aggregates matches and boosts scores.
- Stats and observability: `get_search_stats()` composes vector DB stats with recent query analytics from the SQL database.

Design notes:
- The engine defers embedding and storage choices to `EmbeddingManager` and `VectorDatabase`, enforcing inversion of control. It remains agnostic to backend specifics and can evolve scoring policies without touching underlying components.

---

## 4. Domain Layer

The Domain layer captures the ubiquitous language of the system and defines strongly typed entities, value objects, and contracts. It is framework-agnostic, enabling reuse and rigorous testing.

### 4.1 `domain/types.py`

- Centralizes type aliases and guards: `VectorType` (NumPy float32 arrays), `ModelType`, `SearchType`, `VectorBackendType`, and constants like `MAX_QUERY_LENGTH`, `DEFAULT_SIMILARITY_THRESHOLD`.
- Type guards (`is_valid_vector`, `is_valid_metadata`, `is_valid_embedding_dimension`) provide safer runtime checks in addition to static typing.

### 4.2 `domain/enums.py`

- Enumerations for domain concepts: `QueryType`, `ModelName`, `VectorBackend`, `PatternType`, `ShapeType`, `SizeType`, `BrandType`, plus utility enums for logging and export formats.
- Using string-valued enums keeps serialization and config straightforward while preserving type safety.

### 4.3 `domain/value_objects.py`

- Immutable value objects implemented with `@dataclass(frozen=True)`: `QueryIntent`, `EmbeddingResult`, `SearchScore`, `VectorIdentifier`, `CacheKey`, `ModelVersion`.
- Invariant checks in `__post_init__` ensure values remain valid (e.g., score ranges, vector dimensions, non-empty identifiers).
- Methods like `to_dict()` or `to_string()` support clean boundaries between domain and presentation layers.

### 4.4 `domain/models.py`

- Pydantic models with validation and constrained fields: `ImageMetadata`, `SearchFilters`, `SearchQuery`, `SearchResultItem`, `SearchResponse`, `EmbeddingConfig`, `VectorDatabaseConfig`, `SystemHealth`, `IndexingResult`.
- Validators sanitize input (e.g., malicious content in queries, safe image paths) and enforce constraints (dimensions, counts, enums).
- `use_enum_values=True` bridges ergonomic Python enums with JSON-friendly API responses.

### 4.5 `domain/interfaces.py`

- Protocols for dependency inversion: `IEmbeddingModel`, `IVectorDatabase`, `ISearchStrategy`, `IQueryProcessor`, `IRepository`, `ICache`, `ILogger`, `IEventPublisher`, `IValidator`, `ISerializer`, `IMetadataExtractor`, `IScorer`, `IModelFactory`, `ISearchService`.
- Each interface documents expected behavior without coupling to concrete implementations, enabling easy substitution in tests and infrastructure evolution.

### 4.6 `domain/base_classes.py`

- Abstract base classes that provide default behaviors and logging: `BaseEmbeddingModel`, `BaseVectorDatabase`, `BaseSearchStrategy`, `BaseRepository`, `BaseService`, `BaseValidator`, `BaseEventHandler`.
- These bases reduce boilerplate and standardize logging, initialization, and error handling patterns across implementations.

---

## 5. Application Layer

The Application layer orchestrates use cases by coordinating domain abstractions and core capabilities. It should remain thin, focusing on workflow sequencing and policy decisions rather than deep algorithmic logic.

### 5.1 `services/base_service.py`

- Provides a common service scaffold: lifecycle management (`initialize`, `shutdown`), health reporting, metrics recording, and a context manager API.
- Enforces initialization preconditions via `_ensure_initialized` to prevent accidental use before setup.

### 5.2 `services/indexing_service.py`

- Implements image indexing workflows, including batch traversal of directories, embedding generation, metadata extraction, and persistence in the vector database.
- Integrates with an event publisher (`patterns/observer.EventPublisher`) to emit `image_indexed`, `indexing_failed`, and `indexing_complete` events for observability.
- Produces `IndexingResult` with bounded error lists to keep logs manageable.

Planned extensions:
- A `repositories/` package to abstract persistence beyond the vector store (e.g., relational read models, audit logs) with interfaces defined in Domain.

---

## 6. Infrastructure Layer

The Infrastructure layer packages reusable design patterns and prepares for concrete adapters to external systems.

### 6.1 `patterns/`

- `factory.py`: Families of factories for models, vector databases, strategies, repositories, validators, and services. Centralizes creation logic and supports discovery/registration. Caching in `ServiceFactory` avoids repeated construction.
- `strategy.py`: Defines `SearchContext` and concrete strategies for text, image, hybrid, semantic, and metadata searches. Each strategy delegates to shared dependencies via a context, enabling runtime algorithm swaps.
- `observer.py`: Eventing with `EventPublisher` and observers for search, indexing, cache, and performance metrics. Supports both observer instances and ad-hoc handler callables per event type.
- `adapter.py`: Uniform adapters over vector DBs (`FAISSAdapter`, `ChromaDBAdapter`) and a `VectorDatabaseAdapter` façade for consistent CRUD/search/statistics.
- `singleton.py`: Thread-safe `SingletonMeta` and concrete managers (`ConfigurationManager`, `LoggerManager`, `CacheManager`, `ConnectionPoolManager`).

### 6.2 `infrastructure/adapters/` (planned)

- Reserved for concrete integrations (e.g., cloud object storage, external ML services, alternative vector stores, telemetry sinks). Adapters should conform to the Domain interfaces and/or wrap pattern-provided bases for consistency.

Design guidance:
- Keep adapters thin and stateless; push policy to Application and Domain. Favor composition over inheritance. Provide health checks and small metrics surfaces for operability.

---

## 7. Presentation Layer

The Presentation layer provides external interfaces: a production-style FastAPI app and a lightweight Flask demo.

### 7.1 `api/`

- `main.py`: Application factory `create_app()` sets up CORS, logging middleware, error handlers, and mounts `api_router` under `/api`. Startup/shutdown events provide lifecycle logs.
- `dependencies.py`: Simple DI providers for singletons (`SearchEngine`, `QueryProcessor`) and construction of `IndexingService` (currently wrapping the engine).
- `controllers/`: `base_controller.py`, `health_controller.py`, `search_controller.py` follow an MVC-ish separation between routing and business calls.
- `routes/v1/`: Versioned endpoints (`auth.py`, `search.py`, `system.py`) that bind URIs to controller actions.
- `middleware/`: `logging_middleware.py` and `error_handler.py` standardize observability and error responses.
- `security/`: `jwt_handler.py`, `password_handler.py` for auth concerns.
- `endpoints.py`: An alternate, self-contained FastAPI app exposing health, search (text/image/hybrid/semantic/natural), recommendations, metadata, analytics, system maintenance, and export endpoints. Useful for integration testing or standalone running.

### 7.2 `web/app.py`

- Flask demo interface for simple manual testing and showcasing capabilities; complements the API layer for non-API stakeholders.

### 7.3 `presentation/` (planned)

- Placeholder for future UI components, templates, and static assets aligned to the API.

---

## 8. Utility Layer

The Utility layer contains scaffolding for cross-cutting helpers and templates used across the system.

### 8.1 `utils/` (planned structure)

- `decorators/`: Intended for reusable decorators such as caching, timing, tracing, retries, and circuit breakers. These should be side-effect free and well-tested, with consistent signatures and docstrings.
- `validators/`: Intended for additional standalone validation helpers that complement Pydantic (e.g., media-specific checks, filename policies, path normalization).

### 8.2 `templates/` (planned)

- Placeholder for HTML templates (e.g., Jinja2) to support the `web/` demo or future admin UIs.

Design guidance:
- Keep utilities small and cohesive. Prefer composition and explicit injection over global state. Document decorator behavior clearly, especially around exception handling and metrics side-effects.

---

## 9. Configuration Layer

Centralizes environment, model, vector DB, and API configuration.

### 9.1 `config/settings.py`

- Defines base paths (`BASE_DIR`, `PROJECT_ROOT`), data/cache/vector directories, and ensures their existence.
- Vector DB settings for FAISS and Chroma; model configs for CLIP, Sentence Transformers, and ResNet with device selection from env.
- Search, API, Web, Image, Analytics, Cache, Export, Logging, and JWT settings collected in cohesive dicts for easy consumption across modules.
- `ENV_VARS` exposes effective environment settings; demo users included for JWT examples.

### 9.2 `config/database.py`

- SQLAlchemy engine/session setup, declarative `Base`, and concrete tables: `ShoeImage`, `SearchQuery`, `SearchResult`, `UserSession`, `SystemMetrics`.
- Utility functions `get_db()`, `create_tables()`, `drop_tables()`, `reset_database()`, and `test_connection()` for lifecycle management.
- JSON fields store embeddings and metadata in a flexible format; timestamps track ingestion and usage.

---

## 10. File Inventory

This section summarizes implemented modules in `rag/` and their primary responsibilities. Empty/planned directories are noted for future work.

- `api/`: app factory, DI, controllers, routes (v1), middleware, security, alternate endpoints module.
- `config/`: settings and database models/utilities.
- `core/`: embeddings, query processor, search engine, vector DB abstraction, planned `models/`.
- `domain/`: enums, types, value objects, Pydantic models, interfaces, base classes.
- `patterns/`: factory, strategy, observer, adapter, singleton.
- `services/`: base service, indexing service.
- `web/`: Flask demo app.
- `templates/`, `utils/` (decorators, validators), `infrastructure/adapters/`, `repositories/`, `presentation/`: planned scaffolding.

Note: Several modules include command-line test harnesses under `if __name__ == "__main__":` to validate initialization.

---

## 11. Dependencies Map

- Presentation depends on Application and Domain abstractions; routes/controllers import services and processors.
- Application depends on Domain (models/interfaces) and Core (engines, vector DB façade).
- Core depends on Configuration for settings and on external ML/vector libs; exposes pure-Python APIs up-stack.
- Infrastructure provides patterns used by all layers without creating cyclic imports; adapters are consumed via Domain interfaces.
- Configuration is imported by Core, Presentation, and sometimes Services but imports nothing from them.

Data flow:
- Request → Route → Controller → Service → Core (QueryProcessor, EmbeddingManager, VectorDatabase) → Response via Domain models.

---

## 12. Future Roadmap

- Implement `core/models/` to hold algorithmic policies and typed result aggregations.
- Add `repositories/` to abstract persistence beyond vector stores; back with SQLAlchemy or a document DB.
- Populate `infrastructure/adapters/` for cloud storage (S3), telemetry (OpenTelemetry), and alternate vector stores.
- Build `presentation/` UI components and wire `templates/` to `web/`.
- Expand `utils/decorators` for caching, retries, and tracing; add `utils/validators` for image/file checks.
- Harden security: role-based access, refresh token rotation, and secrets management.

---

## 13. Naming Conventions

- Files and modules use snake_case; classes use PascalCase; functions and variables use snake_case.
- Enums are singular nouns; value objects end with nouns (`...Result`, `...Identifier`).
- Interfaces prefixed with `I...` in Protocols; abstract bases with `Base...`.
- Config keys are lowercase snake_case; environment flags are UPPER_SNAKE_CASE.

---

## 14. Design Patterns Reference

- Factory: object creation centralized for models, databases, strategies, services, repositories, validators.
- Strategy: interchangeable search algorithms (text, image, hybrid, semantic, metadata) with a `SearchContext`.
- Observer: event publisher with typed observers for search, indexing, cache, performance.
- Adapter: normalized vector DB access across FAISS/Chroma and planned external systems.
- Singleton: configuration, logging, caching, and connection pools with thread-safe meta.

These patterns enforce extensibility, reduce coupling, and enable safe substitution in tests and deployments.