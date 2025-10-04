"""
Factory Pattern Implementation
Creates objects without exposing creation logic
"""
from typing import Dict, Any, Type, Optional, Callable
from abc import ABC, abstractmethod
import logging

from ..domain.enums import ModelName, VectorBackend, QueryType
from ..domain.interfaces import IEmbeddingModel, IVectorDatabase, ISearchStrategy
from ..domain.base_classes import BaseEmbeddingModel, BaseVectorDatabase, BaseSearchStrategy


class AbstractFactory(ABC):
    """Abstract Factory base class"""
    
    @abstractmethod
    def create(self, *args, **kwargs) -> Any:
        """Create an object"""
        pass


class ModelFactory:
    """
    Factory for creating embedding models
    Implements Factory Method pattern
    """
    
    _model_classes: Dict[str, Type[BaseEmbeddingModel]] = {}
    _logger = logging.getLogger("ModelFactory")
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseEmbeddingModel]) -> None:
        """Register a new model type"""
        cls._model_classes[model_type] = model_class
        cls._logger.info(f"Registered model: {model_type}")
    
    @classmethod
    def create_model(
        cls, 
        model_type: str, 
        model_name: str, 
        device: str = "cpu",
        **kwargs
    ) -> BaseEmbeddingModel:
        """Create an embedding model"""
        if model_type not in cls._model_classes:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._model_classes.keys())}")
        
        model_class = cls._model_classes[model_type]
        cls._logger.info(f"Creating model: {model_type} ({model_name})")
        
        try:
            model = model_class(model_name=model_name, device=device, **kwargs)
            return model
        except Exception as e:
            cls._logger.error(f"Failed to create model {model_type}: {e}")
            raise
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get list of supported model types"""
        return list(cls._model_classes.keys())
    
    @classmethod
    def create_clip_model(cls, model_name: str = "ViT-B/32", device: str = "cpu") -> BaseEmbeddingModel:
        """Convenience method to create CLIP model"""
        return cls.create_model("clip", model_name, device)
    
    @classmethod
    def create_resnet_model(cls, model_name: str = "resnet50", device: str = "cpu") -> BaseEmbeddingModel:
        """Convenience method to create ResNet model"""
        return cls.create_model("resnet", model_name, device)
    
    @classmethod
    def create_sentence_transformer(cls, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> BaseEmbeddingModel:
        """Convenience method to create Sentence Transformer"""
        return cls.create_model("sentence_transformer", model_name, device)


class VectorDatabaseFactory:
    """
    Factory for creating vector databases
    Implements Abstract Factory pattern
    """
    
    _db_classes: Dict[str, Type[BaseVectorDatabase]] = {}
    _logger = logging.getLogger("VectorDatabaseFactory")
    
    @classmethod
    def register_database(cls, backend: str, db_class: Type[BaseVectorDatabase]) -> None:
        """Register a new database backend"""
        cls._db_classes[backend] = db_class
        cls._logger.info(f"Registered vector database: {backend}")
    
    @classmethod
    def create_database(
        cls, 
        backend: str, 
        dimension: int,
        collection_name: str = "default",
        **kwargs
    ) -> BaseVectorDatabase:
        """Create a vector database"""
        if backend not in cls._db_classes:
            raise ValueError(f"Unknown database backend: {backend}. Available: {list(cls._db_classes.keys())}")
        
        db_class = cls._db_classes[backend]
        cls._logger.info(f"Creating vector database: {backend}")
        
        try:
            database = db_class(dimension=dimension, collection_name=collection_name, **kwargs)
            return database
        except Exception as e:
            cls._logger.error(f"Failed to create database {backend}: {e}")
            raise
    
    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backends"""
        return list(cls._db_classes.keys())
    
    @classmethod
    def create_faiss_database(cls, dimension: int, collection_name: str = "default", **kwargs) -> BaseVectorDatabase:
        """Convenience method to create FAISS database"""
        return cls.create_database("faiss", dimension, collection_name, **kwargs)
    
    @classmethod
    def create_chroma_database(cls, dimension: int, collection_name: str = "default", **kwargs) -> BaseVectorDatabase:
        """Convenience method to create ChromaDB database"""
        return cls.create_database("chroma", dimension, collection_name, **kwargs)


class SearchStrategyFactory:
    """
    Factory for creating search strategies
    Implements Factory Method pattern
    """
    
    _strategy_classes: Dict[str, Type[BaseSearchStrategy]] = {}
    _logger = logging.getLogger("SearchStrategyFactory")
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: Type[BaseSearchStrategy]) -> None:
        """Register a new search strategy"""
        cls._strategy_classes[strategy_type] = strategy_class
        cls._logger.info(f"Registered search strategy: {strategy_type}")
    
    @classmethod
    def create_strategy(
        cls, 
        strategy_type: str,
        **kwargs
    ) -> BaseSearchStrategy:
        """Create a search strategy"""
        if strategy_type not in cls._strategy_classes:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(cls._strategy_classes.keys())}"
            )
        
        strategy_class = cls._strategy_classes[strategy_type]
        cls._logger.info(f"Creating search strategy: {strategy_type}")
        
        try:
            strategy = strategy_class(**kwargs)
            return strategy
        except Exception as e:
            cls._logger.error(f"Failed to create strategy {strategy_type}: {e}")
            raise
    
    @classmethod
    def get_supported_strategies(cls) -> list[str]:
        """Get list of supported strategies"""
        return list(cls._strategy_classes.keys())
    
    @classmethod
    def create_text_strategy(cls, **kwargs) -> BaseSearchStrategy:
        """Convenience method to create text search strategy"""
        return cls.create_strategy("text", **kwargs)
    
    @classmethod
    def create_image_strategy(cls, **kwargs) -> BaseSearchStrategy:
        """Convenience method to create image search strategy"""
        return cls.create_strategy("image", **kwargs)
    
    @classmethod
    def create_hybrid_strategy(cls, **kwargs) -> BaseSearchStrategy:
        """Convenience method to create hybrid search strategy"""
        return cls.create_strategy("hybrid", **kwargs)


class RepositoryFactory:
    """
    Factory for creating repositories
    Implements Factory Method pattern
    """
    
    _repository_classes: Dict[str, Type] = {}
    _logger = logging.getLogger("RepositoryFactory")
    
    @classmethod
    def register_repository(cls, entity_type: str, repository_class: Type) -> None:
        """Register a repository for an entity type"""
        cls._repository_classes[entity_type] = repository_class
        cls._logger.info(f"Registered repository: {entity_type}")
    
    @classmethod
    def create_repository(cls, entity_type: str, **kwargs) -> Any:
        """Create a repository"""
        if entity_type not in cls._repository_classes:
            raise ValueError(
                f"Unknown entity type: {entity_type}. "
                f"Available: {list(cls._repository_classes.keys())}"
            )
        
        repository_class = cls._repository_classes[entity_type]
        cls._logger.info(f"Creating repository: {entity_type}")
        
        try:
            repository = repository_class(**kwargs)
            return repository
        except Exception as e:
            cls._logger.error(f"Failed to create repository {entity_type}: {e}")
            raise
    
    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported entity types"""
        return list(cls._repository_classes.keys())


class ValidatorFactory:
    """
    Factory for creating validators
    Implements Factory Method pattern
    """
    
    _validator_classes: Dict[str, Type] = {}
    _logger = logging.getLogger("ValidatorFactory")
    
    @classmethod
    def register_validator(cls, validator_type: str, validator_class: Type) -> None:
        """Register a validator"""
        cls._validator_classes[validator_type] = validator_class
        cls._logger.info(f"Registered validator: {validator_type}")
    
    @classmethod
    def create_validator(cls, validator_type: str, **kwargs) -> Any:
        """Create a validator"""
        if validator_type not in cls._validator_classes:
            raise ValueError(
                f"Unknown validator type: {validator_type}. "
                f"Available: {list(cls._validator_classes.keys())}"
            )
        
        validator_class = cls._validator_classes[validator_type]
        cls._logger.info(f"Creating validator: {validator_type}")
        
        try:
            validator = validator_class(**kwargs)
            return validator
        except Exception as e:
            cls._logger.error(f"Failed to create validator {validator_type}: {e}")
            raise
    
    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported validator types"""
        return list(cls._validator_classes.keys())


class ServiceFactory:
    """
    Factory for creating services
    Implements Factory Method pattern with dependency injection
    """
    
    _service_classes: Dict[str, Type] = {}
    _service_instances: Dict[str, Any] = {}  # Cached instances
    _logger = logging.getLogger("ServiceFactory")
    
    @classmethod
    def register_service(cls, service_name: str, service_class: Type) -> None:
        """Register a service"""
        cls._service_classes[service_name] = service_class
        cls._logger.info(f"Registered service: {service_name}")
    
    @classmethod
    def create_service(cls, service_name: str, cached: bool = True, **kwargs) -> Any:
        """Create a service (with optional caching)"""
        if cached and service_name in _service_instances:
            cls._logger.debug(f"Returning cached service: {service_name}")
            return cls._service_instances[service_name]
        
        if service_name not in cls._service_classes:
            raise ValueError(
                f"Unknown service: {service_name}. "
                f"Available: {list(cls._service_classes.keys())}"
            )
        
        service_class = cls._service_classes[service_name]
        cls._logger.info(f"Creating service: {service_name}")
        
        try:
            service = service_class(**kwargs)
            if cached:
                cls._service_instances[service_name] = service
            return service
        except Exception as e:
            cls._logger.error(f"Failed to create service {service_name}: {e}")
            raise
    
    @classmethod
    def get_service(cls, service_name: str) -> Optional[Any]:
        """Get cached service instance"""
        return cls._service_instances.get(service_name)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear service cache"""
        cls._service_instances.clear()
        cls._logger.info("Service cache cleared")

