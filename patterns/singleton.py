"""
Singleton Pattern Implementation
Ensures only one instance of a class exists throughout the application
"""
from typing import Dict, Any, Optional
from threading import Lock
import logging


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass
    """
    _instances: Dict[type, Any] = {}
    _lock: Lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    """
    Base Singleton class that can be inherited
    """
    pass


class ConfigurationManager(Singleton):
    """
    Singleton Configuration Manager
    Manages application-wide configuration
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config: Dict[str, Any] = {}
            self._logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True
            self._logger.info("ConfigurationManager initialized")
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
        self._logger.debug(f"Configuration set: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self._config.update(config)
        self._logger.debug(f"Configuration updated with {len(config)} items")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()
    
    def clear(self) -> None:
        """Clear all configuration"""
        self._config.clear()
        self._logger.warning("Configuration cleared")
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return key in self._config
    
    def remove(self, key: str) -> None:
        """Remove configuration key"""
        if key in self._config:
            del self._config[key]
            self._logger.debug(f"Configuration removed: {key}")


class LoggerManager(Singleton):
    """
    Singleton Logger Manager
    Manages application-wide logging
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._loggers: Dict[str, logging.Logger] = {}
            self._initialized = True
            self._default_level = logging.INFO
            self._default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def get_logger(
        self, 
        name: str, 
        level: Optional[int] = None,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """Get or create logger"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level or self._default_level)
            
            # Add handler if not present
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(format_string or self._default_format)
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_default_level(self, level: int) -> None:
        """Set default logging level"""
        self._default_level = level
        for logger in self._loggers.values():
            logger.setLevel(level)
    
    def set_default_format(self, format_string: str) -> None:
        """Set default log format"""
        self._default_format = format_string
        formatter = logging.Formatter(format_string)
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.setFormatter(formatter)


class CacheManager(Singleton):
    """
    Singleton Cache Manager
    Manages application-wide caching
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._cache: Dict[str, Any] = {}
            self._ttl: Dict[str, float] = {}
            self._logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True
            self._logger.info("CacheManager initialized")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value"""
        self._cache[key] = value
        if ttl is not None:
            import time
            self._ttl[key] = time.time() + ttl
        self._logger.debug(f"Cache set: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value"""
        if key in self._cache:
            # Check TTL
            if key in self._ttl:
                import time
                if time.time() > self._ttl[key]:
                    self.delete(key)
                    return default
            return self._cache[key]
        return default
    
    def delete(self, key: str) -> bool:
        """Delete cache value"""
        if key in self._cache:
            del self._cache[key]
            if key in self._ttl:
                del self._ttl[key]
            self._logger.debug(f"Cache deleted: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache"""
        self._cache.clear()
        self._ttl.clear()
        self._logger.info("Cache cleared")
    
    def exists(self, key: str) -> bool:
        """Check if cache key exists"""
        return key in self._cache
    
    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)


class ConnectionPoolManager(Singleton):
    """
    Singleton Connection Pool Manager
    Manages database and service connections
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._pools: Dict[str, Any] = {}
            self._logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True
            self._logger.info("ConnectionPoolManager initialized")
    
    def register_pool(self, name: str, pool: Any) -> None:
        """Register a connection pool"""
        self._pools[name] = pool
        self._logger.info(f"Connection pool registered: {name}")
    
    def get_pool(self, name: str) -> Optional[Any]:
        """Get connection pool"""
        return self._pools.get(name)
    
    def close_pool(self, name: str) -> None:
        """Close and remove connection pool"""
        if name in self._pools:
            pool = self._pools[name]
            if hasattr(pool, 'close'):
                pool.close()
            del self._pools[name]
            self._logger.info(f"Connection pool closed: {name}")
    
    def close_all(self) -> None:
        """Close all connection pools"""
        for name in list(self._pools.keys()):
            self.close_pool(name)
        self._logger.info("All connection pools closed")

