"""
Observer Pattern Implementation
Defines one-to-many dependency between objects
"""
from typing import List, Dict, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import logging


class Observer(ABC):
    """Abstract Observer class"""
    
    @abstractmethod
    def update(self, subject: 'Subject', event_type: str, data: Dict[str, Any]) -> None:
        """Called when subject state changes"""
        pass


class Subject:
    """
    Subject class that observers can subscribe to
    Implements the Observable part of Observer pattern
    """
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
            self._logger.debug(f"Observer attached: {observer.__class__.__name__}")
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
            self._logger.debug(f"Observer detached: {observer.__class__.__name__}")
    
    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all observers"""
        self._logger.debug(f"Notifying {len(self._observers)} observers of event: {event_type}")
        for observer in self._observers:
            try:
                observer.update(self, event_type, data)
            except Exception as e:
                self._logger.error(f"Error notifying observer {observer.__class__.__name__}: {e}")


class EventPublisher(Subject):
    """
    Event Publisher using Observer pattern
    Allows publishing and subscribing to typed events
    """
    
    def __init__(self):
        super().__init__()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        if handler not in self._event_handlers[event_type]:
            self._event_handlers[event_type].append(handler)
            self._logger.info(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type"""
        if event_type in self._event_handlers:
            if handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)
                self._logger.info(f"Unsubscribed from event: {event_type}")
    
    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow()
        }
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        self._logger.info(f"Publishing event: {event_type}")
        
        # Notify specific event handlers
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    self._logger.error(f"Error in event handler: {e}")
        
        # Notify general observers
        self.notify(event_type, data)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history"""
        if event_type:
            filtered = [e for e in self._event_history if e['type'] == event_type]
            return filtered[-limit:]
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history"""
        self._event_history.clear()
        self._logger.info("Event history cleared")


class SearchEventObserver(Observer):
    """Observer for search events"""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._search_count = 0
    
    def update(self, subject: Subject, event_type: str, data: Dict[str, Any]) -> None:
        """Handle search events"""
        if event_type == "search_executed":
            self._search_count += 1
            self._logger.info(
                f"Search executed: query='{data.get('query', '')}', "
                f"results={data.get('result_count', 0)}, "
                f"total_searches={self._search_count}"
            )
        elif event_type == "search_failed":
            self._logger.error(
                f"Search failed: query='{data.get('query', '')}', "
                f"error='{data.get('error', '')}'"
            )
    
    @property
    def search_count(self) -> int:
        """Get total search count"""
        return self._search_count


class IndexingEventObserver(Observer):
    """Observer for indexing events"""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._indexed_count = 0
        self._failed_count = 0
    
    def update(self, subject: Subject, event_type: str, data: Dict[str, Any]) -> None:
        """Handle indexing events"""
        if event_type == "image_indexed":
            self._indexed_count += 1
            if self._indexed_count % 100 == 0:
                self._logger.info(f"Indexed {self._indexed_count} images")
        elif event_type == "indexing_failed":
            self._failed_count += 1
            self._logger.warning(
                f"Indexing failed for: {data.get('filename', '')}, "
                f"error: {data.get('error', '')}"
            )
        elif event_type == "indexing_complete":
            self._logger.info(
                f"Indexing complete: {data.get('total', 0)} total, "
                f"{data.get('successful', 0)} successful, "
                f"{data.get('failed', 0)} failed"
            )
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get indexing statistics"""
        return {
            'indexed': self._indexed_count,
            'failed': self._failed_count
        }


class CacheEventObserver(Observer):
    """Observer for cache events"""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._hits = 0
        self._misses = 0
    
    def update(self, subject: Subject, event_type: str, data: Dict[str, Any]) -> None:
        """Handle cache events"""
        if event_type == "cache_hit":
            self._hits += 1
        elif event_type == "cache_miss":
            self._misses += 1
        elif event_type == "cache_cleared":
            self._logger.info("Cache cleared")
            self._hits = 0
            self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate
        }


class PerformanceEventObserver(Observer):
    """Observer for performance metrics"""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._execution_times: List[float] = []
        self._max_samples = 1000
    
    def update(self, subject: Subject, event_type: str, data: Dict[str, Any]) -> None:
        """Handle performance events"""
        if event_type == "operation_completed":
            execution_time = data.get('execution_time', 0.0)
            self._execution_times.append(execution_time)
            
            if len(self._execution_times) > self._max_samples:
                self._execution_times.pop(0)
            
            if execution_time > 5.0:  # Log slow operations
                self._logger.warning(
                    f"Slow operation detected: {data.get('operation', '')} "
                    f"took {execution_time:.2f}s"
                )
    
    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self._execution_times:
            return 0.0
        return sum(self._execution_times) / len(self._execution_times)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._execution_times:
            return {'count': 0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'count': len(self._execution_times),
            'avg': self.avg_execution_time,
            'min': min(self._execution_times),
            'max': max(self._execution_times)
        }

