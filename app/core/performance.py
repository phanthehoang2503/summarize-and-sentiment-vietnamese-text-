"""
Performance Optimization Utilities
Simple utilities to improve application performance
"""
import hashlib
import time
from functools import wraps
from typing import Any, Dict, Optional, Callable


class SimpleCache:
    """Simple in-memory cache with TTL and size limits"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self._cache.items()
            if current_time - data['timestamp'] > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._cleanup_expired()
        
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] <= self.ttl_seconds:
                return entry['value']
            else:
                del self._cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self._cleanup_expired()
        
        # Simple eviction strategy - remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        self._cleanup_expired()
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_ratio': 0  # Could implement hit tracking if needed
        }


# Global cache instance for API responses
api_response_cache = SimpleCache(max_size=50, ttl_seconds=600)  # 10 minute TTL


def cache_api_response(cache_instance: SimpleCache = None):
    """
    Decorator to cache API responses based on request content
    """
    if cache_instance is None:
        cache_instance = api_response_cache
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            # For text analysis, use the text content as primary key component
            cache_key_data = str(args) + str(sorted(kwargs.items()))
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Only cache successful results
            if isinstance(result, dict) and result.get('success', False):
                cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


def normalize_text_for_caching(text: str) -> str:
    """
    Normalize text for caching purposes to improve cache hit rates
    """
    # Basic normalization
    normalized = text.strip().lower()
    
    # Remove multiple spaces
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Truncate very long texts for cache key generation
    if len(normalized) > 1000:
        normalized = normalized[:1000] + "..."
    
    return normalized


def get_text_cache_key(text: str, analysis_type: str = "") -> str:
    """
    Generate consistent cache key for text analysis requests
    """
    normalized_text = normalize_text_for_caching(text)
    cache_data = f"{analysis_type}:{normalized_text}"
    return hashlib.md5(cache_data.encode()).hexdigest()


class PerformanceTracker:
    """Simple performance tracking for optimization insights"""
    
    def __init__(self):
        self._timings: Dict[str, list] = {}
    
    def time_operation(self, operation_name: str):
        """Context manager to time operations"""
        return TimingContext(self, operation_name)
    
    def record_timing(self, operation_name: str, duration: float):
        """Record timing for an operation"""
        if operation_name not in self._timings:
            self._timings[operation_name] = []
        self._timings[operation_name].append(duration)
        
        # Keep only last 100 timings per operation
        if len(self._timings[operation_name]) > 100:
            self._timings[operation_name] = self._timings[operation_name][-100:]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}
        for op_name, timings in self._timings.items():
            if timings:
                stats[op_name] = {
                    'count': len(timings),
                    'avg': sum(timings) / len(timings),
                    'min': min(timings),
                    'max': max(timings),
                    'total': sum(timings)
                }
        return stats


class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, tracker: PerformanceTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.tracker.record_timing(self.operation_name, duration)


# Global performance tracker
performance_tracker = PerformanceTracker()