"""
Performance Optimization & Monitoring

Provides performance profiling, caching strategies, query optimization,
and resource management for high-performance operations.

Features:
- Function performance profiling
- Automatic caching with TTL
- Query result caching
- Resource pooling
- Memory optimization
- Performance metrics

Usage:
    from app.services.performance import profile, cached, optimize_query
    
    @profile
    @cached(ttl=300)
    async def expensive_operation():
        # Your code here
        pass
"""

import asyncio
import time
import functools
import hashlib
import json
import logging
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


# Sentinel value to distinguish "not in cache" from "cached None"
_CACHE_MISS = object()


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function"""
    function_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, execution_time: float):
        """Update metrics with new execution"""
        self.total_calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.total_calls
        self.last_call_time = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "function_name": self.function_name,
            "total_calls": self.total_calls,
            "total_time_ms": round(self.total_time * 1000, 2),
            "min_time_ms": round(self.min_time * 1000, 2) if self.min_time != float('inf') else 0,
            "max_time_ms": round(self.max_time * 1000, 2),
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "cache_hit_rate": round(self.cache_hits / max(self.total_calls, 1), 4),
            "last_call": self.last_call_time.isoformat() if self.last_call_time else None
        }


class LRUCache:
    """Least Recently Used cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            # Check if key exists and not expired
            if key in self.cache:
                if key in self.expiry and time.time() > self.expiry[key]:
                    # Expired, remove it
                    del self.cache[key]
                    del self.expiry[key]
                    return None

                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]

            return None

    async def contains(self, key: str) -> bool:
        """Check if key exists in cache (and not expired)"""
        async with self._lock:
            if key in self.cache:
                if key in self.expiry and time.time() > self.expiry[key]:
                    # Expired, remove it
                    del self.cache[key]
                    del self.expiry[key]
                    return False
                return True
            return False
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        async with self._lock:
            # Remove oldest if at capacity
            if key not in self.cache and len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.expiry:
                    del self.expiry[oldest_key]
            
            self.cache[key] = value
            self.cache.move_to_end(key)
            
            # Set expiry
            ttl = ttl or self.default_ttl
            self.expiry[key] = time.time() + ttl
    
    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self.cache.clear()
            self.expiry.clear()
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        async with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": round(len(self.cache) / self.max_size, 4)
            }


# Global caches and metrics
_performance_metrics: Dict[str, PerformanceMetrics] = {}
_function_cache = LRUCache(max_size=1000, default_ttl=300)


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments"""
    # Create a deterministic string from args and kwargs
    key_parts = [func_name]
    
    # Add args
    for arg in args:
        try:
            key_parts.append(str(arg))
        except:
            key_parts.append(str(type(arg)))
    
    # Add kwargs (sorted for consistency)
    for k in sorted(kwargs.keys()):
        try:
            key_parts.append(f"{k}={kwargs[k]}")
        except:
            key_parts.append(f"{k}={type(kwargs[k])}")
    
    # Hash the key
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def profile(func: Callable):
    """
    Decorator to profile function performance.
    
    Usage:
        @profile
        async def my_function():
            # Your code here
            pass
    """
    func_name = func.__name__
    
    if func_name not in _performance_metrics:
        _performance_metrics[func_name] = PerformanceMetrics(function_name=func_name)
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            _performance_metrics[func_name].update(execution_time)
            
            # Log slow functions
            if execution_time > 1.0:
                logger.warning(
                    f"Slow function detected: {func_name} took {execution_time:.2f}s"
                )
    
    return wrapper


def cached(ttl: int = 300, cache_none: bool = False):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        cache_none: Whether to cache None results
        
    Usage:
        @cached(ttl=600)
        async def expensive_function(arg1, arg2):
            # Your code here
            pass
    """
    def decorator(func: Callable):
        func_name = func.__name__
        
        if func_name not in _performance_metrics:
            _performance_metrics[func_name] = PerformanceMetrics(function_name=func_name)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(func_name, args, kwargs)

            # Check if key exists in cache
            if await _function_cache.contains(cache_key):
                cached_result = await _function_cache.get(cache_key)
                _performance_metrics[func_name].cache_hits += 1
                logger.debug(f"Cache hit for {func_name}")
                return cached_result

            # Cache miss, execute function
            _performance_metrics[func_name].cache_misses += 1
            result = await func(*args, **kwargs)

            # Cache result if not None or if cache_none is True
            if result is not None or cache_none:
                await _function_cache.set(cache_key, result, ttl)

            return result
        
        return wrapper
    return decorator


async def get_performance_metrics() -> Dict[str, dict]:
    """Get all performance metrics"""
    return {
        name: metrics.to_dict()
        for name, metrics in _performance_metrics.items()
    }


async def get_cache_stats() -> dict:
    """Get cache statistics"""
    return await _function_cache.get_stats()


async def clear_cache():
    """Clear all caches"""
    await _function_cache.clear()
    logger.info("Cleared all caches")


async def clear_metrics():
    """Clear all performance metrics"""
    _performance_metrics.clear()
    logger.info("Cleared all performance metrics")


class ResourcePool:
    """Generic resource pool for connection pooling"""
    
    def __init__(self, create_resource: Callable, max_size: int = 10):
        self.create_resource = create_resource
        self.max_size = max_size
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire a resource from the pool"""
        try:
            # Try to get from pool without waiting
            resource = self.pool.get_nowait()
            return resource
        except asyncio.QueueEmpty:
            # Pool is empty, create new resource if under limit
            async with self._lock:
                if self.size < self.max_size:
                    self.size += 1
                    resource = await self.create_resource()
                    return resource
            
            # Wait for a resource to become available
            resource = await self.pool.get()
            return resource
    
    async def release(self, resource: Any):
        """Release a resource back to the pool"""
        try:
            self.pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool is full, discard the resource
            async with self._lock:
                self.size -= 1
    
    async def close(self):
        """Close all resources in the pool"""
        while not self.pool.empty():
            try:
                resource = self.pool.get_nowait()
                # Close resource if it has a close method
                if hasattr(resource, 'close'):
                    await resource.close()
            except asyncio.QueueEmpty:
                break
        
        self.size = 0


def batch_processor(batch_size: int = 100, max_wait: float = 1.0):
    """
    Decorator to batch process items for efficiency.
    
    Usage:
        @batch_processor(batch_size=50, max_wait=0.5)
        async def process_items(items: List):
            # Process batch of items
            pass
    """
    def decorator(func: Callable):
        batch = []
        last_process_time = [time.time()]  # Use list to avoid nonlocal issues
        lock = asyncio.Lock()
        
        @functools.wraps(func)
        async def wrapper(item: Any):
            async with lock:
                batch.append(item)
                
                # Process if batch is full or max wait time exceeded
                should_process = (
                    len(batch) >= batch_size or
                    time.time() - last_process_time[0] >= max_wait
                )
                
                if should_process:
                    items_to_process = batch.copy()
                    batch.clear()
                    last_process_time[0] = time.time()
                    
                    # Process batch
                    await func(items_to_process)
        
        return wrapper
    return decorator
