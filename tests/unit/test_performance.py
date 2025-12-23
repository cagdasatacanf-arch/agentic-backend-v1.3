"""
Unit tests for Performance Optimization implementation.
"""

import pytest
import asyncio
from app.services.performance import (
    profile,
    cached,
    LRUCache,
    get_performance_metrics,
    get_cache_stats,
    clear_cache,
    clear_metrics
)


@pytest.mark.asyncio
async def test_lru_cache_basic():
    """Test basic LRU cache operations"""
    cache = LRUCache(max_size=3, default_ttl=60)
    
    # Set and get
    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"
    
    # Cache miss
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_lru_cache_eviction():
    """Test LRU eviction when cache is full"""
    cache = LRUCache(max_size=3, default_ttl=60)
    
    # Fill cache
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")
    
    # Add one more (should evict key1)
    await cache.set("key4", "value4")
    
    assert await cache.get("key1") is None
    assert await cache.get("key4") == "value4"


@pytest.mark.asyncio
async def test_lru_cache_ttl():
    """Test cache TTL expiration"""
    cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL
    
    await cache.set("key1", "value1")
    assert await cache.get("key1") == "value1"
    
    # Wait for expiration
    await asyncio.sleep(0.2)
    
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_lru_cache_lru_order():
    """Test that recently used items are kept"""
    cache = LRUCache(max_size=3, default_ttl=60)
    
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")
    
    # Access key1 (makes it most recently used)
    await cache.get("key1")
    
    # Add key4 (should evict key2, not key1)
    await cache.set("key4", "value4")
    
    assert await cache.get("key1") == "value1"
    assert await cache.get("key2") is None
    assert await cache.get("key4") == "value4"


@pytest.mark.asyncio
async def test_cached_decorator():
    """Test caching decorator"""
    call_count = 0
    
    @cached(ttl=60)
    async def expensive_function(x):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # Simulate expensive operation
        return x * 2
    
    # First call
    result1 = await expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call (cached)
    result2 = await expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment
    
    # Different argument
    result3 = await expensive_function(10)
    assert result3 == 20
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_with_kwargs():
    """Test caching with keyword arguments"""
    call_count = 0
    
    @cached(ttl=60)
    async def function_with_kwargs(x, y=10):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # First call
    result1 = await function_with_kwargs(5, y=10)
    assert result1 == 15
    assert call_count == 1
    
    # Same call (cached)
    result2 = await function_with_kwargs(5, y=10)
    assert result2 == 15
    assert call_count == 1
    
    # Different kwargs
    result3 = await function_with_kwargs(5, y=20)
    assert result3 == 25
    assert call_count == 2


@pytest.mark.asyncio
async def test_profile_decorator():
    """Test profiling decorator"""
    @profile
    async def profiled_function():
        await asyncio.sleep(0.01)
        return "result"
    
    # Call function multiple times
    for _ in range(3):
        await profiled_function()
    
    # Check metrics
    metrics = await get_performance_metrics()
    assert "profiled_function" in metrics
    
    func_metrics = metrics["profiled_function"]
    assert func_metrics["total_calls"] == 3
    assert func_metrics["avg_time_ms"] > 0


@pytest.mark.asyncio
async def test_profile_with_cached():
    """Test combining profile and cached decorators"""
    call_count = 0
    
    @profile
    @cached(ttl=60)
    async def combined_function(x):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return x * 2
    
    # Call multiple times with same argument
    for _ in range(5):
        result = await combined_function(10)
        assert result == 20
    
    # Should only execute once (cached)
    assert call_count == 1
    
    # But should be profiled 5 times
    metrics = await get_performance_metrics()
    assert metrics["combined_function"]["total_calls"] == 5


@pytest.mark.asyncio
async def test_cache_stats():
    """Test cache statistics"""
    cache = LRUCache(max_size=100, default_ttl=60)
    
    # Add some items
    for i in range(50):
        await cache.set(f"key{i}", f"value{i}")
    
    stats = await cache.get_stats()
    
    assert stats["size"] == 50
    assert stats["max_size"] == 100
    assert stats["utilization"] == 0.5


@pytest.mark.asyncio
async def test_clear_cache():
    """Test clearing cache"""
    @cached(ttl=60)
    async def cached_func(x):
        return x * 2
    
    # Call to populate cache
    await cached_func(5)
    
    # Clear cache
    await clear_cache()
    
    # Cache should be empty
    stats = await get_cache_stats()
    assert stats["size"] == 0


@pytest.mark.asyncio
async def test_clear_metrics():
    """Test clearing performance metrics"""
    @profile
    async def profiled_func():
        return "result"
    
    # Call to generate metrics
    await profiled_func()
    
    # Clear metrics
    await clear_metrics()
    
    # Metrics should be empty
    metrics = await get_performance_metrics()
    assert len(metrics) == 0


@pytest.mark.asyncio
async def test_cache_none_values():
    """Test caching of None values"""
    call_count = 0
    
    @cached(ttl=60, cache_none=True)
    async def returns_none():
        nonlocal call_count
        call_count += 1
        return None
    
    # First call
    result1 = await returns_none()
    assert result1 is None
    assert call_count == 1
    
    # Second call (should be cached)
    result2 = await returns_none()
    assert result2 is None
    assert call_count == 1


@pytest.mark.asyncio
async def test_cache_without_none_values():
    """Test not caching None values"""
    call_count = 0
    
    @cached(ttl=60, cache_none=False)
    async def returns_none():
        nonlocal call_count
        call_count += 1
        return None
    
    # First call
    result1 = await returns_none()
    assert result1 is None
    assert call_count == 1
    
    # Second call (should NOT be cached)
    result2 = await returns_none()
    assert result2 is None
    assert call_count == 2
