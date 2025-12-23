# Phase 7: Advanced Production Features & Optimization

**Status:** 🚧 In Progress
**Timeline:** 2-3 weeks
**Focus:** Production hardening, performance optimization, and advanced features

## Overview

Phase 7 takes the system to enterprise production-ready status with:
1. **Intelligent Caching** - Response caching, semantic deduplication, embedding cache
2. **Advanced Rate Limiting** - Per-user quotas, adaptive throttling, DDoS protection
3. **Error Recovery & Resilience** - Circuit breakers, automatic retry, fallback strategies
4. **Performance Optimization** - Query optimization, connection pooling, lazy loading
5. **Advanced Analytics** - User behavior tracking, A/B testing, conversion metrics

## Why Phase 7?

After implementing Phases 1-6, we have:
- ✅ Complete multi-agent system with vision capabilities
- ✅ Self-improvement pipeline with RL training
- ✅ Real-time streaming responses
- ✅ Vision and multimodal integration

**What's needed for enterprise production:**
- ❌ Response caching to reduce costs
- ❌ Rate limiting to prevent abuse
- ❌ Error recovery for resilience
- ❌ Performance optimization
- ❌ Advanced analytics and A/B testing
- ❌ Production-grade error handling

## Phase 7 Components

### 1. Intelligent Caching System

**Goal:** Reduce costs and latency through smart caching.

**Problem:**
- Identical queries processed multiple times
- Embeddings recalculated unnecessarily
- LLM calls for similar questions
- High API costs for repeated queries

**Solution:**

```python
# app/services/cache_service.py
from typing import Optional, Dict, Any
import hashlib
import json
from datetime import timedelta
import numpy as np

class SemanticCache:
    """
    Semantic caching using embedding similarity.

    Instead of exact match, finds similar queries and returns cached responses.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl: int = 3600  # 1 hour
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.embedding_service = get_embedding_service()

    async def get(
        self,
        query: str,
        agent_type: str
    ) -> Optional[Dict]:
        """
        Get cached response for semantically similar query.

        Steps:
        1. Embed the query
        2. Search for similar cached queries
        3. If similarity > threshold, return cached response
        """
        # Embed query
        query_embedding = await self.embedding_service.embed(query)

        # Search cache for similar queries
        cache_key_pattern = f"cache:{agent_type}:*"
        cached_queries = await self.redis.keys(cache_key_pattern)

        best_match = None
        best_similarity = 0.0

        for cached_key in cached_queries:
            cached_data = await self.redis.get(cached_key)
            if not cached_data:
                continue

            cached = json.loads(cached_data)
            cached_embedding = np.array(cached["embedding"])

            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                cached_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached

        # Return if similarity above threshold
        if best_similarity >= self.similarity_threshold:
            logger.info(f"Cache hit! Similarity: {best_similarity:.3f}")
            return best_match["response"]

        return None

    async def set(
        self,
        query: str,
        agent_type: str,
        response: Dict,
        embedding: Optional[np.ndarray] = None
    ):
        """Cache response with embedding for semantic search"""
        if embedding is None:
            embedding = await self.embedding_service.embed(query)

        cache_key = f"cache:{agent_type}:{self._hash(query)}"
        cache_data = {
            "query": query,
            "embedding": embedding.tolist(),
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        await self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(cache_data)
        )

class ResponseCache:
    """Simple response cache with TTL"""

    async def get_or_compute(
        self,
        key: str,
        compute_fn: callable,
        ttl: int = 3600
    ) -> Any:
        """
        Get from cache or compute and cache.

        Usage:
            result = await cache.get_or_compute(
                key="math:2+2",
                compute_fn=lambda: agent.solve("2+2"),
                ttl=3600
            )
        """
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            logger.info(f"Cache hit: {key}")
            return json.loads(cached)

        # Compute
        logger.info(f"Cache miss: {key}")
        result = await compute_fn()

        # Cache for next time
        await self.redis.setex(key, ttl, json.dumps(result))

        return result

class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""

    async def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """Get embedding from cache or compute"""
        cache_key = f"embedding:{model}:{self._hash(text)}"

        # Try cache
        cached = await self.redis.get(cache_key)
        if cached:
            return np.array(json.loads(cached))

        # Compute
        embedding = await self.embedding_service.embed(text, model)

        # Cache (no TTL - embeddings don't change)
        await self.redis.set(cache_key, json.dumps(embedding.tolist()))

        return embedding

class DeduplicationCache:
    """Prevent duplicate concurrent requests"""

    async def deduplicate(
        self,
        key: str,
        compute_fn: callable,
        timeout: int = 30
    ) -> Any:
        """
        If request is in progress, wait for it.
        Otherwise, compute.

        Prevents multiple users asking same question simultaneously.
        """
        lock_key = f"lock:{key}"
        result_key = f"result:{key}"

        # Try to acquire lock
        acquired = await self.redis.set(
            lock_key,
            "locked",
            ex=timeout,
            nx=True  # Only if not exists
        )

        if acquired:
            # We got the lock - compute
            try:
                result = await compute_fn()
                await self.redis.setex(result_key, 60, json.dumps(result))
                return result
            finally:
                await self.redis.delete(lock_key)
        else:
            # Someone else is computing - wait for result
            for _ in range(timeout * 2):  # Poll every 0.5s
                result = await self.redis.get(result_key)
                if result:
                    return json.loads(result)
                await asyncio.sleep(0.5)

            # Timeout - compute anyway
            return await compute_fn()
```

**Cache Strategy:**
```python
# app/middleware/cache_middleware.py
@app.middleware("http")
async def cache_middleware(request: Request, call_next):
    """Automatic caching for GET requests"""

    # Only cache GET requests
    if request.method != "GET":
        return await call_next(request)

    # Generate cache key
    cache_key = f"http:{request.url.path}:{request.query_params}"

    # Try cache
    cache = get_response_cache()
    cached = await cache.get(cache_key)

    if cached:
        return JSONResponse(
            content=cached,
            headers={"X-Cache": "HIT"}
        )

    # Process request
    response = await call_next(request)

    # Cache successful responses
    if response.status_code == 200:
        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Cache
        await cache.set(cache_key, json.loads(body), ttl=300)

        # Return response
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers) | {"X-Cache": "MISS"}
        )

    return response
```

**Expected Impact:**
- 💰 50-70% cost reduction for repeated queries
- ⚡ 10-100x faster responses for cached queries
- 📊 Reduced load on external APIs
- 🎯 Better user experience

---

### 2. Advanced Rate Limiting

**Goal:** Protect system from abuse and ensure fair usage.

**Implementation:**

```python
# app/services/rate_limiter.py
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum

class RateLimitTier(str, Enum):
    FREE = "free"           # 100 req/hour
    BASIC = "basic"         # 1,000 req/hour
    PRO = "pro"             # 10,000 req/hour
    ENTERPRISE = "enterprise"  # Unlimited

class RateLimiter:
    """
    Advanced rate limiting with multiple strategies.
    """

    TIER_LIMITS = {
        RateLimitTier.FREE: {"requests": 100, "window": 3600},
        RateLimitTier.BASIC: {"requests": 1000, "window": 3600},
        RateLimitTier.PRO: {"requests": 10000, "window": 3600},
        RateLimitTier.ENTERPRISE: {"requests": 999999, "window": 3600}
    }

    async def check_rate_limit(
        self,
        user_id: str,
        tier: RateLimitTier = RateLimitTier.FREE
    ) -> Dict:
        """
        Check if user is within rate limit.

        Returns:
            {
                "allowed": bool,
                "current": int,
                "limit": int,
                "reset_at": datetime,
                "retry_after": int  # seconds
            }
        """
        limit_config = self.TIER_LIMITS[tier]
        window = limit_config["window"]
        max_requests = limit_config["requests"]

        # Sliding window counter
        key = f"ratelimit:{tier}:{user_id}"
        now = datetime.now().timestamp()
        window_start = now - window

        # Remove old entries
        await self.redis.zremrangebyscore(key, "-inf", window_start)

        # Count current requests
        current_count = await self.redis.zcard(key)

        if current_count >= max_requests:
            # Get oldest request timestamp
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_at = datetime.fromtimestamp(oldest[0][1] + window)
                retry_after = int((reset_at - datetime.now()).total_seconds())
            else:
                reset_at = datetime.now() + timedelta(seconds=window)
                retry_after = window

            return {
                "allowed": False,
                "current": current_count,
                "limit": max_requests,
                "reset_at": reset_at,
                "retry_after": retry_after
            }

        # Add current request
        await self.redis.zadd(key, {str(now): now})
        await self.redis.expire(key, window)

        return {
            "allowed": True,
            "current": current_count + 1,
            "limit": max_requests,
            "reset_at": datetime.now() + timedelta(seconds=window),
            "retry_after": 0
        }

    async def get_adaptive_limit(
        self,
        user_id: str
    ) -> int:
        """
        Adaptive rate limiting based on system load.

        Reduces limits during high load, increases during low load.
        """
        # Get system metrics
        system_load = await self.get_system_load()
        user_tier = await self.get_user_tier(user_id)

        base_limit = self.TIER_LIMITS[user_tier]["requests"]

        if system_load > 0.9:  # 90% capacity
            return int(base_limit * 0.5)  # Reduce by 50%
        elif system_load > 0.75:  # 75% capacity
            return int(base_limit * 0.75)  # Reduce by 25%
        else:
            return base_limit

class TokenBucketLimiter:
    """
    Token bucket algorithm for smooth rate limiting.

    Allows bursts but maintains average rate.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int  # max bucket size
    ):
        self.rate = rate
        self.capacity = capacity

    async def consume(
        self,
        user_id: str,
        tokens: int = 1
    ) -> bool:
        """
        Try to consume tokens.

        Returns True if allowed, False if rate limited.
        """
        key = f"bucket:{user_id}"
        now = datetime.now().timestamp()

        # Get current bucket state
        bucket_data = await self.redis.hgetall(key)

        if not bucket_data:
            # Initialize bucket
            current_tokens = self.capacity - tokens
            last_update = now
        else:
            last_tokens = float(bucket_data[b"tokens"])
            last_update = float(bucket_data[b"last_update"])

            # Calculate tokens added since last update
            time_passed = now - last_update
            tokens_added = time_passed * self.rate

            # Update bucket
            current_tokens = min(
                self.capacity,
                last_tokens + tokens_added
            )
            current_tokens -= tokens

        # Check if we have enough tokens
        if current_tokens < 0:
            return False

        # Update bucket state
        await self.redis.hset(key, mapping={
            "tokens": current_tokens,
            "last_update": now
        })
        await self.redis.expire(key, 3600)

        return True

# Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests"""

    # Extract user ID from auth
    user_id = request.state.user.id if hasattr(request.state, "user") else "anonymous"
    tier = request.state.user.tier if hasattr(request.state, "user") else RateLimitTier.FREE

    # Check rate limit
    limiter = get_rate_limiter()
    result = await limiter.check_rate_limit(user_id, tier)

    # Add rate limit headers
    headers = {
        "X-RateLimit-Limit": str(result["limit"]),
        "X-RateLimit-Remaining": str(result["limit"] - result["current"]),
        "X-RateLimit-Reset": result["reset_at"].isoformat()
    }

    if not result["allowed"]:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": result["retry_after"]
            },
            headers=headers | {"Retry-After": str(result["retry_after"])}
        )

    # Process request
    response = await call_next(request)

    # Add headers to response
    for key, value in headers.items():
        response.headers[key] = value

    return response
```

**Expected Impact:**
- 🛡️ Protection from abuse and DDoS
- 📊 Fair usage across users
- 💰 Cost control
- 🎯 Better system stability

---

### 3. Error Recovery & Resilience

**Goal:** Graceful degradation and automatic recovery from failures.

**Implementation:**

```python
# app/services/resilience.py
from typing import Callable, Optional, Any
import asyncio
from functools import wraps

class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        recovery_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == "OPEN":
            # Check if we should try recovery
            if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: HALF_OPEN")
            else:
                raise CircuitBreakerOpen("Service unavailable")

        try:
            result = await func(*args, **kwargs)

            # Success - reset if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info("Circuit breaker: CLOSED (recovered)")

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()

            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker: OPEN (threshold reached)")

            raise

class RetryStrategy:
    """
    Automatic retry with exponential backoff.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute with retry logic.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Calculate backoff delay
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )

                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} "
                        f"after {delay}s: {e}"
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries exceeded: {e}")

        raise last_exception

class FallbackStrategy:
    """
    Fallback to alternative implementations.
    """

    async def execute_with_fallback(
        self,
        primary: Callable,
        fallback: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Try primary, fall back to alternative on failure.

        Usage:
            result = await fallback.execute_with_fallback(
                primary=lambda: expensive_llm_call(),
                fallback=lambda: cached_response()
            )
        """
        try:
            return await primary(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary failed, using fallback: {e}")
            return await fallback(*args, **kwargs)

# Decorators
def with_retry(max_retries: int = 3):
    """Decorator for automatic retry"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry = RetryStrategy(max_retries=max_retries)
            return await retry.execute(func, *args, **kwargs)
        return wrapper
    return decorator

def with_circuit_breaker(service_name: str):
    """Decorator for circuit breaker"""
    breakers = {}  # Global registry

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if service_name not in breakers:
                breakers[service_name] = CircuitBreaker()

            breaker = breakers[service_name]
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

def with_timeout(seconds: float):
    """Decorator for timeout"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        return wrapper
    return decorator

# Usage example
class ResilientAgent:
    """Agent with built-in resilience"""

    @with_retry(max_retries=3)
    @with_circuit_breaker("openai_api")
    @with_timeout(30.0)
    async def query(self, query: str) -> Dict:
        """Query with automatic retry, circuit breaker, and timeout"""
        return await self.llm.ainvoke(query)
```

**Expected Impact:**
- 🛡️ Automatic recovery from transient failures
- 📈 99.9% uptime
- 🎯 Better user experience
- 💰 Reduced manual intervention

---

### 4. Performance Optimization

**Goal:** Maximize throughput and minimize latency.

**Strategies:**

```python
# app/services/performance.py

# 1. Connection Pooling
class ConnectionPool:
    """Reuse database/API connections"""

    def __init__(self, max_size: int = 10):
        self.pool = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self.max_size = max_size

    async def get_connection(self):
        """Get connection from pool"""
        if self.pool.empty() and self.size < self.max_size:
            # Create new connection
            conn = await self._create_connection()
            self.size += 1
            return conn

        # Wait for available connection
        return await self.pool.get()

    async def release(self, conn):
        """Return connection to pool"""
        await self.pool.put(conn)

# 2. Lazy Loading
class LazyLoader:
    """Load resources only when needed"""

    def __init__(self):
        self._cache = {}

    async def get(self, key: str, loader: Callable):
        """Get resource, loading if not cached"""
        if key not in self._cache:
            self._cache[key] = await loader()
        return self._cache[key]

# 3. Batch Processing
class BatchProcessor:
    """Batch multiple requests together"""

    def __init__(self, batch_size: int = 10, max_wait_ms: int = 100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.processing = False

    async def add(self, item):
        """Add item to batch"""
        self.queue.append(item)

        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        elif not self.processing:
            # Start timer for partial batch
            asyncio.create_task(self._process_after_delay())

    async def _process_batch(self):
        """Process current batch"""
        if not self.queue:
            return

        self.processing = True
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        # Process batch
        results = await self._batch_operation(batch)

        self.processing = False
        return results

# 4. Query Optimization
class QueryOptimizer:
    """Optimize database/vector queries"""

    async def optimize_rag_query(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:
        """
        Optimized RAG query:
        1. Use index hints
        2. Limit fields returned
        3. Use approximate search for large datasets
        """
        # Use approximate nearest neighbor for speed
        if top_k > 100:
            return await self._approximate_search(query, top_k)

        # Use exact search for accuracy
        return await self._exact_search(query, top_k)
```

**Expected Impact:**
- ⚡ 2-5x faster response times
- 📊 10x higher throughput
- 💰 Lower infrastructure costs
- 🎯 Better resource utilization

---

### 5. Advanced Analytics

**Goal:** Understand usage patterns and optimize user experience.

**Implementation:**

```python
# app/services/analytics.py

class AnalyticsService:
    """Track user behavior and system performance"""

    async def track_event(
        self,
        user_id: str,
        event_type: str,
        properties: Dict
    ):
        """Track analytics event"""
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "properties": properties,
            "timestamp": datetime.now().isoformat()
        }

        # Store in analytics DB (ClickHouse, BigQuery, etc.)
        await self.analytics_db.insert(event)

    async def get_user_journey(
        self,
        user_id: str
    ) -> List[Dict]:
        """Get user's interaction history"""
        return await self.analytics_db.query(
            f"SELECT * FROM events WHERE user_id = '{user_id}' ORDER BY timestamp"
        )

    async def get_conversion_funnel(self) -> Dict:
        """Analyze conversion funnel"""
        return {
            "visits": await self.count_events("page_view"),
            "queries": await self.count_events("query_submitted"),
            "satisfied": await self.count_events("positive_feedback"),
            "conversion_rate": ...
        }

class ABTestingService:
    """A/B testing for features"""

    async def assign_variant(
        self,
        user_id: str,
        experiment: str
    ) -> str:
        """Assign user to experiment variant"""
        # Consistent hashing for stable assignment
        hash_val = hash(f"{user_id}:{experiment}") % 100

        if hash_val < 50:
            return "control"
        else:
            return "treatment"

    async def track_experiment_result(
        self,
        experiment: str,
        variant: str,
        metric: str,
        value: float
    ):
        """Track experiment metric"""
        await self.analytics.track_event(
            user_id="system",
            event_type="experiment_metric",
            properties={
                "experiment": experiment,
                "variant": variant,
                "metric": metric,
                "value": value
            }
        )
```

**Expected Impact:**
- 📊 Data-driven decision making
- 🎯 Optimized user experience
- 📈 Continuous improvement
- 💡 Feature insights

---

## Implementation Plan

### Week 1: Caching & Rate Limiting
- **Days 1-2**: Semantic cache implementation
- **Days 3-4**: Advanced rate limiting
- **Day 5**: Testing and optimization

### Week 2: Resilience & Performance
- **Days 1-2**: Circuit breakers and retry logic
- **Days 3-4**: Performance optimization
- **Day 5**: Load testing

### Week 3: Analytics & Polish
- **Days 1-2**: Analytics service
- **Days 3-4**: A/B testing framework
- **Day 5**: Documentation and deployment

## Success Criteria

**Caching:**
- ✅ 50%+ cache hit rate
- ✅ 50%+ cost reduction
- ✅ Sub-100ms cache retrieval

**Rate Limiting:**
- ✅ Per-user quotas enforced
- ✅ Adaptive limiting during high load
- ✅ No DDoS vulnerability

**Resilience:**
- ✅ 99.9% uptime
- ✅ Automatic recovery from failures
- ✅ Graceful degradation

**Performance:**
- ✅ 2x faster average response time
- ✅ 10x higher throughput
- ✅ Optimized resource usage

**Analytics:**
- ✅ All events tracked
- ✅ User journeys visible
- ✅ A/B testing operational

## Technologies

**New Dependencies:**
```txt
# Caching
redis-py==5.0.1
hiredis==2.2.3

# Performance
aiocache==0.12.2
orjson==3.9.10

# Analytics
clickhouse-driver==0.2.6
```

## Next Steps

Ready to start Phase 7 implementation!

**Recommended order:**
1. **Caching** - Immediate cost savings
2. **Rate Limiting** - Protection and fairness
3. **Resilience** - Production stability
4. **Performance** - Better UX
5. **Analytics** - Continuous improvement

Which component should I start with?
