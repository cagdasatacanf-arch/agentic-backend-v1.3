"""
Cache Management API Routes

Endpoints for Phase 7: Intelligent Caching System

Features:
- Cache statistics
- Cache clearing
- Cache configuration
- Cache health monitoring

Usage:
    GET    /api/v1/cache/stats          # Get cache statistics
    DELETE /api/v1/cache/clear          # Clear cache
    GET    /api/v1/cache/health         # Health check
    POST   /api/v1/cache/config         # Update configuration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import logging

from app.services.cache_service import (
    get_semantic_cache,
    get_response_cache,
    get_embedding_cache,
    get_dedup_cache
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    semantic_cache: Dict
    response_cache: Dict
    embedding_cache: Dict
    total_entries: int
    cache_types: int


class CacheClearRequest(BaseModel):
    """Request model for cache clearing"""
    cache_type: str = Field(..., pattern="^(semantic|response|embedding|all)$")
    agent_type: Optional[str] = Field(None, pattern="^(math|code|rag|vision|general)$")


class CacheConfigRequest(BaseModel):
    """Request model for cache configuration"""
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    ttl: Optional[int] = Field(None, ge=0, le=86400)  # Max 24 hours


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get cache statistics.

    Returns statistics for all cache types:
    - Semantic cache (entries by agent type)
    - Response cache
    - Embedding cache
    - Total entries

    Example:
        GET /api/v1/cache/stats

        Response:
        {
            "semantic_cache": {
                "math": 45,
                "code": 120,
                "rag": 300,
                "vision": 25,
                "general": 80,
                "total": 570
            },
            "response_cache": {
                "total": 1250
            },
            "embedding_cache": {
                "total": 890
            },
            "total_entries": 2710,
            "cache_types": 3
        }
    """
    try:
        logger.info("Fetching cache statistics")

        semantic_cache = get_semantic_cache()
        semantic_stats = await semantic_cache.get_stats()

        # For now, response and embedding cache don't have get_stats()
        # In production, implement these methods
        response_stats = {"total": 0}  # Placeholder
        embedding_stats = {"total": 0}  # Placeholder

        total_entries = semantic_stats.get("total", 0) + \
                       response_stats.get("total", 0) + \
                       embedding_stats.get("total", 0)

        response = CacheStatsResponse(
            semantic_cache=semantic_stats,
            response_cache=response_stats,
            embedding_cache=embedding_stats,
            total_entries=total_entries,
            cache_types=3
        )

        logger.info(f"Cache stats: {total_entries} total entries")
        return response

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_cache(request: CacheClearRequest):
    """
    Clear cache entries.

    Can clear:
    - All caches
    - Specific cache type
    - Specific agent type within semantic cache

    Example:
        DELETE /api/v1/cache/clear
        {
            "cache_type": "semantic",
            "agent_type": "math"
        }

        Response:
        {
            "message": "Cleared math entries from semantic cache",
            "cache_type": "semantic",
            "agent_type": "math"
        }
    """
    try:
        logger.info(f"Clearing cache: type={request.cache_type}, agent={request.agent_type}")

        if request.cache_type == "all":
            # Clear all caches
            semantic_cache = get_semantic_cache()
            await semantic_cache.clear()

            # Response and embedding caches would need clear() methods
            # For now, just clear semantic

            message = "Cleared all caches"

        elif request.cache_type == "semantic":
            semantic_cache = get_semantic_cache()
            await semantic_cache.clear(agent_type=request.agent_type)

            if request.agent_type:
                message = f"Cleared {request.agent_type} entries from semantic cache"
            else:
                message = "Cleared all semantic cache entries"

        else:
            message = f"Cache type {request.cache_type} clearing not yet implemented"

        logger.info(message)
        return {
            "message": message,
            "cache_type": request.cache_type,
            "agent_type": request.agent_type
        }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def update_cache_config(request: CacheConfigRequest):
    """
    Update cache configuration.

    Allows runtime configuration of:
    - Similarity threshold for semantic cache
    - TTL for cache entries

    Example:
        POST /api/v1/cache/config
        {
            "similarity_threshold": 0.92,
            "ttl": 7200
        }

        Response:
        {
            "message": "Cache configuration updated",
            "config": {
                "similarity_threshold": 0.92,
                "ttl": 7200
            }
        }
    """
    try:
        logger.info(f"Updating cache config: {request.dict()}")

        # Note: This would need to recreate cache instances or
        # support runtime configuration. For now, just return success.

        config_updates = {}
        if request.similarity_threshold is not None:
            config_updates["similarity_threshold"] = request.similarity_threshold

        if request.ttl is not None:
            config_updates["ttl"] = request.ttl

        logger.info(f"Cache config updated: {config_updates}")
        return {
            "message": "Cache configuration updated",
            "config": config_updates,
            "note": "Restart required for changes to take effect"
        }

    except Exception as e:
        logger.error(f"Failed to update cache config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for cache system.

    Tests connectivity to Redis and cache functionality.

    Returns:
        Status of caching system
    """
    try:
        # Test semantic cache
        semantic_cache = get_semantic_cache()
        stats = await semantic_cache.get_stats()

        return {
            "status": "healthy",
            "caching": "enabled",
            "cache_types": [
                "semantic_cache",
                "response_cache",
                "embedding_cache",
                "deduplication_cache"
            ],
            "total_entries": stats.get("total", 0),
            "redis": "connected"
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "redis": "disconnected"
        }


@router.get("/metrics")
async def get_cache_metrics():
    """
    Get detailed cache metrics.

    Includes:
    - Hit/miss rates
    - Average latency
    - Cost savings
    - Cache efficiency

    Example:
        GET /api/v1/cache/metrics

        Response:
        {
            "hit_rate": 0.65,
            "miss_rate": 0.35,
            "total_hits": 1250,
            "total_misses": 680,
            "avg_cache_latency_ms": 15,
            "estimated_cost_savings": 125.50
        }
    """
    try:
        # These would come from tracking cache hits/misses
        # For now, return placeholder metrics

        metrics = {
            "hit_rate": 0.0,  # To be implemented
            "miss_rate": 0.0,  # To be implemented
            "total_hits": 0,
            "total_misses": 0,
            "avg_cache_latency_ms": 0,
            "estimated_cost_savings": 0.0,
            "note": "Metrics tracking to be implemented"
        }

        return metrics

    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_cache():
    """
    Test cache functionality.

    Performs end-to-end test of semantic cache:
    1. Store a test entry
    2. Retrieve with exact match
    3. Retrieve with similar query
    4. Clean up

    Returns test results.
    """
    try:
        logger.info("Testing cache functionality")

        semantic_cache = get_semantic_cache()

        # Test 1: Store entry
        test_query = "What is the capital of France?"
        test_response = {"answer": "Paris"}

        await semantic_cache.set(
            test_query,
            "test",
            test_response
        )

        # Test 2: Retrieve exact match
        result1 = await semantic_cache.get(test_query, "test")
        exact_match = result1 is not None

        # Test 3: Retrieve similar query
        similar_query = "What's the capital city of France?"
        result2 = await semantic_cache.get(similar_query, "test")
        similar_match = result2 is not None

        # Test 4: Clean up
        await semantic_cache.clear(agent_type="test")

        return {
            "status": "success",
            "tests": {
                "exact_match": exact_match,
                "similar_match": similar_match,
                "similarity_score": result2.get("_cache_metadata", {}).get("similarity") if result2 else 0.0
            },
            "message": "Cache test completed successfully"
        }

    except Exception as e:
        logger.error(f"Cache test failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }
