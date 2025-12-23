"""
Monitoring API Routes

Provides endpoints for monitoring circuit breakers, performance metrics,
error recovery, and system health.

Endpoints:
    GET  /api/v1/monitoring/circuit-breakers       - Circuit breaker metrics
    POST /api/v1/monitoring/circuit-breakers/reset - Reset all circuit breakers
    GET  /api/v1/monitoring/performance            - Performance metrics
    GET  /api/v1/monitoring/cache/stats            - Cache statistics
    POST /api/v1/monitoring/cache/clear            - Clear all caches
    GET  /api/v1/monitoring/dead-letter-queue      - Failed requests queue
    POST /api/v1/monitoring/dead-letter-queue/clear - Clear dead letter queue
    GET  /api/v1/monitoring/health                 - Detailed health check
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.services.circuit_breaker import (
    get_all_circuit_breaker_metrics,
    reset_all_circuit_breakers
)
from app.services.performance import (
    get_performance_metrics,
    get_cache_stats,
    clear_cache,
    clear_metrics
)
from app.services.error_recovery import (
    get_dead_letter_queue_stats,
    get_failed_requests,
    clear_dead_letter_queue
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/circuit-breakers")
async def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get status and metrics for all circuit breakers.
    
    Returns:
        Dictionary of circuit breaker names to their metrics
        
    Example Response:
        {
            "openai": {
                "state": "closed",
                "total_calls": 1000,
                "successful_calls": 995,
                "failed_calls": 5,
                "failure_rate": 0.005
            }
        }
    """
    try:
        metrics = get_all_circuit_breaker_metrics()
        return {
            "circuit_breakers": metrics,
            "total_breakers": len(metrics),
            "open_breakers": sum(1 for m in metrics.values() if m.get("state") == "open")
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breakers/reset")
async def reset_circuit_breakers():
    """
    Manually reset all circuit breakers to closed state.
    
    Use this to recover from a system-wide issue after fixing the root cause.
    
    Returns:
        Success message
    """
    try:
        reset_all_circuit_breakers()
        logger.info("All circuit breakers reset")
        return {
            "success": True,
            "message": "All circuit breakers have been reset to closed state"
        }
    except Exception as e:
        logger.error(f"Error resetting circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_stats() -> Dict[str, Any]:
    """
    Get performance metrics for all profiled functions.
    
    Returns:
        Dictionary of function names to their performance metrics
        
    Example Response:
        {
            "expensive_operation": {
                "total_calls": 1000,
                "avg_time_ms": 150.5,
                "min_time_ms": 50.2,
                "max_time_ms": 500.8,
                "cache_hit_rate": 0.85
            }
        }
    """
    try:
        metrics = await get_performance_metrics()
        
        # Calculate summary statistics
        total_calls = sum(m.get("total_calls", 0) for m in metrics.values())
        avg_response_time = sum(
            m.get("avg_time_ms", 0) * m.get("total_calls", 0) 
            for m in metrics.values()
        ) / max(total_calls, 1)
        
        return {
            "functions": metrics,
            "summary": {
                "total_functions": len(metrics),
                "total_calls": total_calls,
                "avg_response_time_ms": round(avg_response_time, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get cache statistics including size and utilization.
    
    Returns:
        Cache statistics
        
    Example Response:
        {
            "size": 750,
            "max_size": 1000,
            "utilization": 0.75
        }
    """
    try:
        stats = await get_cache_stats()
        return {
            "cache": stats,
            "status": "healthy" if stats.get("utilization", 0) < 0.9 else "near_capacity"
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_all_caches():
    """
    Clear all caches and optionally performance metrics.
    
    Query Parameters:
        clear_metrics: Whether to also clear performance metrics (default: false)
    
    Returns:
        Success message
    """
    try:
        await clear_cache()
        logger.info("All caches cleared")
        
        return {
            "success": True,
            "message": "All caches have been cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/clear")
async def clear_performance_metrics():
    """
    Clear all performance metrics.
    
    Returns:
        Success message
    """
    try:
        await clear_metrics()
        logger.info("Performance metrics cleared")
        
        return {
            "success": True,
            "message": "Performance metrics have been cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dead-letter-queue")
async def get_dead_letter_queue() -> Dict[str, Any]:
    """
    Get statistics and contents of the dead letter queue.
    
    Returns:
        Dead letter queue statistics and failed requests
        
    Example Response:
        {
            "stats": {
                "total_failures": 10,
                "by_function": {"api_call": 7, "db_query": 3},
                "by_severity": {"recoverable": 8, "critical": 2}
            },
            "recent_failures": [...]
        }
    """
    try:
        stats = await get_dead_letter_queue_stats()
        failed_requests = await get_failed_requests()
        
        # Get only recent failures (last 50)
        recent = failed_requests[-50:] if len(failed_requests) > 50 else failed_requests
        
        return {
            "stats": stats,
            "recent_failures": [
                {
                    "function_name": req.function_name,
                    "timestamp": req.timestamp.isoformat(),
                    "attempts": req.attempts,
                    "severity": req.severity.value,
                    "error": str(req.exception)
                }
                for req in recent
            ],
            "total_in_queue": len(failed_requests)
        }
    except Exception as e:
        logger.error(f"Error getting dead letter queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dead-letter-queue/clear")
async def clear_dlq():
    """
    Clear the dead letter queue.
    
    Returns:
        Success message with count of cleared items
    """
    try:
        stats = await get_dead_letter_queue_stats()
        count = stats.get("total_failures", 0)
        
        await clear_dead_letter_queue()
        logger.info(f"Dead letter queue cleared ({count} items)")
        
        return {
            "success": True,
            "message": f"Dead letter queue cleared ({count} items removed)"
        }
    except Exception as e:
        logger.error(f"Error clearing dead letter queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check including all monitoring systems.
    
    Returns:
        Comprehensive health status
    """
    try:
        # Get all metrics
        circuit_breakers = get_all_circuit_breaker_metrics()
        performance = await get_performance_metrics()
        cache = await get_cache_stats()
        dlq_stats = await get_dead_letter_queue_stats()
        
        # Determine overall health
        open_circuits = sum(1 for m in circuit_breakers.values() if m.get("state") == "open")
        cache_utilization = cache.get("utilization", 0)
        dlq_size = dlq_stats.get("total_failures", 0)
        
        # Health status
        is_healthy = (
            open_circuits == 0 and
            cache_utilization < 0.9 and
            dlq_size < 100
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": "2025-12-23T23:33:00Z",
            "components": {
                "circuit_breakers": {
                    "status": "healthy" if open_circuits == 0 else "degraded",
                    "total": len(circuit_breakers),
                    "open": open_circuits
                },
                "cache": {
                    "status": "healthy" if cache_utilization < 0.9 else "near_capacity",
                    "utilization": cache_utilization
                },
                "error_recovery": {
                    "status": "healthy" if dlq_size < 100 else "degraded",
                    "failed_requests": dlq_size
                },
                "performance": {
                    "status": "healthy",
                    "tracked_functions": len(performance)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
