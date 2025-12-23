"""
Monitoring & Metrics API Routes

Endpoints for production monitoring and observability:
- Prometheus metrics endpoint
- Health checks with dependencies
- Metrics summary (JSON)
- Custom metric recording

Usage:
    GET  /api/v1/monitoring/metrics      # Prometheus metrics
    GET  /api/v1/monitoring/health       # Comprehensive health check
    GET  /api/v1/monitoring/summary      # Metrics summary (JSON)
    POST /api/v1/monitoring/record       # Record custom metric
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import time
from datetime import datetime

from app.services.prometheus_service import get_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# ============================================================================
# Request Models
# ============================================================================

class RecordMetricRequest(BaseModel):
    """Request to record a custom metric"""
    metric_type: str = Field(..., regex="^(counter|gauge|histogram)$", description="Metric type")
    name: str = Field(..., min_length=1, description="Metric name")
    value: float = Field(..., description="Metric value")
    labels: Optional[Dict[str, str]] = Field(None, description="Metric labels")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format.
    Configure Prometheus to scrape this endpoint.

    Example Prometheus config:
        scrape_configs:
          - job_name: 'agentic-backend'
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: '/api/v1/monitoring/metrics'
            scrape_interval: 15s

    Returns:
        Plain text in Prometheus format

    Example output:
        # HELP http_requests_total Total HTTP requests
        # TYPE http_requests_total counter
        http_requests_total{method="POST",endpoint="/api/v1/query",status="200"} 1234

        # HELP http_request_duration_seconds HTTP request latency in seconds
        # TYPE http_request_duration_seconds histogram
        http_request_duration_seconds_bucket{method="POST",endpoint="/api/v1/query",le="0.1"} 1000
        http_request_duration_seconds_sum{method="POST",endpoint="/api/v1/query"} 45.3
        http_request_duration_seconds_count{method="POST",endpoint="/api/v1/query"} 1234
    """
    try:
        metrics = get_metrics()
        prometheus_text = metrics.export_prometheus_format()

        return Response(
            content=prometheus_text,
            media_type="text/plain; version=0.0.4"
        )

    except Exception as e:
        logger.error(f"Failed to export metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def metrics_summary():
    """
    Get metrics summary in JSON format.

    Returns all metrics with calculated statistics.

    Example:
        GET /api/v1/monitoring/summary

        Response:
        {
            "counters": {
                "http_requests_total": {
                    "method=POST,endpoint=/api/v1/query,status=200": 1234,
                    "method=GET,endpoint=/api/v1/health,status=200": 5678
                }
            },
            "gauges": {
                "active_sessions": {
                    "": 42
                }
            },
            "histograms": {
                "http_request_duration_seconds": {
                    "method=POST,endpoint=/api/v1/query": {
                        "count": 1234,
                        "sum": 45.3,
                        "min": 0.012,
                        "max": 2.456,
                        "avg": 0.0367,
                        "p50": 0.025,
                        "p90": 0.089,
                        "p95": 0.156,
                        "p99": 0.523
                    }
                }
            }
        }
    """
    try:
        metrics = get_metrics()
        summary = metrics.get_metrics_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": summary
        }

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Comprehensive health check.

    Checks all system dependencies and returns detailed status.

    Returns:
        {
            "status": "healthy",
            "timestamp": "2024-01-20T14:30:00",
            "uptime_seconds": 86400,
            "dependencies": {
                "redis": "healthy",
                "qdrant": "healthy",
                "openai": "healthy"
            },
            "metrics": {
                "total_requests": 12345,
                "active_sessions": 42,
                "cache_hit_rate": 0.85
            }
        }
    """
    try:
        from app.config import settings
        import redis.asyncio as redis

        start_time = time.time()

        # Check dependencies
        dependencies = {}

        # Check Redis
        try:
            redis_client = await redis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/0",
                socket_connect_timeout=2
            )
            await redis_client.ping()
            await redis_client.close()
            dependencies["redis"] = "healthy"
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            dependencies["redis"] = f"unhealthy: {str(e)}"

        # Check Qdrant (vector store)
        try:
            from qdrant_client import QdrantClient
            qdrant = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=2
            )
            # Simple check - try to list collections
            collections = qdrant.get_collections()
            dependencies["qdrant"] = "healthy"
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            dependencies["qdrant"] = f"unhealthy: {str(e)}"

        # Check OpenAI API
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=5.0
            )
            # Just verify API key is set - don't make actual API call
            if settings.openai_api_key and len(settings.openai_api_key) > 0:
                dependencies["openai"] = "configured"
            else:
                dependencies["openai"] = "not configured"
        except Exception as e:
            logger.warning(f"OpenAI check failed: {e}")
            dependencies["openai"] = f"error: {str(e)}"

        # Get metrics summary
        metrics = get_metrics()
        summary = metrics.get_metrics_summary()

        # Calculate some stats
        total_requests = 0
        if "http_requests_total" in summary["counters"]:
            for count in summary["counters"]["http_requests_total"].values():
                total_requests += count

        active_sessions = 0
        if "active_sessions" in summary["gauges"]:
            for count in summary["gauges"]["active_sessions"].values():
                active_sessions += count

        # Determine overall status
        all_healthy = all(
            "healthy" in status or "configured" in status
            for status in dependencies.values()
        )
        overall_status = "healthy" if all_healthy else "degraded"

        health_check_duration = time.time() - start_time

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "health_check_duration_seconds": round(health_check_duration, 3),
            "dependencies": dependencies,
            "metrics": {
                "total_requests": int(total_requests),
                "active_sessions": int(active_sessions)
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.post("/record")
async def record_metric(request: RecordMetricRequest):
    """
    Record a custom metric.

    Allows external services or scripts to record metrics.

    Example:
        POST /api/v1/monitoring/record
        {
            "metric_type": "counter",
            "name": "custom_events_total",
            "value": 1,
            "labels": {
                "event_type": "user_signup",
                "source": "web"
            }
        }

        Response:
        {
            "status": "success",
            "metric": "custom_events_total",
            "type": "counter",
            "value": 1
        }
    """
    try:
        metrics = get_metrics()

        if request.metric_type == "counter":
            metrics.increment_counter(request.name, request.labels, request.value)
        elif request.metric_type == "gauge":
            metrics.set_gauge(request.name, request.value, request.labels)
        elif request.metric_type == "histogram":
            metrics.record_histogram(request.name, request.value, request.labels)
        else:
            raise ValueError(f"Unsupported metric type: {request.metric_type}")

        logger.info(f"Recorded metric: {request.name} ({request.metric_type}) = {request.value}")

        return {
            "status": "success",
            "metric": request.name,
            "type": request.metric_type,
            "value": request.value,
            "labels": request.labels
        }

    except Exception as e:
        logger.error(f"Failed to record metric: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.

    Returns 200 if service is ready to handle requests.

    Example:
        GET /api/v1/monitoring/ready

        Response (200 OK):
        {
            "ready": true
        }
    """
    # Could add more sophisticated checks here
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.

    Returns 200 if service is alive (even if degraded).

    Example:
        GET /api/v1/monitoring/live

        Response (200 OK):
        {
            "alive": true
        }
    """
    return {"alive": True}


@router.get("/info")
async def system_info():
    """
    Get system information.

    Returns version, configuration, and runtime info.

    Example:
        GET /api/v1/monitoring/info

        Response:
        {
            "version": "1.0.0",
            "environment": "production",
            "features": ["streaming", "caching", "monitoring"],
            "models": {
                "chat": "gpt-4o",
                "embedding": "text-embedding-3-small"
            }
        }
    """
    from app.config import settings

    return {
        "version": "1.0.0",
        "environment": "production",
        "features": [
            "multi-agent system",
            "streaming responses (SSE)",
            "intelligent caching",
            "cost tracking",
            "RBAC security",
            "prometheus monitoring",
            "vision & multimodal",
            "RAG with hybrid search"
        ],
        "models": {
            "chat": settings.openai_chat_model,
            "embedding": settings.openai_embedding_model
        },
        "agents": [
            "math",
            "code",
            "rag",
            "vision"
        ]
    }
