"""
Integration tests for Monitoring API endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_circuit_breaker_status(test_client: AsyncClient):
    """Test getting circuit breaker status"""
    response = await test_client.get("/api/v1/monitoring/circuit-breakers")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "circuit_breakers" in data
    assert "total_breakers" in data
    assert "open_breakers" in data
    assert isinstance(data["circuit_breakers"], dict)


@pytest.mark.asyncio
async def test_reset_circuit_breakers(test_client: AsyncClient):
    """Test resetting circuit breakers"""
    response = await test_client.post("/api/v1/monitoring/circuit-breakers/reset")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "message" in data


@pytest.mark.asyncio
async def test_get_performance_stats(test_client: AsyncClient):
    """Test getting performance statistics"""
    response = await test_client.get("/api/v1/monitoring/performance")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "functions" in data
    assert "summary" in data
    assert isinstance(data["functions"], dict)
    assert "total_functions" in data["summary"]


@pytest.mark.asyncio
async def test_get_cache_stats(test_client: AsyncClient):
    """Test getting cache statistics"""
    response = await test_client.get("/api/v1/monitoring/cache/stats")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "cache" in data
    assert "status" in data
    assert "size" in data["cache"]
    assert "max_size" in data["cache"]
    assert "utilization" in data["cache"]


@pytest.mark.asyncio
async def test_clear_cache(test_client: AsyncClient):
    """Test clearing cache"""
    response = await test_client.post("/api/v1/monitoring/cache/clear")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "message" in data


@pytest.mark.asyncio
async def test_clear_metrics(test_client: AsyncClient):
    """Test clearing performance metrics"""
    response = await test_client.post("/api/v1/monitoring/metrics/clear")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True


@pytest.mark.asyncio
async def test_get_dead_letter_queue(test_client: AsyncClient):
    """Test getting dead letter queue"""
    response = await test_client.get("/api/v1/monitoring/dead-letter-queue")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "stats" in data
    assert "recent_failures" in data
    assert "total_in_queue" in data
    assert isinstance(data["recent_failures"], list)


@pytest.mark.asyncio
async def test_clear_dead_letter_queue(test_client: AsyncClient):
    """Test clearing dead letter queue"""
    response = await test_client.post("/api/v1/monitoring/dead-letter-queue/clear")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "message" in data


@pytest.mark.asyncio
async def test_detailed_health_check(test_client: AsyncClient):
    """Test detailed health check"""
    response = await test_client.get("/api/v1/monitoring/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "components" in data
    assert "circuit_breakers" in data["components"]
    assert "cache" in data["components"]
    assert "error_recovery" in data["components"]
    assert "performance" in data["components"]


@pytest.mark.asyncio
async def test_health_check_basic(test_client: AsyncClient):
    """Test basic health check endpoint"""
    response = await test_client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_monitoring_endpoints_sequence(test_client: AsyncClient):
    """Test sequence of monitoring operations"""
    # 1. Check initial state
    response = await test_client.get("/api/v1/monitoring/health")
    assert response.status_code == 200
    
    # 2. Clear everything
    await test_client.post("/api/v1/monitoring/cache/clear")
    await test_client.post("/api/v1/monitoring/metrics/clear")
    await test_client.post("/api/v1/monitoring/circuit-breakers/reset")
    
    # 3. Verify cleared state
    response = await test_client.get("/api/v1/monitoring/performance")
    data = response.json()
    assert len(data["functions"]) == 0
    
    # 4. Check cache is empty
    response = await test_client.get("/api/v1/monitoring/cache/stats")
    data = response.json()
    assert data["cache"]["size"] == 0
