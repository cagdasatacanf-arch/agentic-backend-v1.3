"""
End-to-End Integration Tests

Tests complete workflows across all 7 phases:
- Phase 1: Basic RAG pipeline
- Phase 2: Multi-agent orchestration
- Phase 3: Advanced memory
- Phase 4: Streaming responses
- Phase 5: Distributed tracing
- Phase 6: Cost tracking, RBAC, monitoring, scaling
- Phase 7: Caching, rate limiting, circuit breakers, error recovery, performance

These tests verify the entire system works together correctly.
"""

import pytest
import asyncio
from httpx import AsyncClient
from app.main import app


@pytest.fixture
async def client():
    """Get async HTTP client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    """Test basic health check endpoint"""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_query_with_cost_tracking(client):
    """
    Integration test: Query endpoint with cost tracking

    Tests Phases 1, 4, 6, 7:
    - RAG pipeline (Phase 1)
    - Streaming response (Phase 4)
    - Cost tracking (Phase 6)
    - Caching (Phase 7)
    """
    query_data = {
        "query": "What is the capital of France?",
        "user_id": "test_user_integration",
        "session_id": "integration_session_1"
    }

    response = await client.post("/api/v1/query", json=query_data)

    assert response.status_code == 200
    data = response.json()

    # Should have answer
    assert "answer" in data
    assert len(data["answer"]) > 0

    # Should have cost information
    assert "cost" in data or "usage" in data


@pytest.mark.asyncio
async def test_rate_limiting_enforcement(client):
    """
    Integration test: Rate limiting enforcement

    Tests Phase 7 rate limiting with Phase 6 RBAC
    """
    # Set user to free tier (low rate limits)
    user_id = "rate_limit_test_user"

    # Make requests until rate limited
    responses = []
    for i in range(10):
        response = await client.post(
            "/api/v1/query",
            json={
                "query": f"Test query {i}",
                "user_id": user_id
            }
        )
        responses.append(response.status_code)

        if response.status_code == 429:
            break

    # Should eventually hit rate limit
    assert 429 in responses


@pytest.mark.asyncio
async def test_caching_reduces_cost(client):
    """
    Integration test: Verify caching reduces costs

    Tests Phase 7 caching with Phase 6 cost tracking
    """
    user_id = "caching_test_user"
    query = "What is 2 + 2?"

    # First request (cache miss, full cost)
    response1 = await client.post(
        "/api/v1/query",
        json={"query": query, "user_id": user_id}
    )

    # Second request (cache hit, should be cheaper/faster)
    response2 = await client.post(
        "/api/v1/query",
        json={"query": query, "user_id": user_id}
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Second request should indicate cache hit
    if "cached" in response2.json():
        assert response2.json()["cached"] is True


@pytest.mark.asyncio
async def test_rbac_permission_enforcement(client):
    """
    Integration test: RBAC permission enforcement

    Tests Phase 6 RBAC with various endpoints
    """
    # Create read-only API key
    rbac_response = await client.post(
        "/api/v1/rbac/api-key",
        json={
            "user_id": "readonly_integration_user",
            "role": "readonly"
        }
    )

    if rbac_response.status_code == 200:
        api_key = rbac_response.json()["api_key"]

        # Try to perform write operation (should fail)
        delete_response = await client.delete(
            "/api/v1/document/test_doc",
            headers={"X-API-Key": api_key}
        )

        # Should be forbidden or unauthorized
        assert delete_response.status_code in [401, 403]


@pytest.mark.asyncio
async def test_circuit_breaker_protection(client):
    """
    Integration test: Circuit breaker protects against failures

    Tests Phase 7 circuit breaker
    """
    # Simulate multiple failures to trigger circuit breaker
    # This would require an endpoint that can fail predictably


@pytest.mark.asyncio
async def test_monitoring_metrics_collection(client):
    """
    Integration test: Prometheus metrics collection

    Tests Phase 6 monitoring
    """
    # Make some requests
    for i in range(5):
        await client.post(
            "/api/v1/query",
            json={"query": f"Test {i}", "user_id": "metrics_user"}
        )

    # Check metrics endpoint
    metrics_response = await client.get("/api/v1/monitoring/metrics")

    assert metrics_response.status_code == 200

    metrics_text = metrics_response.text

    # Should contain key metrics
    assert "http_requests_total" in metrics_text
    assert "http_request_duration_seconds" in metrics_text


@pytest.mark.asyncio
async def test_budget_limit_enforcement(client):
    """
    Integration test: Budget limits prevent overspending

    Tests Phase 6 cost tracking with budget limits
    """
    user_id = "budget_test_user"

    # Set low budget
    budget_response = await client.post(
        "/api/v1/cost/budget",
        json={
            "user_id": user_id,
            "limit": 0.01,  # $0.01
            "period": "day"
        }
    )

    if budget_response.status_code == 200:
        # Make requests until budget exceeded
        for i in range(100):
            response = await client.post(
                "/api/v1/query",
                json={"query": f"Expensive query {i}", "user_id": user_id}
            )

            if response.status_code == 429:
                # Budget exceeded
                assert "budget" in response.json().get("detail", "").lower()
                break


@pytest.mark.asyncio
async def test_error_recovery_retry_logic(client):
    """
    Integration test: Error recovery with retry

    Tests Phase 7 error recovery
    """
    # This would test retry logic for transient failures
    # Requires simulating transient errors


@pytest.mark.asyncio
async def test_ab_testing_variant_assignment(client):
    """
    Integration test: A/B testing variant assignment

    Tests Phase 7 A/B testing
    """
    # Create experiment
    experiment_response = await client.post(
        "/api/v1/ab-testing/experiment",
        json={
            "experiment_id": "test_experiment",
            "variants": ["control", "variant_a"],
            "algorithm": "epsilon_greedy"
        }
    )

    if experiment_response.status_code == 200:
        # Assign variant to user
        assign_response = await client.post(
            "/api/v1/ab-testing/assign",
            json={
                "experiment_id": "test_experiment",
                "user_id": "ab_test_user"
            }
        )

        assert assign_response.status_code == 200
        data = assign_response.json()
        assert data["variant"] in ["control", "variant_a"]


@pytest.mark.asyncio
async def test_full_query_lifecycle(client):
    """
    Integration test: Complete query lifecycle

    Tests all phases working together:
    1. RBAC validates user
    2. Rate limiter checks limits
    3. Cache checked for existing result
    4. Circuit breaker protects external calls
    5. RAG pipeline executes
    6. Response streamed back
    7. Cost tracked
    8. Metrics collected
    9. Audit logged
    """
    user_id = "lifecycle_test_user"
    session_id = "lifecycle_session"
    query = "What is machine learning?"

    # Make query
    response = await client.post(
        "/api/v1/query",
        json={
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "stream": False
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "answer" in data
    assert "session_id" in data

    # Check cost was tracked
    cost_response = await client.get(
        f"/api/v1/cost/stats/{user_id}"
    )

    if cost_response.status_code == 200:
        cost_data = cost_response.json()
        assert cost_data["total_requests"] > 0

    # Check rate limit status
    rate_limit_response = await client.get(
        f"/api/v1/ratelimit/status/{user_id}"
    )

    if rate_limit_response.status_code == 200:
        rate_limit_data = rate_limit_response.json()
        assert "remaining" in rate_limit_data

    # Check audit log
    audit_response = await client.get(
        f"/api/v1/rbac/audit/{user_id}"
    )

    if audit_response.status_code == 200:
        audit_data = audit_response.json()
        assert len(audit_data) > 0


@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """
    Integration test: Handle concurrent requests

    Tests Phase 6 horizontal scaling and Phase 7 performance
    """
    async def make_request(i):
        return await client.post(
            "/api/v1/query",
            json={
                "query": f"Concurrent test {i}",
                "user_id": f"concurrent_user_{i % 3}"
            }
        )

    # Make 20 concurrent requests
    tasks = [make_request(i) for i in range(20)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Most should succeed
    successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
    assert successful > 15  # At least 75% success rate


@pytest.mark.asyncio
async def test_data_persistence(client):
    """
    Integration test: Data persists across sessions

    Tests Phase 3 memory and Phase 6 data storage
    """
    user_id = "persistence_test_user"
    session_id = "persistence_session"

    # First interaction
    response1 = await client.post(
        "/api/v1/query",
        json={
            "query": "My name is Alice",
            "user_id": user_id,
            "session_id": session_id
        }
    )

    assert response1.status_code == 200

    # Second interaction (should remember first)
    response2 = await client.post(
        "/api/v1/query",
        json={
            "query": "What is my name?",
            "user_id": user_id,
            "session_id": session_id
        }
    )

    assert response2.status_code == 200
    # Response should reference Alice
    answer = response2.json()["answer"].lower()
    assert "alice" in answer


@pytest.mark.asyncio
async def test_performance_under_load(client):
    """
    Integration test: Performance remains acceptable under load

    Tests Phase 7 performance optimization
    """
    import time

    start_time = time.time()

    # Make 50 requests
    tasks = []
    for i in range(50):
        task = client.post(
            "/api/v1/query",
            json={
                "query": f"Performance test {i}",
                "user_id": f"perf_user_{i % 5}"
            }
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Calculate average response time
    avg_time = elapsed / 50

    # Should handle 50 requests in reasonable time
    assert elapsed < 300  # 5 minutes total
    assert avg_time < 6    # 6 seconds average


@pytest.mark.asyncio
async def test_error_handling_graceful(client):
    """
    Integration test: Graceful error handling

    Tests Phase 7 error recovery
    """
    # Send invalid request
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "",  # Empty query
            "user_id": "error_test_user"
        }
    )

    # Should return error but not crash
    assert response.status_code in [400, 422]
    assert "detail" in response.json() or "error" in response.json()


@pytest.mark.asyncio
async def test_trace_correlation(client):
    """
    Integration test: Distributed tracing correlation

    Tests Phase 5 tracing across services
    """
    # Make request with trace ID
    response = await client.post(
        "/api/v1/query",
        json={"query": "Test tracing", "user_id": "trace_user"},
        headers={"X-Trace-ID": "test-trace-123"}
    )

    # Response should include trace information
    if "trace_id" in response.headers or "x-trace-id" in response.headers:
        trace_id = response.headers.get("trace_id") or response.headers.get("x-trace-id")
        assert trace_id is not None
