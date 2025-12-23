"""
Unit tests for Circuit Breaker implementation.
"""

import pytest
import asyncio
from app.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenError,
    get_circuit_breaker
)


@pytest.mark.asyncio
async def test_circuit_breaker_closed_state():
    """Test circuit breaker starts in closed state"""
    config = CircuitBreakerConfig(name="test", failure_threshold=3)
    breaker = CircuitBreaker(config)
    
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """Test that circuit opens after threshold failures"""
    config = CircuitBreakerConfig(
        name="test",
        failure_threshold=3,
        timeout=1
    )
    breaker = CircuitBreaker(config)
    
    async def failing_function():
        raise Exception("Test failure")
    
    # Should fail 3 times before opening
    for i in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
        
        if i < 2:
            assert breaker.state == CircuitState.CLOSED
    
    assert breaker.state == CircuitState.OPEN
    assert breaker.failure_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open():
    """Test that circuit breaker rejects calls when open"""
    config = CircuitBreakerConfig(
        name="test",
        failure_threshold=2,
        timeout=10  # Long timeout
    )
    breaker = CircuitBreaker(config)
    
    async def failing_function():
        raise Exception("Fail")
    
    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
    
    assert breaker.state == CircuitState.OPEN
    
    # Should reject new calls
    async def success_function():
        return "success"
    
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(success_function)


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    """Test recovery through half-open state"""
    config = CircuitBreakerConfig(
        name="test",
        failure_threshold=2,
        success_threshold=2,
        timeout=0.1  # Short timeout for testing
    )
    breaker = CircuitBreaker(config)
    
    # Open the circuit
    async def failing_function():
        raise Exception("Fail")
    
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
    
    assert breaker.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(0.2)
    
    # Should transition to half-open
    async def success_function():
        return "success"
    
    # First success should transition to half-open
    result = await breaker.call(success_function)
    assert result == "success"
    assert breaker.state == CircuitState.HALF_OPEN
    
    # Second success should close circuit
    result = await breaker.call(success_function)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    """Test that failure in half-open state reopens circuit"""
    config = CircuitBreakerConfig(
        name="test",
        failure_threshold=2,
        success_threshold=2,
        timeout=0.1
    )
    breaker = CircuitBreaker(config)
    
    # Open the circuit
    async def failing_function():
        raise Exception("Fail")
    
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
    
    # Wait for timeout
    await asyncio.sleep(0.2)
    
    # Fail in half-open state
    with pytest.raises(Exception):
        await breaker.call(failing_function)
    
    # Should be open again
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_metrics():
    """Test circuit breaker metrics tracking"""
    config = CircuitBreakerConfig(name="test", failure_threshold=5)
    breaker = CircuitBreaker(config)
    
    async def success_function():
        return "success"
    
    async def failing_function():
        raise Exception("Fail")
    
    # Make some successful calls
    for _ in range(3):
        await breaker.call(success_function)
    
    # Make some failed calls
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
    
    metrics = breaker.get_metrics()
    
    assert metrics["total_calls"] == 5
    assert metrics["successful_calls"] == 3
    assert metrics["failed_calls"] == 2
    assert metrics["failure_rate"] == 0.4


@pytest.mark.asyncio
async def test_get_circuit_breaker_singleton():
    """Test that get_circuit_breaker returns same instance"""
    breaker1 = get_circuit_breaker("test_service")
    breaker2 = get_circuit_breaker("test_service")
    
    assert breaker1 is breaker2


@pytest.mark.asyncio
async def test_circuit_breaker_reset():
    """Test manual circuit breaker reset"""
    config = CircuitBreakerConfig(name="test", failure_threshold=2)
    breaker = CircuitBreaker(config)
    
    # Open the circuit
    async def failing_function():
        raise Exception("Fail")
    
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(failing_function)
    
    assert breaker.state == CircuitState.OPEN
    
    # Reset
    breaker.reset()
    
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
