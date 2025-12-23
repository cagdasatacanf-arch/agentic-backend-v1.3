"""
Unit tests for Error Recovery implementation.
"""

import pytest
import asyncio
from app.services.error_recovery import (
    retry_with_backoff,
    classify_error,
    ErrorSeverity,
    calculate_backoff_delay,
    with_fallback,
    get_dead_letter_queue_stats,
    clear_dead_letter_queue
)


@pytest.mark.asyncio
async def test_retry_with_backoff_success():
    """Test successful retry after failures"""
    attempt_count = 0
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = await flaky_function()
    assert result == "success"
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_all_failures():
    """Test that all retries fail"""
    attempt_count = 0
    
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    async def always_failing():
        nonlocal attempt_count
        attempt_count += 1
        raise Exception("Always fails")
    
    with pytest.raises(Exception, match="Always fails"):
        await always_failing()
    
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_error_classification_transient():
    """Test transient error classification"""
    errors = [
        Exception("Connection timeout"),
        Exception("Connection reset by peer"),
        Exception("Temporary failure, try again")
    ]
    
    for error in errors:
        assert classify_error(error) == ErrorSeverity.TRANSIENT


@pytest.mark.asyncio
async def test_error_classification_permanent():
    """Test permanent error classification"""
    errors = [
        Exception("Not found"),
        Exception("Unauthorized access"),
        Exception("Invalid request"),
        Exception("Bad request")
    ]
    
    for error in errors:
        assert classify_error(error) == ErrorSeverity.PERMANENT


@pytest.mark.asyncio
async def test_error_classification_critical():
    """Test critical error classification"""
    errors = [
        Exception("Out of memory"),
        Exception("Rate limit exceeded"),
        Exception("Service unavailable")
    ]
    
    for error in errors:
        assert classify_error(error) == ErrorSeverity.CRITICAL


@pytest.mark.asyncio
async def test_error_classification_recoverable():
    """Test recoverable error classification (default)"""
    error = Exception("Unknown error")
    assert classify_error(error) == ErrorSeverity.RECOVERABLE


def test_calculate_backoff_delay():
    """Test exponential backoff calculation"""
    # Without jitter
    delay = calculate_backoff_delay(
        attempt=0,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=False
    )
    assert delay == 1.0
    
    delay = calculate_backoff_delay(
        attempt=1,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=False
    )
    assert delay == 2.0
    
    delay = calculate_backoff_delay(
        attempt=2,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=False
    )
    assert delay == 4.0
    
    # Should cap at max_delay
    delay = calculate_backoff_delay(
        attempt=10,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=False
    )
    assert delay == 60.0


def test_calculate_backoff_delay_with_jitter():
    """Test that jitter adds randomness"""
    delays = []
    for _ in range(10):
        delay = calculate_backoff_delay(
            attempt=2,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True
        )
        delays.append(delay)
    
    # With jitter, delays should vary
    assert len(set(delays)) > 1
    # All delays should be between base and max
    assert all(2.0 <= d <= 4.0 for d in delays)


@pytest.mark.asyncio
async def test_with_fallback_success():
    """Test fallback when primary succeeds"""
    async def primary():
        return "primary"
    
    async def fallback():
        return "fallback"
    
    result = await with_fallback(primary, fallback)
    assert result == "primary"


@pytest.mark.asyncio
async def test_with_fallback_failure():
    """Test fallback when primary fails"""
    async def primary():
        raise Exception("Primary failed")
    
    async def fallback():
        return "fallback"
    
    result = await with_fallback(primary, fallback)
    assert result == "fallback"


@pytest.mark.asyncio
async def test_dead_letter_queue():
    """Test dead letter queue functionality"""
    # Clear queue first
    await clear_dead_letter_queue()
    
    # Create a function that always fails
    @retry_with_backoff(max_attempts=2, base_delay=0.01)
    async def failing_function():
        raise Exception("Test failure")
    
    # Should fail and add to DLQ
    with pytest.raises(Exception):
        await failing_function()
    
    # Check DLQ stats
    stats = await get_dead_letter_queue_stats()
    assert stats["total_failures"] >= 1


@pytest.mark.asyncio
async def test_retry_with_specific_exceptions():
    """Test retry only on specific exception types"""
    attempt_count = 0
    
    class RetryableError(Exception):
        pass
    
    class NonRetryableError(Exception):
        pass
    
    @retry_with_backoff(
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=[RetryableError]
    )
    async def selective_retry():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RetryableError("Should retry")
        elif attempt_count == 2:
            raise NonRetryableError("Should not retry")
        return "success"
    
    with pytest.raises(NonRetryableError):
        await selective_retry()
    
    # Should only attempt twice (first retry, then non-retryable)
    assert attempt_count == 2
