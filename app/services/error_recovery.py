"""
Error Recovery and Retry Strategies

Provides intelligent retry mechanisms with exponential backoff,
jitter, and fallback strategies for resilient API calls.

Features:
- Exponential backoff with jitter
- Configurable retry policies
- Fallback strategies
- Dead letter queue for failed requests
- Automatic error classification
- Recovery metrics

Usage:
    from app.services.error_recovery import retry_with_backoff, RetryPolicy
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    async def unreliable_api_call():
        # Your code here
        pass
"""

import asyncio
import random
import logging
from typing import Callable, Optional, Any, List, Type
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    TRANSIENT = "transient"  # Temporary, retry immediately
    RECOVERABLE = "recoverable"  # Can recover with backoff
    PERMANENT = "permanent"  # Don't retry
    CRITICAL = "critical"  # Requires immediate attention


@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [Exception]


@dataclass
class FailedRequest:
    """Record of a failed request for dead letter queue"""
    request_id: str
    function_name: str
    args: tuple
    kwargs: dict
    exception: Exception
    timestamp: datetime
    attempts: int
    severity: ErrorSeverity


class DeadLetterQueue:
    """Queue for permanently failed requests"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[FailedRequest] = []
        self._lock = asyncio.Lock()
    
    async def add(self, failed_request: FailedRequest):
        """Add failed request to queue"""
        async with self._lock:
            self.queue.append(failed_request)
            
            # Trim queue if too large
            if len(self.queue) > self.max_size:
                self.queue = self.queue[-self.max_size:]
            
            logger.error(
                f"Added to dead letter queue: {failed_request.function_name} "
                f"after {failed_request.attempts} attempts"
            )
    
    async def get_all(self) -> List[FailedRequest]:
        """Get all failed requests"""
        async with self._lock:
            return self.queue.copy()
    
    async def clear(self):
        """Clear the queue"""
        async with self._lock:
            count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {count} items from dead letter queue")
    
    async def get_stats(self) -> dict:
        """Get queue statistics"""
        async with self._lock:
            return {
                "total_failures": len(self.queue),
                "by_function": self._count_by_function(),
                "by_severity": self._count_by_severity(),
                "oldest_failure": self.queue[0].timestamp.isoformat() if self.queue else None,
                "newest_failure": self.queue[-1].timestamp.isoformat() if self.queue else None
            }
    
    def _count_by_function(self) -> dict:
        """Count failures by function name"""
        counts = {}
        for req in self.queue:
            counts[req.function_name] = counts.get(req.function_name, 0) + 1
        return counts
    
    def _count_by_severity(self) -> dict:
        """Count failures by severity"""
        counts = {}
        for req in self.queue:
            severity = req.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts


# Global dead letter queue
_dead_letter_queue = DeadLetterQueue()


def classify_error(exception: Exception) -> ErrorSeverity:
    """
    Classify error severity for retry decisions.
    
    Args:
        exception: The exception to classify
        
    Returns:
        ErrorSeverity level
    """
    error_msg = str(exception).lower()
    
    # Transient errors - retry immediately
    transient_indicators = [
        "timeout", "connection reset", "connection refused",
        "temporary", "try again"
    ]
    if any(indicator in error_msg for indicator in transient_indicators):
        return ErrorSeverity.TRANSIENT
    
    # Permanent errors - don't retry
    permanent_indicators = [
        "not found", "unauthorized", "forbidden",
        "invalid", "bad request", "not allowed"
    ]
    if any(indicator in error_msg for indicator in permanent_indicators):
        return ErrorSeverity.PERMANENT
    
    # Critical errors - requires attention
    critical_indicators = [
        "out of memory", "disk full", "quota exceeded",
        "rate limit", "service unavailable"
    ]
    if any(indicator in error_msg for indicator in critical_indicators):
        return ErrorSeverity.CRITICAL
    
    # Default to recoverable
    return ErrorSeverity.RECOVERABLE


def calculate_backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """
    Calculate delay for exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add jitter
        
    Returns:
        Delay in seconds
    """
    # Calculate exponential backoff
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    
    # Add jitter to prevent thundering herd
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


async def retry_with_backoff_async(
    func: Callable,
    policy: RetryPolicy,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with retry and exponential backoff.
    
    Args:
        func: Async function to execute
        policy: Retry policy configuration
        *args, **kwargs: Arguments for the function
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(policy.max_attempts):
        try:
            result = await func(*args, **kwargs)
            
            # Log recovery if this wasn't the first attempt
            if attempt > 0:
                logger.info(
                    f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                )
            
            return result
            
        except Exception as e:
            last_exception = e
            severity = classify_error(e)
            
            # Don't retry permanent errors
            if severity == ErrorSeverity.PERMANENT:
                logger.error(
                    f"Permanent error in {func.__name__}: {e}, not retrying"
                )
                raise
            
            # Check if this exception type is retryable
            if not any(isinstance(e, exc_type) for exc_type in policy.retryable_exceptions):
                logger.error(
                    f"Non-retryable exception in {func.__name__}: {type(e).__name__}"
                )
                raise
            
            # Last attempt - don't wait
            if attempt == policy.max_attempts - 1:
                logger.error(
                    f"Function {func.__name__} failed after {policy.max_attempts} attempts"
                )
                
                # Add to dead letter queue
                await _dead_letter_queue.add(FailedRequest(
                    request_id=f"{func.__name__}_{datetime.utcnow().timestamp()}",
                    function_name=func.__name__,
                    args=args,
                    kwargs=kwargs,
                    exception=e,
                    timestamp=datetime.utcnow(),
                    attempts=policy.max_attempts,
                    severity=severity
                ))
                
                raise
            
            # Calculate backoff delay
            delay = calculate_backoff_delay(
                attempt,
                policy.base_delay,
                policy.max_delay,
                policy.exponential_base,
                policy.jitter
            )
            
            logger.warning(
                f"Function {func.__name__} failed on attempt {attempt + 1}, "
                f"retrying in {delay:.2f}s. Error: {e}"
            )
            
            await asyncio.sleep(delay)
    
    # Should never reach here, but just in case
    raise last_exception


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for retry with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        async def unreliable_function():
            # Your code here
            pass
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff_async(func, policy, *args, **kwargs)
        
        return wrapper
    return decorator


async def with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Execute primary function with fallback on failure.
    
    Args:
        primary_func: Primary async function to try
        fallback_func: Fallback async function if primary fails
        *args, **kwargs: Arguments for both functions
        
    Returns:
        Result from primary or fallback function
    """
    try:
        return await primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(
            f"Primary function {primary_func.__name__} failed: {e}, "
            f"using fallback {fallback_func.__name__}"
        )
        return await fallback_func(*args, **kwargs)


def fallback(fallback_func: Callable):
    """
    Decorator for fallback strategy.
    
    Usage:
        async def fallback_handler(*args, **kwargs):
            return "fallback result"
        
        @fallback(fallback_handler)
        async def primary_function():
            # Your code here
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_fallback(func, fallback_func, *args, **kwargs)
        
        return wrapper
    return decorator


async def get_dead_letter_queue_stats() -> dict:
    """Get statistics from dead letter queue"""
    return await _dead_letter_queue.get_stats()


async def get_failed_requests() -> List[FailedRequest]:
    """Get all failed requests from dead letter queue"""
    return await _dead_letter_queue.get_all()


async def clear_dead_letter_queue():
    """Clear the dead letter queue"""
    await _dead_letter_queue.clear()
