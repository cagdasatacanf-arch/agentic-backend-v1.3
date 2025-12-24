"""
Circuit Breaker Pattern Implementation for External API Calls

Provides fault tolerance and graceful degradation when external services fail.
Prevents cascading failures and allows automatic recovery.

Features:
- Automatic circuit breaking on repeated failures
- Configurable thresholds and timeouts
- Half-open state for recovery testing
- Metrics and monitoring integration
- Fallback strategies

Usage:
    from app.services.circuit_breaker import CircuitBreaker, get_circuit_breaker
    
    breaker = get_circuit_breaker("openai")
    
    @breaker.call
    async def call_openai_api():
        # Your API call here
        pass
"""

import asyncio
import time
import logging
from typing import Callable, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: int = 60  # Seconds to wait before trying half-open
    expected_exception: type = Exception  # Exception type to catch
    name: str = "default"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker"""
    name: str = "default"
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED
    failure_rate: float = 0.0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Too many failures, calls are rejected
    - HALF_OPEN: Testing recovery, limited calls allowed
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.metrics = CircuitBreakerMetrics(name=config.name)
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{config.name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._lock:
            self.metrics.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.metrics.rejected_calls += 1
                    logger.warning(
                        f"Circuit breaker '{self.config.name}' is OPEN, "
                        f"rejecting call"
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.config.name}' is open"
                    )
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            self._update_failure_rate()
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.last_failure_time = time.time()
            self.failure_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
            
            self._update_failure_rate()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        if self.state != CircuitState.OPEN:
            logger.error(
                f"Circuit breaker '{self.config.name}' transitioning to OPEN "
                f"after {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN
            self.metrics.current_state = CircuitState.OPEN
            self.metrics.state_changes += 1
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        logger.info(
            f"Circuit breaker '{self.config.name}' transitioning to HALF_OPEN "
            f"for recovery test"
        )
        self.state = CircuitState.HALF_OPEN
        self.metrics.current_state = CircuitState.HALF_OPEN
        self.metrics.state_changes += 1
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info(
            f"Circuit breaker '{self.config.name}' transitioning to CLOSED "
            f"after successful recovery"
        )
        self.state = CircuitState.CLOSED
        self.metrics.current_state = CircuitState.CLOSED
        self.metrics.state_changes += 1
        self.failure_count = 0
        self.success_count = 0
    
    def _update_failure_rate(self):
        """Update failure rate metric"""
        total = self.metrics.total_calls
        if total > 0:
            self.metrics.failure_rate = self.metrics.failed_calls / total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "rejected_calls": self.metrics.rejected_calls,
            "failure_rate": round(self.metrics.failure_rate, 4),
            "state_changes": self.metrics.state_changes,
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info(f"Manually resetting circuit breaker '{self.config.name}'")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: int = 60,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """
    Get or create a circuit breaker.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening
        success_threshold: Number of successes to close from half-open
        timeout: Seconds to wait before trying half-open
        expected_exception: Exception type to catch
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        config = CircuitBreakerConfig(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )
        _circuit_breakers[name] = CircuitBreaker(config)
    
    return _circuit_breakers[name]


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: int = 60,
    expected_exception: type = Exception
):
    """
    Decorator for circuit breaker protection.
    
    Usage:
        @circuit_breaker("openai", failure_threshold=3, timeout=30)
        async def call_openai():
            # Your code here
            pass
    """
    def decorator(func: Callable):
        breaker = get_circuit_breaker(
            name,
            failure_threshold,
            success_threshold,
            timeout,
            expected_exception
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all circuit breakers"""
    return {
        name: breaker.get_metrics()
        for name, breaker in _circuit_breakers.items()
    }


def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    for breaker in _circuit_breakers.values():
        breaker.reset()
