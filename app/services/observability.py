"""
Observability service for monitoring and tracking.
"""
from typing import Dict, Any, Optional, List
import logging
import time
from functools import wraps
from contextvars import ContextVar
import uuid

logger = logging.getLogger("app")

# Context variable for tracking current run ID
_current_run_id: ContextVar[Optional[str]] = ContextVar("current_run_id", default=None)


class RequestContext:
    """Context for tracking request metadata."""
    
    def __init__(self, request_id: str, user_id: Optional[str] = None, endpoint: str = ""):
        self.request_id = request_id
        self.user_id = user_id
        self.endpoint = endpoint
        self.metadata = {}
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the request context."""
        self.metadata[key] = value


class CostTracker:
    """Track costs for LLM API calls."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def track_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4"):
        """Track token usage and calculate cost."""
        # Simplified cost calculation (update with actual pricing)
        cost_per_1k_prompt = 0.03 if "gpt-4" in model else 0.0015
        cost_per_1k_completion = 0.06 if "gpt-4" in model else 0.002
        
        cost = (prompt_tokens / 1000 * cost_per_1k_prompt + 
                completion_tokens / 1000 * cost_per_1k_completion)
        
        self.total_cost += cost
        self.total_tokens += prompt_tokens + completion_tokens
        
        return cost


class LangSmithManager:
    """Manager for LangSmith integration."""
    
    def __init__(self):
        from app.config import settings
        self.enabled = settings.langchain_tracing_v2
        self.project = settings.langchain_project
    
    def create_feedback(self, run_id: str, key: str, score: float, comment: Optional[str] = None) -> bool:
        """Create feedback for a run."""
        if not self.enabled:
            return False
        
        logger.info(f"Feedback: run_id={run_id}, key={key}, score={score}, comment={comment}")
        return True
    
    def get_run_stats(self, run_id: str) -> Optional[Dict]:
        """Get statistics for a run."""
        if not self.enabled:
            return None
        
        # Return mock stats
        return {
            "run_id": run_id,
            "status": "completed",
            "latency_ms": 1500.0,
            "total_tokens": 500,
            "prompt_tokens": 300,
            "completion_tokens": 200,
            "total_cost": 0.015
        }
    
    def list_recent_runs(self, limit: int = 20, filter_str: Optional[str] = None) -> List[Dict]:
        """List recent runs."""
        if not self.enabled:
            return []
        
        # Return empty list for now
        return []


class ObservabilityService:
    """Service for tracking metrics, traces, and logs."""
    
    def __init__(self):
        self.metrics = {}
    
    def track_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Track a metric value."""
        logger.info(f"Metric: {name}={value} tags={tags}")
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "tags": tags, "timestamp": time.time()})
    
    def track_event(self, event_name: str, properties: Optional[Dict[str, Any]] = None):
        """Track an event."""
        logger.info(f"Event: {event_name} properties={properties}")
    
    def start_trace(self, trace_name: str) -> str:
        """Start a trace and return trace ID."""
        trace_id = f"trace_{int(time.time() * 1000)}"
        logger.info(f"Starting trace: {trace_name} (ID: {trace_id})")
        return trace_id
    
    def end_trace(self, trace_id: str, success: bool = True, metadata: Optional[Dict] = None):
        """End a trace."""
        logger.info(f"Ending trace: {trace_id} success={success} metadata={metadata}")


# Global instances
observability_service = ObservabilityService()
langsmith_manager = LangSmithManager()


def get_current_run_id() -> Optional[str]:
    """Get the current run ID from context."""
    return _current_run_id.get()


def trace_agent_call(name: str, tags: Optional[List[str]] = None):
    """Decorator to trace agent calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            run_id = str(uuid.uuid4())
            _current_run_id.set(run_id)
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Trace: {name} completed in {duration:.2f}s (run_id={run_id}, tags={tags})")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Trace: {name} failed after {duration:.2f}s (run_id={run_id}, error={e})")
                raise
            finally:
                _current_run_id.set(None)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            run_id = str(uuid.uuid4())
            _current_run_id.set(run_id)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Trace: {name} completed in {duration:.2f}s (run_id={run_id}, tags={tags})")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Trace: {name} failed after {duration:.2f}s (run_id={run_id}, error={e})")
                raise
            finally:
                _current_run_id.set(None)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_performance(func):
    """Decorator to track function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            observability_service.track_metric(
                f"{func.__name__}_duration",
                duration,
                {"status": "success"}
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            observability_service.track_metric(
                f"{func.__name__}_duration",
                duration,
                {"status": "error"}
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            observability_service.track_metric(
                f"{func.__name__}_duration",
                duration,
                {"status": "success"}
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            observability_service.track_metric(
                f"{func.__name__}_duration",
                duration,
                {"status": "error"}
            )
            raise
    
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

