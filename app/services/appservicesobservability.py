"""
LangSmith observability utilities.
Comprehensive tracing, monitoring, and evaluation for LangGraph agents.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from functools import wraps
import json

from langsmith import Client
from langsmith.run_helpers import traceable, get_current_run_tree
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# LANGSMITH CLIENT SETUP
# ============================================================================

class LangSmithManager:
    """
    Centralized LangSmith management.
    Handles tracing, feedback, and monitoring.
    """
    
    def __init__(self):
        """Initialize LangSmith client if tracing is enabled."""
        self.enabled = settings.langchain_tracing_v2
        self.client = None
        
        if self.enabled:
            try:
                # Set environment variables for LangChain
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
                os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
                
                # Initialize client
                self.client = Client(
                    api_key=settings.langchain_api_key,
                    api_url=settings.langsmith_endpoint
                )
                
                logger.info(f"LangSmith tracing enabled for project: {settings.langchain_project}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith: {e}")
                self.enabled = False
                self.client = None
        else:
            logger.info("LangSmith tracing is disabled")
    
    def get_tracer(self) -> Optional[LangChainTracer]:
        """Get a LangChain tracer for callback management."""
        if not self.enabled:
            return None
        
        return LangChainTracer(
            project_name=settings.langchain_project,
            client=self.client
        )
    
    def create_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        comment: Optional[str] = None
    ) -> bool:
        """
        Create feedback for a run.
        
        Args:
            run_id: The run ID to attach feedback to
            key: Feedback key (e.g., "user_rating", "correctness")
            score: Numeric score (0.0 to 1.0)
            comment: Optional text comment
            
        Returns:
            True if feedback was created successfully
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
            logger.info(f"Feedback created for run {run_id}: {key}={score}")
            return True
        except Exception as e:
            logger.error(f"Failed to create feedback: {e}")
            return False
    
    def get_run_stats(
        self,
        run_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific run.
        
        Returns:
            Dictionary with tokens, cost, latency, etc.
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            run = self.client.read_run(run_id)
            
            # Extract key metrics
            stats = {
                "run_id": run_id,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "latency_ms": run.latency,
                "total_tokens": run.total_tokens,
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "total_cost": run.total_cost,
                "error": run.error
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get run stats: {e}")
            return None
    
    def list_recent_runs(
        self,
        limit: int = 20,
        filter_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent runs with optional filtering.
        
        Args:
            limit: Maximum number of runs to return
            filter_str: Optional filter string
            
        Returns:
            List of run summaries
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            runs = self.client.list_runs(
                project_name=settings.langchain_project,
                limit=limit,
                filter=filter_str
            )
            
            summaries = []
            for run in runs:
                summaries.append({
                    "run_id": str(run.id),
                    "name": run.name,
                    "status": run.status,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "latency_ms": run.latency,
                    "total_tokens": run.total_tokens,
                    "total_cost": run.total_cost
                })
            
            return summaries
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []


# Global manager instance
langsmith_manager = LangSmithManager()


# ============================================================================
# TRACING DECORATORS
# ============================================================================

def trace_agent_call(
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to trace agent calls with LangSmith.
    
    Usage:
        @trace_agent_call(name="rag_query", tags=["rag", "production"])
        def my_function(query: str):
            return process(query)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_manager.enabled:
                # If tracing disabled, just run the function
                return func(*args, **kwargs)
            
            # Prepare tracing metadata
            trace_name = name or func.__name__
            trace_metadata = metadata or {}
            trace_metadata.update({
                "function": func.__name__,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            trace_tags = tags or []
            trace_tags.append(settings.environment)
            
            # Use LangSmith traceable decorator
            traced_func = traceable(
                name=trace_name,
                metadata=trace_metadata,
                tags=trace_tags
            )(func)
            
            return traced_func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# COST TRACKING
# ============================================================================

class CostTracker:
    """
    Track LLM costs across requests.
    Uses LangSmith data to calculate costs.
    """
    
    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        "text-embedding-3-small": {
            "input": 0.02,
            "output": 0.00
        },
        "text-embedding-3-large": {
            "input": 0.13,
            "output": 0.00
        }
    }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate cost for a model call.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Find pricing for model
        pricing = None
        for model_key, prices in cls.PRICING.items():
            if model_key in model.lower():
                pricing = prices
                break
        
        if not pricing:
            logger.warning(f"Unknown model for pricing: {model}")
            return 0.0
        
        # Calculate cost
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    @classmethod
    def get_run_cost(cls, run_id: str) -> Optional[float]:
        """Get the cost for a specific run from LangSmith."""
        stats = langsmith_manager.get_run_stats(run_id)
        if not stats:
            return None
        
        return stats.get("total_cost")


# ============================================================================
# REQUEST CONTEXT TRACKING
# ============================================================================

class RequestContext:
    """
    Track request-level context for tracing.
    Use this to add metadata to all traces within a request.
    """
    
    def __init__(
        self,
        request_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.endpoint = endpoint
        self.start_time = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
    
    def add_metadata(self, key: str, value: Any):
        """Add custom metadata to the request context."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for tracing."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "start_time": self.start_time.isoformat(),
            "custom_metadata": self.metadata
        }


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

class EvaluationDataset:
    """
    Create and manage evaluation datasets in LangSmith.
    """
    
    @staticmethod
    def create_dataset(
        name: str,
        description: str,
        examples: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Create a new evaluation dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            examples: List of example dicts with "inputs" and "outputs"
            
        Returns:
            Dataset ID if successful
        """
        if not langsmith_manager.enabled or not langsmith_manager.client:
            logger.warning("Cannot create dataset - LangSmith not enabled")
            return None
        
        try:
            dataset = langsmith_manager.client.create_dataset(
                dataset_name=name,
                description=description
            )
            
            # Add examples
            for example in examples:
                langsmith_manager.client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example.get("outputs")
                )
            
            logger.info(f"Created dataset '{name}' with {len(examples)} examples")
            return str(dataset.id)
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
    
    @staticmethod
    def add_example_from_run(
        dataset_name: str,
        run_id: str
    ) -> bool:
        """
        Add an example to a dataset from a production run.
        Useful for building test cases from real usage.
        """
        if not langsmith_manager.enabled or not langsmith_manager.client:
            return False
        
        try:
            langsmith_manager.client.create_example_from_run(
                run_id=run_id,
                dataset_name=dataset_name
            )
            logger.info(f"Added example from run {run_id} to dataset {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add example: {e}")
            return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_current_run_id() -> Optional[str]:
    """Get the current LangSmith run ID if tracing is active."""
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            return str(run_tree.id)
    except Exception:
        pass
    return None


def log_agent_decision(
    decision: str,
    reasoning: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log an agent decision to current trace.
    
    Args:
        decision: The decision made (e.g., "use_tool", "end_conversation")
        reasoning: Why the decision was made
        metadata: Additional context
    """
    if not langsmith_manager.enabled:
        return
    
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "decision": decision,
                "reasoning": reasoning,
                **(metadata or {})
            })
    except Exception as e:
        logger.debug(f"Failed to log decision: {e}")
