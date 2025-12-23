"""
Tool Execution Metrics Tracking

Tracks tool execution success rates, latency, and error patterns for:
- Performance monitoring
- Quality optimization
- Future RL-based tool selection

Based on research from ToolLLM and Gorilla papers.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import redis
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Single tool execution record"""
    tool_name: str
    session_id: str
    timestamp: datetime
    success: bool
    latency_ms: float
    error_message: Optional[str] = None
    input_params: Optional[Dict] = None
    output: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ToolExecution':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ToolMetricsCollector:
    """
    Collects and analyzes tool execution metrics.

    Storage: Redis Sorted Sets (ZADD with timestamp as score)
    Key format: tool_metrics:{tool_name}

    Features:
    - Rolling window metrics (last N executions)
    - Success rate tracking
    - Latency percentiles (P50, P95, P99)
    - Error pattern analysis
    - Tool quality scoring (0.0-1.0)
    """

    def __init__(self, redis_client: redis.Redis, max_executions_per_tool: int = 1000):
        """
        Initialize metrics collector.

        Args:
            redis_client: Redis connection
            max_executions_per_tool: Max executions to keep per tool (rolling window)
        """
        self.redis = redis_client
        self.max_executions = max_executions_per_tool

    def log_execution(self, execution: ToolExecution) -> None:
        """
        Log a tool execution to Redis.

        Args:
            execution: ToolExecution record
        """
        try:
            key = f"tool_metrics:{execution.tool_name}"
            score = execution.timestamp.timestamp()
            value = execution.to_json()

            # Add to sorted set (score = timestamp)
            self.redis.zadd(key, {value: score})

            # Trim to keep only last N executions
            self.redis.zremrangebyrank(key, 0, -(self.max_executions + 1))

            # Also maintain a global tool list
            self.redis.sadd("tool_metrics:all_tools", execution.tool_name)

            logger.debug(
                f"Logged execution for {execution.tool_name}: "
                f"success={execution.success}, latency={execution.latency_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Failed to log tool execution: {e}", exc_info=True)

    def get_executions(
        self,
        tool_name: str,
        last_n: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[ToolExecution]:
        """
        Retrieve execution records for a tool.

        Args:
            tool_name: Tool name
            last_n: Get last N executions (default: all)
            since: Get executions since this datetime

        Returns:
            List of ToolExecution records
        """
        try:
            key = f"tool_metrics:{tool_name}"

            if since:
                # Get executions since timestamp
                min_score = since.timestamp()
                records = self.redis.zrangebyscore(key, min_score, '+inf')
            elif last_n:
                # Get last N executions
                records = self.redis.zrange(key, -last_n, -1)
            else:
                # Get all executions
                records = self.redis.zrange(key, 0, -1)

            return [ToolExecution.from_json(r.decode() if isinstance(r, bytes) else r) for r in records]

        except Exception as e:
            logger.error(f"Failed to get executions for {tool_name}: {e}")
            return []

    def get_success_rate(self, tool_name: str, last_n: int = 100) -> float:
        """
        Calculate rolling success rate.

        Args:
            tool_name: Tool name
            last_n: Consider last N executions

        Returns:
            Success rate (0.0-1.0), or 1.0 if no data
        """
        executions = self.get_executions(tool_name, last_n=last_n)

        if not executions:
            return 1.0  # Assume good until proven otherwise

        successes = sum(1 for e in executions if e.success)
        return successes / len(executions)

    def get_latency_stats(self, tool_name: str, last_n: int = 100) -> Dict[str, float]:
        """
        Calculate latency statistics.

        Args:
            tool_name: Tool name
            last_n: Consider last N executions

        Returns:
            Dict with p50, p95, p99, mean, min, max (all in ms)
        """
        executions = self.get_executions(tool_name, last_n=last_n)

        if not executions:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0
            }

        latencies = sorted([e.latency_ms for e in executions])
        n = len(latencies)

        def percentile(p: float) -> float:
            """Calculate percentile"""
            idx = int(n * p)
            return latencies[min(idx, n-1)]

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": sum(latencies) / n,
            "min": latencies[0],
            "max": latencies[-1]
        }

    def get_error_summary(self, tool_name: str, last_n: int = 100) -> Dict[str, int]:
        """
        Analyze error patterns.

        Args:
            tool_name: Tool name
            last_n: Consider last N executions

        Returns:
            Dict mapping error messages to occurrence counts
        """
        executions = self.get_executions(tool_name, last_n=last_n)

        error_counts = {}
        for e in executions:
            if not e.success and e.error_message:
                error_counts[e.error_message] = error_counts.get(e.error_message, 0) + 1

        # Sort by frequency
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))

    def get_tool_quality_score(
        self,
        tool_name: str,
        last_n: int = 100,
        success_weight: float = 0.7,
        latency_weight: float = 0.3
    ) -> float:
        """
        Calculate overall tool quality score (0.0-1.0).

        Formula:
            quality = (success_weight * success_rate) +
                     (latency_weight * latency_score)

        Where latency_score = 1.0 - min(1.0, p95_latency / 5000ms)
        (Assumes 5000ms is "bad", 0ms is "good")

        Args:
            tool_name: Tool name
            last_n: Consider last N executions
            success_weight: Weight for success rate (0.0-1.0)
            latency_weight: Weight for latency (0.0-1.0)

        Returns:
            Quality score (0.0-1.0)
        """
        success_rate = self.get_success_rate(tool_name, last_n=last_n)
        latency_stats = self.get_latency_stats(tool_name, last_n=last_n)

        # Latency score: 1.0 = instant, 0.0 = >5000ms
        p95 = latency_stats["p95"]
        latency_threshold = 5000.0  # ms
        latency_score = max(0.0, 1.0 - (p95 / latency_threshold))

        # Weighted combination
        quality = (success_weight * success_rate) + (latency_weight * latency_score)

        return quality

    def get_all_tools_summary(self, last_n: int = 100) -> List[Dict]:
        """
        Get quality summary for all tools.

        Args:
            last_n: Consider last N executions per tool

        Returns:
            List of dicts with tool name, quality score, success rate, latency
        """
        try:
            # Get all tracked tools
            tool_names = self.redis.smembers("tool_metrics:all_tools")
            tool_names = [t.decode() if isinstance(t, bytes) else t for t in tool_names]

            summaries = []
            for tool_name in tool_names:
                executions = self.get_executions(tool_name, last_n=last_n)
                if not executions:
                    continue

                success_rate = self.get_success_rate(tool_name, last_n=last_n)
                latency_stats = self.get_latency_stats(tool_name, last_n=last_n)
                quality_score = self.get_tool_quality_score(tool_name, last_n=last_n)

                summaries.append({
                    "tool_name": tool_name,
                    "quality_score": round(quality_score, 3),
                    "success_rate": round(success_rate, 3),
                    "execution_count": len(executions),
                    "latency_p50": round(latency_stats["p50"], 2),
                    "latency_p95": round(latency_stats["p95"], 2),
                    "latency_p99": round(latency_stats["p99"], 2),
                })

            # Sort by quality score descending
            summaries.sort(key=lambda x: x["quality_score"], reverse=True)
            return summaries

        except Exception as e:
            logger.error(f"Failed to get tools summary: {e}")
            return []

    def clear_tool_metrics(self, tool_name: str) -> bool:
        """
        Clear all metrics for a tool.

        Args:
            tool_name: Tool name

        Returns:
            True if cleared successfully
        """
        try:
            key = f"tool_metrics:{tool_name}"
            self.redis.delete(key)
            self.redis.srem("tool_metrics:all_tools", tool_name)
            logger.info(f"Cleared metrics for {tool_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear metrics for {tool_name}: {e}")
            return False


# Singleton instance (initialized in app startup)
_metrics_collector: Optional[ToolMetricsCollector] = None


def get_metrics_collector() -> Optional[ToolMetricsCollector]:
    """Get global metrics collector instance"""
    return _metrics_collector


def initialize_metrics_collector(redis_client: redis.Redis) -> ToolMetricsCollector:
    """Initialize global metrics collector"""
    global _metrics_collector
    _metrics_collector = ToolMetricsCollector(redis_client)
    logger.info("Tool metrics collector initialized")
    return _metrics_collector
