"""
Prometheus Metrics Exporter

Production-grade monitoring and metrics:
- Request metrics (latency, throughput, errors)
- Agent performance metrics
- Cost tracking metrics
- Cache performance metrics
- Security metrics
- Custom business metrics

Benefits:
- 📊 Real-time monitoring
- 🎯 Performance insights
- 🔔 Alerting capabilities
- 📈 Historical analysis
- 🔍 Debugging support

Usage:
    from app.services.prometheus_service import metrics

    # Increment counter
    metrics.increment_counter("query_total", {"agent": "math"})

    # Record timing
    with metrics.timer("query_duration", {"agent": "math"}):
        # ... query logic ...

    # Set gauge
    metrics.set_gauge("active_users", 42)
"""

import time
import logging
from typing import Dict, Optional, Any
from contextlib import contextmanager
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics exporter.

    Tracks and exposes metrics in Prometheus format.

    Metric types:
    - Counter: Monotonically increasing value (requests, errors)
    - Gauge: Can go up and down (active users, queue size)
    - Histogram: Distribution of values (latency, response size)
    - Summary: Similar to histogram with percentiles

    Example:
        metrics = PrometheusMetrics()

        # Counter
        metrics.increment_counter("http_requests_total", {"method": "POST", "status": "200"})

        # Gauge
        metrics.set_gauge("active_connections", 25)

        # Histogram
        metrics.record_histogram("request_duration_seconds", 0.123, {"endpoint": "/api/query"})

        # Timer context
        with metrics.timer("database_query_duration", {"table": "users"}):
            # ... database query ...
    """

    def __init__(self):
        """Initialize Prometheus metrics"""
        # Thread-safe storage
        self._lock = threading.Lock()

        # Metric storage
        self._counters = defaultdict(lambda: defaultdict(float))
        self._gauges = defaultdict(lambda: defaultdict(float))
        self._histograms = defaultdict(lambda: defaultdict(list))
        self._summaries = defaultdict(lambda: defaultdict(list))

        # Metric metadata
        self._counter_help = {}
        self._gauge_help = {}
        self._histogram_help = {}
        self._summary_help = {}

        # Initialize standard metrics
        self._init_standard_metrics()

        logger.info("PrometheusMetrics initialized")

    def _init_standard_metrics(self):
        """Initialize standard application metrics"""
        # HTTP metrics
        self.register_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        self.register_histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"]
        )

        # Agent metrics
        self.register_counter(
            "agent_queries_total",
            "Total agent queries",
            ["agent_type", "status"]
        )
        self.register_histogram(
            "agent_query_duration_seconds",
            "Agent query duration in seconds",
            ["agent_type"]
        )

        # Cost metrics
        self.register_counter(
            "api_cost_dollars_total",
            "Total API costs in dollars",
            ["model"]
        )
        self.register_counter(
            "api_tokens_total",
            "Total API tokens used",
            ["model", "type"]  # type: input/output
        )

        # Cache metrics
        self.register_counter(
            "cache_operations_total",
            "Total cache operations",
            ["cache_type", "operation", "result"]  # result: hit/miss
        )
        self.register_gauge(
            "cache_size_bytes",
            "Current cache size in bytes",
            ["cache_type"]
        )

        # Security metrics
        self.register_counter(
            "auth_attempts_total",
            "Total authentication attempts",
            ["result"]  # result: success/failure
        )
        self.register_counter(
            "permission_checks_total",
            "Total permission checks",
            ["result"]  # result: allowed/denied
        )

        # System metrics
        self.register_gauge(
            "active_sessions",
            "Number of active user sessions",
            []
        )
        self.register_gauge(
            "active_requests",
            "Number of active requests",
            []
        )

    # ============================================================================
    # Counter Metrics
    # ============================================================================

    def register_counter(self, name: str, help_text: str, labels: list):
        """Register a counter metric"""
        self._counter_help[name] = (help_text, labels)

    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """
        Increment a counter.

        Args:
            name: Metric name
            labels: Label dictionary
            value: Increment value (default 1.0)

        Example:
            metrics.increment_counter("http_requests_total", {
                "method": "POST",
                "endpoint": "/api/query",
                "status": "200"
            })
        """
        with self._lock:
            label_key = self._make_label_key(labels or {})
            self._counters[name][label_key] += value

    # ============================================================================
    # Gauge Metrics
    # ============================================================================

    def register_gauge(self, name: str, help_text: str, labels: list):
        """Register a gauge metric"""
        self._gauge_help[name] = (help_text, labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Label dictionary

        Example:
            metrics.set_gauge("active_connections", 42)
        """
        with self._lock:
            label_key = self._make_label_key(labels or {})
            self._gauges[name][label_key] = value

    def increment_gauge(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """Increment a gauge"""
        with self._lock:
            label_key = self._make_label_key(labels or {})
            self._gauges[name][label_key] = self._gauges[name].get(label_key, 0) + value

    def decrement_gauge(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """Decrement a gauge"""
        self.increment_gauge(name, labels, -value)

    # ============================================================================
    # Histogram Metrics
    # ============================================================================

    def register_histogram(self, name: str, help_text: str, labels: list):
        """Register a histogram metric"""
        self._histogram_help[name] = (help_text, labels)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a value in histogram.

        Args:
            name: Metric name
            value: Value to record
            labels: Label dictionary

        Example:
            metrics.record_histogram("request_duration_seconds", 0.123, {
                "endpoint": "/api/query"
            })
        """
        with self._lock:
            label_key = self._make_label_key(labels or {})
            self._histograms[name][label_key].append(value)

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Time a block of code.

        Args:
            name: Histogram metric name
            labels: Label dictionary

        Example:
            with metrics.timer("database_query_duration", {"table": "users"}):
                # ... code to time ...
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record_histogram(name, duration, labels)

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _make_label_key(self, labels: Dict[str, str]) -> str:
        """Create a consistent label key"""
        if not labels:
            return ""
        # Sort by key for consistency
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _calculate_histogram_stats(self, values: list) -> Dict[str, float]:
        """Calculate histogram statistics"""
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0}

        sorted_values = sorted(values)
        count = len(sorted_values)
        total = sum(sorted_values)

        # Calculate percentiles
        def percentile(p):
            k = (count - 1) * p
            f = int(k)
            c = f + 1
            if c >= count:
                return sorted_values[-1]
            return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

        return {
            "count": count,
            "sum": total,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": total / count,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    # ============================================================================
    # Export Metrics
    # ============================================================================

    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format

        Format:
            # HELP metric_name Help text
            # TYPE metric_name counter
            metric_name{label1="value1",label2="value2"} 42
        """
        lines = []

        with self._lock:
            # Export counters
            for name, labels_dict in self._counters.items():
                if name in self._counter_help:
                    help_text, _ = self._counter_help[name]
                    lines.append(f"# HELP {name} {help_text}")
                    lines.append(f"# TYPE {name} counter")

                for label_key, value in labels_dict.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export gauges
            for name, labels_dict in self._gauges.items():
                if name in self._gauge_help:
                    help_text, _ = self._gauge_help[name]
                    lines.append(f"# HELP {name} {help_text}")
                    lines.append(f"# TYPE {name} gauge")

                for label_key, value in labels_dict.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export histograms
            for name, labels_dict in self._histograms.items():
                if name in self._histogram_help:
                    help_text, _ = self._histogram_help[name]
                    lines.append(f"# HELP {name} {help_text}")
                    lines.append(f"# TYPE {name} histogram")

                for label_key, values in labels_dict.items():
                    stats = self._calculate_histogram_stats(values)

                    # Histogram buckets (simplified - using percentiles as buckets)
                    buckets = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, float("inf")]
                    cumulative_count = 0

                    for bucket in buckets:
                        count = sum(1 for v in values if v <= bucket)
                        bucket_labels = f"{label_key},le=\"{bucket}\"" if label_key else f"le=\"{bucket}\""
                        lines.append(f"{name}_bucket{{{bucket_labels}}} {count}")

                    # Summary stats
                    if label_key:
                        lines.append(f"{name}_sum{{{label_key}}} {stats['sum']}")
                        lines.append(f"{name}_count{{{label_key}}} {stats['count']}")
                    else:
                        lines.append(f"{name}_sum {stats['sum']}")
                        lines.append(f"{name}_count {stats['count']}")

        return "\n".join(lines) + "\n"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary (for JSON API).

        Returns:
            Dictionary with all metrics and statistics
        """
        summary = {
            "counters": {},
            "gauges": {},
            "histograms": {}
        }

        with self._lock:
            # Counters
            for name, labels_dict in self._counters.items():
                summary["counters"][name] = dict(labels_dict)

            # Gauges
            for name, labels_dict in self._gauges.items():
                summary["gauges"][name] = dict(labels_dict)

            # Histograms with stats
            for name, labels_dict in self._histograms.items():
                summary["histograms"][name] = {}
                for label_key, values in labels_dict.items():
                    summary["histograms"][name][label_key] = self._calculate_histogram_stats(values)

        return summary

    def reset(self):
        """Reset all metrics (for testing)"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()


# ============================================================================
# Singleton Instance
# ============================================================================

_metrics: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get or create global metrics instance"""
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
    return _metrics


# Export singleton instance
metrics = get_metrics()
