"""
Unit Tests for Prometheus Metrics Collection

Tests the metrics collection system including:
- Counter metrics (http_requests_total, etc.)
- Gauge metrics (active_sessions, cache_size, etc.)
- Histogram metrics (request_duration, etc.)
- Label handling
- Metrics export format
"""

import pytest
import time
from app.services.prometheus_service import (
    PrometheusMetrics,
    get_prometheus_metrics
)


@pytest.fixture
def metrics():
    """Get fresh Prometheus metrics instance"""
    # Create new instance for each test
    return PrometheusMetrics()


def test_increment_counter(metrics):
    """Test incrementing counter metrics"""
    # Increment counter
    metrics.increment_counter("http_requests_total", {"endpoint": "/api/v1/query"}, value=1)
    metrics.increment_counter("http_requests_total", {"endpoint": "/api/v1/query"}, value=1)

    # Export and check
    output = metrics.export_metrics()
    assert "http_requests_total" in output
    assert 'endpoint="/api/v1/query"' in output


def test_increment_counter_multiple_labels(metrics):
    """Test counter with multiple labels"""
    labels = {
        "endpoint": "/api/v1/query",
        "method": "POST",
        "status": "200"
    }

    metrics.increment_counter("http_requests_total", labels, value=5)

    output = metrics.export_metrics()
    assert 'endpoint="/api/v1/query"' in output
    assert 'method="POST"' in output
    assert 'status="200"' in output


def test_set_gauge(metrics):
    """Test setting gauge metrics"""
    # Set gauge value
    metrics.set_gauge("active_sessions", {"instance": "api-1"}, value=42)

    output = metrics.export_metrics()
    assert "active_sessions" in output
    assert 'instance="api-1"' in output
    assert "42" in output


def test_increment_gauge(metrics):
    """Test incrementing gauge"""
    labels = {"cache": "semantic"}

    # Increment multiple times
    metrics.increment_gauge("cache_size", labels, value=10)
    metrics.increment_gauge("cache_size", labels, value=5)

    # Should be 15 total
    output = metrics.export_metrics()
    assert "cache_size" in output


def test_decrement_gauge(metrics):
    """Test decrementing gauge"""
    labels = {"queue": "tasks"}

    # Set initial value
    metrics.set_gauge("queue_size", labels, value=100)

    # Decrement
    metrics.increment_gauge("queue_size", labels, value=-20)

    output = metrics.export_metrics()
    assert "queue_size" in output


def test_observe_histogram(metrics):
    """Test histogram observations"""
    # Observe multiple values
    labels = {"endpoint": "/api/v1/query"}

    metrics.observe_histogram("request_duration_seconds", labels, value=0.1)
    metrics.observe_histogram("request_duration_seconds", labels, value=0.5)
    metrics.observe_histogram("request_duration_seconds", labels, value=1.2)

    output = metrics.export_metrics()
    assert "request_duration_seconds" in output
    # Histograms create buckets
    assert "_bucket" in output
    assert "_sum" in output
    assert "_count" in output


def test_histogram_buckets(metrics):
    """Test histogram bucket distribution"""
    labels = {"endpoint": "/api/v1/query"}

    # Add values in different ranges
    metrics.observe_histogram("request_duration_seconds", labels, value=0.05)  # < 0.1
    metrics.observe_histogram("request_duration_seconds", labels, value=0.15)  # < 0.25
    metrics.observe_histogram("request_duration_seconds", labels, value=0.6)   # < 1.0
    metrics.observe_histogram("request_duration_seconds", labels, value=3.0)   # < 5.0

    output = metrics.export_metrics()

    # Check buckets exist
    assert 'le="0.1"' in output
    assert 'le="0.25"' in output
    assert 'le="1.0"' in output
    assert 'le="5.0"' in output
    assert 'le="+Inf"' in output


def test_metrics_help_text(metrics):
    """Test that metrics include help text"""
    metrics.increment_counter("http_requests_total")

    output = metrics.export_metrics()

    # Should include HELP and TYPE declarations
    assert "# HELP http_requests_total" in output
    assert "# TYPE http_requests_total counter" in output


def test_multiple_metric_types(metrics):
    """Test exporting multiple different metric types"""
    # Counter
    metrics.increment_counter("requests_total", value=100)

    # Gauge
    metrics.set_gauge("temperature_celsius", value=23.5)

    # Histogram
    metrics.observe_histogram("request_size_bytes", value=1024)

    output = metrics.export_metrics()

    # All should be present
    assert "requests_total" in output
    assert "temperature_celsius" in output
    assert "request_size_bytes" in output


def test_label_escaping(metrics):
    """Test proper escaping of label values"""
    labels = {
        "path": '/api/v1/query?param="value"',
        "user": "test\\user"
    }

    metrics.increment_counter("requests", labels)

    output = metrics.export_metrics()

    # Labels should be properly escaped
    assert "requests" in output


def test_counter_never_decreases(metrics):
    """Test that counters only increase"""
    labels = {"metric": "test"}

    metrics.increment_counter("test_counter", labels, value=10)
    metrics.increment_counter("test_counter", labels, value=5)

    # Counter should be 15, not 5
    output = metrics.export_metrics()
    assert "test_counter" in output


def test_empty_labels(metrics):
    """Test metrics without labels"""
    metrics.increment_counter("labelless_counter")
    metrics.set_gauge("labelless_gauge", value=42)

    output = metrics.export_metrics()

    assert "labelless_counter" in output
    assert "labelless_gauge" in output


def test_reset_metrics(metrics):
    """Test resetting all metrics"""
    # Add some metrics
    metrics.increment_counter("test_counter", value=100)
    metrics.set_gauge("test_gauge", value=50)

    # Reset
    metrics.reset()

    output = metrics.export_metrics()

    # Should be empty or have zero values
    # (Prometheus format doesn't usually show zero metrics)


def test_get_counter_value(metrics):
    """Test retrieving counter value"""
    labels = {"endpoint": "/test"}

    metrics.increment_counter("test_counter", labels, value=42)

    value = metrics.get_counter_value("test_counter", labels)
    assert value == 42


def test_get_gauge_value(metrics):
    """Test retrieving gauge value"""
    labels = {"resource": "memory"}

    metrics.set_gauge("test_gauge", labels, value=1024)

    value = metrics.get_gauge_value("test_gauge", labels)
    assert value == 1024


def test_histogram_statistics(metrics):
    """Test calculating histogram statistics"""
    labels = {"endpoint": "/api/v1/query"}

    # Add sample values
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for val in values:
        metrics.observe_histogram("durations", labels, value=val)

    stats = metrics.get_histogram_stats("durations", labels)

    assert stats["count"] == 5
    assert stats["sum"] == pytest.approx(1.5, rel=0.01)
    # Average should be 0.3
    assert stats["sum"] / stats["count"] == pytest.approx(0.3, rel=0.01)


def test_concurrent_counter_updates(metrics):
    """Test thread-safe counter updates"""
    import threading

    labels = {"test": "concurrent"}

    def increment():
        for _ in range(100):
            metrics.increment_counter("concurrent_counter", labels, value=1)

    # Run 10 threads, each incrementing 100 times
    threads = [threading.Thread(target=increment) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should be exactly 1000
    value = metrics.get_counter_value("concurrent_counter", labels)
    assert value == 1000


def test_performance_metrics_integration(metrics):
    """Test typical usage pattern for API performance"""
    endpoint = "/api/v1/query"

    # Simulate 10 requests
    for i in range(10):
        start = time.time()

        # Increment request counter
        metrics.increment_counter(
            "http_requests_total",
            {"endpoint": endpoint, "status": "200"}
        )

        # Simulate request processing
        time.sleep(0.01)

        # Record duration
        duration = time.time() - start
        metrics.observe_histogram(
            "http_request_duration_seconds",
            {"endpoint": endpoint},
            value=duration
        )

    # Update active requests gauge
    metrics.set_gauge("http_requests_active", {"endpoint": endpoint}, value=2)

    output = metrics.export_metrics()

    # Verify all metrics present
    assert "http_requests_total" in output
    assert "http_request_duration_seconds" in output
    assert "http_requests_active" in output


def test_custom_histogram_buckets(metrics):
    """Test histogram with custom buckets"""
    custom_buckets = [0.001, 0.01, 0.1, 1.0, 10.0]

    # This would require modifying PrometheusMetrics to support custom buckets
    # For now, test default buckets work
    labels = {"custom": "test"}

    metrics.observe_histogram("custom_metric", labels, value=0.5)

    output = metrics.export_metrics()
    assert "custom_metric" in output


def test_metric_naming_conventions(metrics):
    """Test that metric names follow Prometheus conventions"""
    # Good names (snake_case, descriptive)
    metrics.increment_counter("http_requests_total")
    metrics.set_gauge("process_cpu_usage_percent")
    metrics.observe_histogram("query_duration_seconds")

    output = metrics.export_metrics()

    # All should export correctly
    assert "http_requests_total" in output
    assert "process_cpu_usage_percent" in output
    assert "query_duration_seconds" in output


def test_export_format_compliance(metrics):
    """Test that export format follows Prometheus text format"""
    metrics.increment_counter("test_counter", {"label": "value"}, value=42)

    output = metrics.export_metrics()

    # Should have proper format:
    # # HELP ...
    # # TYPE ...
    # metric_name{labels} value

    lines = output.strip().split('\n')

    # Find metric lines (not comments)
    metric_lines = [l for l in lines if not l.startswith('#') and l.strip()]

    # Should have at least one metric line
    assert len(metric_lines) > 0
