# Production Monitoring Setup

Production-grade monitoring with Prometheus and Grafana for the Agentic Backend.

## 🎯 Overview

This setup provides comprehensive monitoring and observability:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Metrics**: HTTP requests, agent performance, costs, cache, security

## 🚀 Quick Start

### 1. Start Monitoring Stack

```bash
docker-compose up -d prometheus grafana
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin` (or set via `GRAFANA_PASSWORD` env var)

- **Prometheus**: http://localhost:9090

### 3. View Metrics

The main dashboard "Agentic Backend - Production Monitoring" is automatically provisioned.

## 📊 Available Metrics

### HTTP Metrics

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency

### Agent Metrics

- `agent_queries_total` - Total agent queries by type
- `agent_query_duration_seconds` - Agent query duration

### Cost Metrics

- `api_cost_dollars_total` - Total API costs
- `api_tokens_total` - Token usage by model

### Cache Metrics

- `cache_operations_total` - Cache hits/misses
- `cache_size_bytes` - Cache size

### Security Metrics

- `auth_attempts_total` - Authentication attempts
- `permission_checks_total` - Permission checks

### System Metrics

- `active_sessions` - Active user sessions
- `active_requests` - Active requests

## 📈 Dashboards

### Main Dashboard Panels

1. **HTTP Requests** - Total request count
2. **Active Sessions** - Current active sessions
3. **Request Rate** - Requests per second over time
4. **Request Latency** - p50, p90, p99 latencies
5. **Agent Query Rate** - Queries by agent type
6. **Agent Duration** - Query duration by agent
7. **API Cost** - Cost by model
8. **Token Usage** - Input/output tokens by model
9. **Cache Hit Rate** - Cache effectiveness
10. **Total Cost** - Cumulative API costs
11. **Total Tokens** - Cumulative token usage
12. **Authentication** - Success/failure rates
13. **Permissions** - Allowed/denied checks

## 🔍 Querying Metrics

### Prometheus Queries

Access Prometheus UI at http://localhost:9090

Example queries:

```promql
# Request rate
rate(http_requests_total[5m])

# Average latency
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Cache hit rate
rate(cache_operations_total{result="hit"}[5m]) / (rate(cache_operations_total[5m]))

# Cost per hour
rate(api_cost_dollars_total[1h])
```

### API Endpoints

Get metrics via API:

```bash
# Prometheus format
curl http://localhost:8000/api/v1/monitoring/metrics

# JSON summary
curl http://localhost:8000/api/v1/monitoring/summary

# Health check
curl http://localhost:8000/api/v1/monitoring/health
```

## 🔧 Configuration

### Prometheus

Edit `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s      # Scrape frequency
  evaluation_interval: 15s  # Rule evaluation frequency

scrape_configs:
  - job_name: 'agentic-backend'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/api/v1/monitoring/metrics'
```

### Grafana

Datasources and dashboards are automatically provisioned via:

- `monitoring/grafana/datasources/prometheus.yml`
- `monitoring/grafana/dashboards/agentic-backend.json`

## 📝 Custom Metrics

Record custom metrics via API:

```bash
curl -X POST http://localhost:8000/api/v1/monitoring/record \
  -H "Content-Type: application/json" \
  -d '{
    "metric_type": "counter",
    "name": "custom_events_total",
    "value": 1,
    "labels": {"event_type": "signup"}
  }'
```

## 🚨 Alerting (Optional)

To add alerting:

1. Set up Alertmanager:

```yaml
# docker-compose.yml
alertmanager:
  image: prom/alertmanager:latest
  ports:
    - "9093:9093"
  volumes:
    - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

2. Create alert rules in `monitoring/alerts/`:

```yaml
# monitoring/alerts/api.yml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

3. Update `prometheus.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'
```

## 🔐 Security

### Production Hardening

1. **Enable authentication** in Grafana:
   - Set strong `GRAFANA_PASSWORD`
   - Disable anonymous access

2. **Restrict Prometheus access**:
   - Use reverse proxy with authentication
   - Firewall rules

3. **Use HTTPS**:
   - Configure TLS certificates
   - Enable HTTPS in Grafana

## 📦 Data Retention

Prometheus retains data for 30 days by default:

```yaml
command:
  - '--storage.tsdb.retention.time=30d'
```

Adjust based on your needs:
- Short-term: `7d` (saves disk space)
- Long-term: `90d` (more historical data)

## 🧹 Maintenance

### Clear Old Data

```bash
# Stop Prometheus
docker-compose stop prometheus

# Remove data
docker volume rm agentic-backend-v1.3_prometheus_data

# Restart
docker-compose up -d prometheus
```

### Backup Dashboards

Dashboards are in `monitoring/grafana/dashboards/` and version controlled.

## 📚 Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)

## 🐛 Troubleshooting

### Metrics not appearing

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify API is running: `docker-compose ps api`
3. Test metrics endpoint:
   ```bash
   curl http://localhost:8000/api/v1/monitoring/metrics
   ```

### Grafana not connecting

1. Check Prometheus is running: `docker-compose ps prometheus`
2. Test Prometheus from Grafana container:
   ```bash
   docker exec agentic-grafana curl http://prometheus:9090/api/v1/query?query=up
   ```

### High memory usage

Prometheus memory usage grows with:
- Number of metrics
- Cardinality (unique label combinations)
- Retention period

Reduce by:
- Decreasing retention time
- Reducing scrape frequency
- Limiting metric labels
