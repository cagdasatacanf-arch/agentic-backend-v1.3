# Horizontal Scaling Guide

Complete guide to scaling the Agentic Backend for production workloads.

## 🎯 Overview

This system supports horizontal scaling with:
- **Nginx Load Balancer**: Distributes requests across API instances
- **Stateless API Instances**: Share state via Redis and Qdrant
- **Shared Storage**: Redis for caching, Qdrant for vectors
- **Monitoring**: Prometheus tracks all instances

## 🚀 Quick Start

### 1. Single Instance (Development)

```bash
docker-compose up -d
```

Access: http://localhost:8000

### 2. Scaled Deployment (Production)

```bash
# Start infrastructure
docker-compose up -d redis qdrant jaeger prometheus grafana

# Start scaled API (3 instances + load balancer)
docker-compose -f docker-compose.yml -f docker-compose.scaled.yml up -d
```

Access: http://localhost (load balanced)

## 📊 Architecture

### Load Balanced Architecture

```
                    ┌─────────────┐
                    │   Nginx LB  │ :80
                    │  (least_conn)│
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
         ┌──▼──┐        ┌──▼──┐       ┌──▼──┐
         │API-1│        │API-2│       │API-3│
         │:8000│        │:8000│       │:8000│
         └──┬──┘        └──┬──┘       └──┬──┘
            │              │              │
            └──────────────┼──────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
         ┌──▼────┐    ┌───▼───┐     ┌───▼───┐
         │ Redis │    │Qdrant │     │Jaeger │
         │  :6379│    │ :6333 │     │ :4317 │
         └───────┘    └───────┘     └───────┘
```

### Components

1. **Nginx**: Load balancer and reverse proxy
   - Algorithm: Least connections
   - Health checks with automatic failover
   - SSE streaming support
   - Request buffering disabled for streaming

2. **API Instances**: Stateless FastAPI applications
   - Each runs with Gunicorn (2 workers per instance)
   - Share data via Redis and Qdrant
   - Independent scaling

3. **Shared Services**:
   - **Redis**: Session state, caching, costs, RBAC
   - **Qdrant**: Vector embeddings
   - **Prometheus**: Metrics aggregation
   - **Grafana**: Unified dashboards

## 🔧 Configuration

### Load Balancer Algorithm

Edit `nginx.conf`:

```nginx
upstream agentic_backend {
    # Choose load balancing strategy:

    # 1. Round-robin (default)
    # Simple rotation through servers

    # 2. Least connections (recommended for AI workloads)
    least_conn;

    # 3. IP hash (sticky sessions)
    # ip_hash;

    server api-1:8000 max_fails=3 fail_timeout=30s;
    server api-2:8000 max_fails=3 fail_timeout=30s;
    server api-3:8000 max_fails=3 fail_timeout=30s;
}
```

### Instance Count

#### Scale to 5 instances:

Edit `docker-compose.scaled.yml`, add:

```yaml
api-4:
  # ... same as api-1 but with INSTANCE_ID=api-4 ...

api-5:
  # ... same as api-1 but with INSTANCE_ID=api-5 ...
```

Update `nginx.conf`:

```nginx
upstream agentic_backend {
    least_conn;
    server api-1:8000 max_fails=3 fail_timeout=30s;
    server api-2:8000 max_fails=3 fail_timeout=30s;
    server api-3:8000 max_fails=3 fail_timeout=30s;
    server api-4:8000 max_fails=3 fail_timeout=30s;
    server api-5:8000 max_fails=3 fail_timeout=30s;
}
```

Restart:

```bash
docker-compose -f docker-compose.yml -f docker-compose.scaled.yml up -d --scale api=5
```

### Workers Per Instance

Edit instance command in `docker-compose.scaled.yml`:

```yaml
command: gunicorn app.main:app \
  --workers 4 \  # Increase workers
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --max-requests 1000 \  # Restart workers after 1000 requests
  --max-requests-jitter 100
```

**Recommendations**:
- Workers: `2 * CPU cores + 1`
- Max 4-6 workers per instance for AI workloads
- More instances > more workers per instance

## 📈 Capacity Planning

### Current Configuration

With 3 instances × 2 workers = 6 concurrent requests

**Estimated capacity**:
- Simple queries: 60-120 req/min
- Complex queries: 20-40 req/min
- Mixed workload: 40-80 req/min

### Scaling Calculator

```python
# Instances needed for target throughput
target_rps = 10  # requests per second
avg_response_time = 5  # seconds
workers_per_instance = 2
instances = (target_rps * avg_response_time) / workers_per_instance

# Example: 10 req/sec with 5sec responses
# instances = (10 * 5) / 2 = 25 workers needed
# With 2 workers/instance = 13 instances
```

### Resource Requirements

Per API instance:
- **CPU**: 1-2 cores
- **Memory**: 1-2GB
- **Network**: 100 Mbps

For 3 instances + infrastructure:
- **CPU**: 8 cores
- **Memory**: 12GB
- **Storage**: 50GB (includes data retention)

## 🔍 Monitoring Scaled Deployment

### Check Instance Health

```bash
# Via Nginx
curl http://localhost/health

# Individual instances (requires docker network access)
docker exec agentic-api-1 curl http://localhost:8000/api/v1/health
docker exec agentic-api-2 curl http://localhost:8000/api/v1/health
docker exec agentic-api-3 curl http://localhost:8000/api/v1/health
```

### View Load Distribution

Check Nginx logs:

```bash
docker exec agentic-nginx tail -f /var/log/nginx/access.log
```

### Prometheus Metrics

View per-instance metrics:

```promql
# Request rate per instance
sum(rate(http_requests_total[5m])) by (instance)

# Latency per instance
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) by (instance)
```

### Grafana Dashboard

The main dashboard shows aggregated metrics across all instances.

## 🚨 Health Checks & Failover

### Nginx Health Checks

Nginx monitors instances and automatically removes failed instances:

```nginx
server api-1:8000 max_fails=3 fail_timeout=30s;
```

- **max_fails**: Mark unhealthy after 3 failed requests
- **fail_timeout**: Wait 30s before retrying

### Application Health Check

Each instance exposes:
- `/api/v1/health`: Application health
- `/api/v1/monitoring/ready`: Kubernetes readiness
- `/api/v1/monitoring/live`: Kubernetes liveness

### Manual Failover

Remove instance from rotation:

```bash
# Stop instance
docker stop agentic-api-2

# Nginx automatically redirects to healthy instances
# Restart when ready
docker start agentic-api-2
```

## 🔄 Zero-Downtime Deployment

### Rolling Update

Update one instance at a time:

```bash
#!/bin/bash
# rolling-update.sh

for instance in api-1 api-2 api-3; do
    echo "Updating $instance..."

    # Stop instance
    docker stop $instance

    # Pull new image
    docker-compose -f docker-compose.scaled.yml pull $instance

    # Start with new code
    docker-compose -f docker-compose.scaled.yml up -d $instance

    # Wait for health check
    sleep 10

    echo "$instance updated successfully"
done
```

### Blue-Green Deployment

1. Start new instances (green):
   ```bash
   docker-compose -f docker-compose.scaled.yml up -d api-4 api-5 api-6
   ```

2. Update nginx.conf to include new instances

3. Reload Nginx:
   ```bash
   docker exec agentic-nginx nginx -s reload
   ```

4. Monitor for issues

5. Remove old instances (blue):
   ```bash
   docker stop agentic-api-1 agentic-api-2 agentic-api-3
   ```

## 🌐 Cloud Deployment

### Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-backend
  template:
    metadata:
      labels:
        app: agentic-backend
    spec:
      containers:
      - name: api
        image: your-registry/agentic-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: QDRANT_HOST
          value: "qdrant-service"
        resources:
          requests:
            cpu: "1000m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /api/v1/monitoring/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/monitoring/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-backend
spec:
  selector:
    app: agentic-backend
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentic-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### AWS ECS

Use Application Load Balancer with target groups.

### GCP Cloud Run

Supports auto-scaling based on request concurrency.

## 🔐 Production Hardening

### HTTPS/TLS

Update `nginx.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://agentic_backend;
    }
}
```

### Rate Limiting (Nginx Level)

Add to `nginx.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://agentic_backend;
    }
}
```

### IP Whitelisting

```nginx
location /api/v1/security/ {
    allow 10.0.0.0/8;
    allow 192.168.1.0/24;
    deny all;

    proxy_pass http://agentic_backend;
}
```

## 📊 Performance Tuning

### Optimize Redis

```yaml
redis:
  command: redis-server \
    --appendonly yes \
    --maxmemory 2gb \
    --maxmemory-policy allkeys-lru \
    --tcp-backlog 511 \
    --timeout 300 \
    --tcp-keepalive 300
```

### Optimize Qdrant

```yaml
qdrant:
  environment:
    - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=100
    - QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD_KB=50000
```

### Connection Pooling

Application already uses connection pooling:
- Redis: `redis.asyncio` with connection pool
- Qdrant: Built-in connection management

## 🐛 Troubleshooting

### Unbalanced Load

**Symptom**: One instance receiving more requests

**Solution**:
1. Check health status of all instances
2. Verify `least_conn` algorithm in nginx.conf
3. Check instance resource usage (CPU/memory)

### High Latency

**Symptom**: Slow responses

**Solution**:
1. Check Prometheus for bottlenecks
2. Scale up instances
3. Increase workers per instance
4. Optimize database queries
5. Enable more aggressive caching

### Memory Leaks

**Symptom**: Memory usage grows over time

**Solution**:
1. Enable worker recycling:
   ```yaml
   command: gunicorn ... --max-requests 1000 --max-requests-jitter 100
   ```
2. Monitor with Prometheus
3. Implement memory profiling

### Session Stickiness Issues

**Symptom**: Users losing session state

**Solution**:
- System is stateless by design
- Session data stored in Redis
- No sticky sessions needed
- Verify Redis connectivity

## 📚 Best Practices

1. **Start Small**: Begin with 2-3 instances, scale up as needed
2. **Monitor First**: Use Prometheus/Grafana to identify bottlenecks
3. **Gradual Scaling**: Add instances incrementally
4. **Test Under Load**: Use load testing tools (k6, Locust)
5. **Health Checks**: Always configure proper health checks
6. **Rolling Updates**: Never update all instances simultaneously
7. **Resource Limits**: Set CPU/memory limits to prevent resource exhaustion
8. **Auto-Scaling**: Configure based on actual metrics, not just CPU

## 🧪 Load Testing

Example with k6:

```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },  // Ramp up to 10 users
    { duration: '5m', target: 10 },  // Stay at 10 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
};

export default function () {
  let res = http.post('http://localhost/api/v1/query', JSON.stringify({
    query: 'What is 2+2?',
    agent_type: 'math'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 5s': (r) => r.timings.duration < 5000,
  });

  sleep(1);
}
```

Run:
```bash
k6 run load-test.js
```

## 📖 Further Reading

- [Nginx Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [Gunicorn Deployment](https://docs.gunicorn.org/en/stable/deploy.html)
- [Kubernetes Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
