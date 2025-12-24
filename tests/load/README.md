# Load Testing with K6

Comprehensive load tests for the Agentic Backend using [k6](https://k6.io/).

## Prerequisites

Install k6:

```bash
# macOS
brew install k6

# Linux (Debian/Ubuntu)
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Docker
docker pull grafana/k6
```

## Test Scripts

### 1. query-load-test.js - Standard Load Test

Tests normal production traffic patterns.

**Stages:**
- Ramp up: 2min to 10 users
- Sustained: 5min at 10 users
- Increase: 2min to 20 users
- Peak: 5min at 20 users
- Ramp down: 2min to 0 users

**Run:**
```bash
k6 run tests/load/query-load-test.js
```

**With custom config:**
```bash
# Set base URL
BASE_URL=https://your-domain.com k6 run tests/load/query-load-test.js

# Override VUs and duration
k6 run --vus 50 --duration 10m tests/load/query-load-test.js
```

### 2. stress-test.js - Stress Test

Gradually increases load to find breaking point.

**Stages:**
- 10 → 50 → 100 → 200 → 300 → 400 users
- Recovery period at the end

**Run:**
```bash
k6 run tests/load/stress-test.js
```

### 3. spike-test.js - Spike Test

Tests sudden traffic surges (e.g., viral post, DDoS).

**Stages:**
- Normal: 10 users
- Spike: 200 users in 30 seconds
- Recovery: Back to 10 users

**Run:**
```bash
k6 run tests/load/spike-test.js
```

## Running All Tests

Use the run script:

```bash
./tests/load/run-all-tests.sh
```

Or manually:

```bash
# Run each test and save results
k6 run --out json=results-load.json tests/load/query-load-test.js
k6 run --out json=results-stress.json tests/load/stress-test.js
k6 run --out json=results-spike.json tests/load/spike-test.js
```

## Results Analysis

### Reading K6 Output

K6 provides detailed metrics:

```
http_req_duration.............: avg=1.2s  min=234ms med=987ms max=4.5s  p(90)=2.1s p(95)=2.8s
http_req_failed...............: 2.34%    ✓ 45      ✗ 1,920
http_reqs.....................: 1,965    32.75/s
iterations....................: 1,965    32.75/s
vus...........................: 10       min=0     max=20
vus_max.......................: 20       min=20    max=20
```

**Key Metrics:**
- `http_req_duration`: How long requests take (p95 is most important)
- `http_req_failed`: Percentage of failed requests
- `http_reqs`: Total requests and requests per second
- `vus`: Current/max virtual users

### Interpreting Results

**Good Performance:**
- p(95) response time < 3 seconds
- Error rate < 1%
- No 500 errors
- Consistent response times

**Warning Signs:**
- p(95) response time > 5 seconds
- Error rate > 5%
- Increasing response times over time (memory leak?)
- 429 rate limit errors (need to scale or adjust limits)

**Critical Issues:**
- 500 errors
- Error rate > 10%
- Response times > 10 seconds
- System crashes

## Performance Targets

Based on current infrastructure (3 API instances, 2 workers each):

| Metric | Target | Limit |
|--------|--------|-------|
| Concurrent Users | 50 | 100 |
| Requests/Second | 10 | 20 |
| p(95) Response Time | < 3s | < 5s |
| Error Rate | < 1% | < 5% |
| Availability | > 99.5% | > 99% |

## Scaling Recommendations

If load tests show:

**High response times (> 5s p95):**
- Add more API instances
- Increase workers per instance
- Enable more aggressive caching
- Optimize database queries

**Rate limit errors:**
- Increase rate limits for legitimate users
- Implement user tiers with different limits
- Add request queuing

**Memory issues:**
- Enable worker recycling: `--max-requests 1000`
- Increase instance memory limits
- Review cache sizes

**CPU bottleneck:**
- Add more CPU cores per instance
- Horizontal scaling (more instances)
- Implement result caching

## CI/CD Integration

Add to GitHub Actions:

```yaml
- name: Run Load Tests
  run: |
    k6 run --vus 10 --duration 2m tests/load/query-load-test.js
```

For scheduled performance regression testing:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
```

## Monitoring During Tests

While running load tests, monitor:

1. **Prometheus Metrics:**
   - http://localhost:9090

2. **Grafana Dashboards:**
   - http://localhost:3000

3. **System Resources:**
   ```bash
   docker stats
   ```

4. **API Logs:**
   ```bash
   docker logs -f agentic-api-1
   ```

## Best Practices

1. **Run tests in isolated environment** - Don't test against production!

2. **Start small** - Begin with smoke tests (1-2 VUs) before full load

3. **Monitor throughout** - Watch metrics in real-time

4. **Test incrementally** - Gradually increase load to find limits

5. **Run multiple times** - Average results across 3+ runs

6. **Test during deployment** - Ensure new code doesn't degrade performance

7. **Document results** - Keep baseline metrics for comparison

## Troubleshooting

### K6 connection errors

```
WARN[0001] Request Failed error="Get http://localhost:8000/api/v1/query: dial tcp [::1]:8000: connect: connection refused"
```

**Fix:** Ensure API is running and accessible:
```bash
curl http://localhost:8000/api/v1/health
```

### Rate limiting too aggressive

```
WARN[0030] Status 429 (Too Many Requests) received for 50% of requests
```

**Fix:** Adjust rate limits or distribute load across more users:
```javascript
function getRandomUserId() {
  return `user_${Math.floor(Math.random() * 1000)}`; // Increase pool size
}
```

### Out of memory

```
ERRO[0120] panic: runtime error: out of memory
```

**Fix:**
- Reduce concurrent VUs
- Increase system memory
- Scale API instances

## Further Reading

- [K6 Documentation](https://k6.io/docs/)
- [Load Testing Best Practices](https://k6.io/docs/testing-guides/test-types/)
- [Performance Testing Guide](https://k6.io/docs/testing-guides/)
