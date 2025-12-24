/**
 * K6 Load Test - Query Endpoint
 *
 * Tests the /api/v1/query endpoint under various load patterns.
 *
 * Usage:
 *   # Smoke test (minimal load)
 *   k6 run tests/load/query-load-test.js
 *
 *   # Load test (normal traffic)
 *   k6 run --vus 10 --duration 5m tests/load/query-load-test.js
 *
 *   # Stress test (peak traffic)
 *   k6 run --vus 50 --duration 10m tests/load/query-load-test.js
 *
 *   # Spike test
 *   k6 run tests/load/query-load-test.js --stage 1m:10,30s:100,1m:10
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const queryDuration = new Trend('query_duration');
const cachHitRate = new Rate('cache_hits');
const rateLimitErrors = new Counter('rate_limit_errors');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 },   // Ramp up to 20 users
    { duration: '5m', target: 20 },   // Stay at 20 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],

  thresholds: {
    'http_req_duration': ['p(95)<5000'],  // 95% of requests must complete below 5s
    'http_req_failed': ['rate<0.1'],      // Less than 10% of requests can fail
    'errors': ['rate<0.1'],                // Error rate should be below 10%
    'query_duration': ['p(90)<3000'],      // 90% of queries under 3s
  },

  // Additional options
  noConnectionReuse: false,
  userAgent: 'K6LoadTest/1.0',
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test queries of varying complexity
const TEST_QUERIES = [
  // Simple queries (should be fast, possibly cached)
  'What is 2 + 2?',
  'What is the capital of France?',
  'Define machine learning',

  // Medium complexity queries
  'Explain how neural networks work',
  'What are the main differences between SQL and NoSQL databases?',
  'How does gradient descent optimization work?',

  // Complex queries (longer processing time)
  'Explain the architecture of a transformer model in detail',
  'Compare and contrast microservices vs monolithic architectures',
  'Describe the full machine learning pipeline from data collection to deployment',
];

// User pool for testing
const USER_POOL_SIZE = 100;

function getRandomQuery() {
  return TEST_QUERIES[Math.floor(Math.random() * TEST_QUERIES.length)];
}

function getRandomUserId() {
  return `load_test_user_${Math.floor(Math.random() * USER_POOL_SIZE)}`;
}

export default function () {
  group('Query API', function () {
    const payload = JSON.stringify({
      query: getRandomQuery(),
      user_id: getRandomUserId(),
      session_id: `session_${__VU}_${__ITER}`,
      stream: false,
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: '30s',
    };

    const startTime = new Date();
    const response = http.post(`${BASE_URL}/api/v1/query`, payload, params);
    const duration = new Date() - startTime;

    // Record metrics
    queryDuration.add(duration);

    // Check response
    const success = check(response, {
      'status is 200': (r) => r.status === 200,
      'has answer': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.answer && body.answer.length > 0;
        } catch {
          return false;
        }
      },
      'response time < 10s': (r) => r.timings.duration < 10000,
    });

    if (!success) {
      errorRate.add(1);

      if (response.status === 429) {
        rateLimitErrors.add(1);
        console.log('Rate limit hit for user:', getRandomUserId());
      } else if (response.status >= 500) {
        console.log('Server error:', response.status, response.body.substring(0, 200));
      }
    } else {
      errorRate.add(0);

      // Check if response was cached
      try {
        const body = JSON.parse(response.body);
        if (body.cached || body.cache_hit) {
          cachHitRate.add(1);
        } else {
          cachHitRate.add(0);
        }
      } catch {
        cachHitRate.add(0);
      }
    }
  });

  // Think time (simulate realistic user behavior)
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

export function handleSummary(data) {
  return {
    'load-test-results.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;

  const metrics = data.metrics;

  let summary = '\n';
  summary += indent + '==== Load Test Summary ====\n\n';

  summary += indent + 'HTTP Requests:\n';
  summary += indent + `  Total:     ${metrics.http_reqs.values.count}\n`;
  summary += indent + `  Failed:    ${metrics.http_req_failed.values.rate * 100}%\n`;
  summary += indent + `  Duration:  p(95)=${metrics.http_req_duration.values['p(95)']}ms\n\n`;

  summary += indent + 'Query Performance:\n';
  summary += indent + `  Avg:       ${metrics.query_duration.values.avg}ms\n`;
  summary += indent + `  p(90):     ${metrics.query_duration.values['p(90)']}ms\n`;
  summary += indent + `  p(95):     ${metrics.query_duration.values['p(95)']}ms\n\n`;

  summary += indent + 'Cache Performance:\n';
  summary += indent + `  Hit Rate:  ${metrics.cache_hits.values.rate * 100}%\n\n`;

  summary += indent + 'Errors:\n';
  summary += indent + `  Error Rate:      ${metrics.errors.values.rate * 100}%\n`;
  summary += indent + `  Rate Limit Hits: ${metrics.rate_limit_errors.values.count}\n\n`;

  summary += indent + 'Virtual Users:\n';
  summary += indent + `  Peak:      ${metrics.vus_max.values.max}\n\n`;

  return summary;
}
