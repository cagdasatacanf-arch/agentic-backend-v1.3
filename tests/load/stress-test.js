/**
 * K6 Stress Test - Find Breaking Point
 *
 * Gradually increases load to find the system's breaking point.
 *
 * Usage:
 *   k6 run tests/load/stress-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 10 },    // Baseline
    { duration: '5m', target: 50 },    // Normal load
    { duration: '5m', target: 100 },   // High load
    { duration: '5m', target: 200 },   // Stress
    { duration: '5m', target: 300 },   // Breaking point?
    { duration: '5m', target: 400 },   // Beyond breaking point
    { duration: '10m', target: 0 },    // Recovery
  ],

  thresholds: {
    'http_req_duration': ['p(99)<10000'], // 99% under 10s
    'errors': ['rate<0.5'],                // Allow up to 50% errors at peak
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const payload = JSON.stringify({
    query: 'What is artificial intelligence?',
    user_id: `stress_user_${__VU}`,
  });

  const response = http.post(`${BASE_URL}/api/v1/query`, payload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: '30s',
  });

  const success = check(response, {
    'status is 200 or 429': (r) => r.status === 200 || r.status === 429,
  });

  errorRate.add(!success);

  if (response.status >= 500) {
    console.log(`[VU ${__VU}] Server error at load level ${__STAGE}`);
  }

  sleep(1);
}
