/**
 * K6 Spike Test - Sudden Traffic Surge
 *
 * Tests system behavior under sudden traffic spikes.
 *
 * Usage:
 *   k6 run tests/load/spike-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 10 },    // Normal load
    { duration: '30s', target: 200 },  // SPIKE!
    { duration: '2m', target: 200 },   // Sustained spike
    { duration: '30s', target: 10 },   // Back to normal
    { duration: '2m', target: 10 },    // Recovery
  ],

  thresholds: {
    'http_req_duration': ['p(95)<8000'],  // Allow slower responses during spike
    'http_req_failed': ['rate<0.2'],      // Allow 20% failures during spike
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const response = http.post(
    `${BASE_URL}/api/v1/query`,
    JSON.stringify({
      query: 'Test spike resilience',
      user_id: `spike_user_${__VU}`,
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  check(response, {
    'survived spike': (r) => r.status < 500,
  });

  sleep(Math.random() * 2);
}
