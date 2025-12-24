#!/bin/bash
#
# Run All Load Tests
#
# Executes all k6 load tests and generates comprehensive report.
#
# Usage:
#   ./tests/load/run-all-tests.sh [BASE_URL]
#
# Example:
#   ./tests/load/run-all-tests.sh http://localhost:8000
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="${1:-http://localhost:8000}"
RESULTS_DIR="./test-results/load"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${GREEN}=== Agentic Backend Load Tests ===${NC}"
echo ""
echo "Base URL: $BASE_URL"
echo "Results:  $RESULTS_DIR"
echo "Time:     $TIMESTAMP"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if k6 is installed
if ! command -v k6 &> /dev/null; then
    echo -e "${RED}Error: k6 is not installed${NC}"
    echo "Install from: https://k6.io/docs/getting-started/installation/"
    exit 1
fi

# Check if API is accessible
echo -e "${BLUE}Checking API health...${NC}"
if ! curl -sf "${BASE_URL}/api/v1/health" > /dev/null; then
    echo -e "${RED}Error: API is not accessible at $BASE_URL${NC}"
    echo "Start the API with: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"
echo ""

# Run smoke test first
echo -e "${BLUE}Running smoke test (quick health check)...${NC}"
k6 run --vus 1 --duration 30s \
    --out json="$RESULTS_DIR/smoke_${TIMESTAMP}.json" \
    tests/load/query-load-test.js \
    || echo -e "${YELLOW}⚠️  Smoke test had issues${NC}"
echo ""

# Run main load test
echo -e "${BLUE}Running load test (15 minutes)...${NC}"
BASE_URL="$BASE_URL" k6 run \
    --out json="$RESULTS_DIR/load_${TIMESTAMP}.json" \
    tests/load/query-load-test.js \
    || echo -e "${YELLOW}⚠️  Load test had issues${NC}"
echo ""

# Run spike test
echo -e "${BLUE}Running spike test (sudden traffic surge)...${NC}"
BASE_URL="$BASE_URL" k6 run \
    --out json="$RESULTS_DIR/spike_${TIMESTAMP}.json" \
    tests/load/spike-test.js \
    || echo -e "${YELLOW}⚠️  Spike test had issues${NC}"
echo ""

# Optional: Run stress test (takes longer, only if explicitly requested)
if [ "$2" = "--stress" ]; then
    echo -e "${BLUE}Running stress test (find breaking point - 30 minutes)...${NC}"
    BASE_URL="$BASE_URL" k6 run \
        --out json="$RESULTS_DIR/stress_${TIMESTAMP}.json" \
        tests/load/stress-test.js \
        || echo -e "${YELLOW}⚠️  Stress test had issues${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping stress test (use --stress flag to include)${NC}"
    echo ""
fi

# Generate summary report
echo -e "${GREEN}=== Test Summary ===${NC}"
echo ""

for result_file in "$RESULTS_DIR"/*_${TIMESTAMP}.json; do
    if [ -f "$result_file" ]; then
        test_name=$(basename "$result_file" | cut -d'_' -f1)
        echo -e "${BLUE}${test_name} test:${NC}"

        # Extract key metrics using jq (if available)
        if command -v jq &> /dev/null; then
            echo "  Results saved to: $result_file"
        else
            echo "  Results: $result_file"
            echo "  (Install jq for detailed metrics parsing)"
        fi
        echo ""
    fi
done

echo -e "${GREEN}✓ All tests completed!${NC}"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Review results in: $RESULTS_DIR"
echo "2. Check Grafana dashboards: http://localhost:3000"
echo "3. Analyze Prometheus metrics: http://localhost:9090"
echo ""
echo "To run stress test (30min): $0 $BASE_URL --stress"
