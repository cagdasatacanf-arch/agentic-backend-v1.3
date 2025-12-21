#!/bin/bash

# ============================================================================
# Agentic Backend v1.3 - System Test Script
# ============================================================================
# This script runs comprehensive tests to verify the system is working
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_test() {
    echo -e "\n${YELLOW}Test: $1${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get API key from .env
if [ ! -f ".env" ]; then
    print_error ".env file not found. Please run ./quick-start.sh first"
    exit 1
fi

API_KEY=$(grep "INTERNAL_API_KEY=" .env | cut -d '=' -f2)
if [ -z "$API_KEY" ]; then
    print_error "INTERNAL_API_KEY not found in .env"
    exit 1
fi

print_header "🧪 Running System Tests"

PASSED=0
FAILED=0

# ============================================================================
# Test 1: Health Check
# ============================================================================
print_test "Health Check Endpoint"

RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/api/v1/health)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    print_success "Health check passed (HTTP $HTTP_CODE)"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    ((PASSED++))
else
    print_error "Health check failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 2: Metadata Endpoint
# ============================================================================
print_test "Metadata Endpoint"

RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/api/v1/metadata)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    print_success "Metadata endpoint passed (HTTP $HTTP_CODE)"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    ((PASSED++))
else
    print_error "Metadata endpoint failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 3: Agent Health Check
# ============================================================================
print_test "LangGraph Agent Health"

RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/api/v1/langgraph/health/agent)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    print_success "Agent health check passed (HTTP $HTTP_CODE)"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    ((PASSED++))
else
    print_error "Agent health check failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 4: Simple Calculator Query (No RAG)
# ============================================================================
print_test "Simple Calculator Query (25 × 47)"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What is 25 * 47? Just give me the number.",
    "use_rag": false
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    ANSWER=$(echo "$BODY" | jq -r '.answer' 2>/dev/null || echo "")
    if echo "$ANSWER" | grep -q "1175"; then
        print_success "Calculator query passed (HTTP $HTTP_CODE) - Correct answer: 1175"
        echo "Response: $ANSWER"
        ((PASSED++))
    else
        print_error "Calculator query returned wrong answer"
        echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
        ((FAILED++))
    fi
else
    print_error "Calculator query failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 5: Document Upload
# ============================================================================
print_test "Document Upload for RAG"

# Create test document
TEST_DOC="test_document_$(date +%s).txt"
cat > "$TEST_DOC" << EOF
The capital of France is Paris.
Paris is known for the Eiffel Tower.
The Eiffel Tower was built in 1889.
EOF

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$TEST_DOC")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    print_success "Document upload passed (HTTP $HTTP_CODE)"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    ((PASSED++))
else
    print_error "Document upload failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# Clean up test document
rm -f "$TEST_DOC"

# Wait for indexing
echo "Waiting 3 seconds for document to be indexed..."
sleep 3

# ============================================================================
# Test 6: RAG Query
# ============================================================================
print_test "RAG Query (Using Uploaded Document)"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What is the capital of France?",
    "use_rag": true
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    ANSWER=$(echo "$BODY" | jq -r '.answer' 2>/dev/null || echo "")
    if echo "$ANSWER" | grep -iq "paris"; then
        print_success "RAG query passed (HTTP $HTTP_CODE) - Found 'Paris' in answer"
        echo "Answer: $ANSWER"
        ((PASSED++))
    else
        print_error "RAG query didn't find expected answer"
        echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
        ((FAILED++))
    fi
else
    print_error "RAG query failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 7: Session Management
# ============================================================================
print_test "Session Management"

# Create a session
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/api/v1/langgraph/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "session_id": "test-session-'$(date +%s)'",
    "metadata": {"test": true}
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    SESSION_ID=$(echo "$BODY" | jq -r '.session_id' 2>/dev/null || echo "")
    print_success "Session creation passed (HTTP $HTTP_CODE) - Session ID: $SESSION_ID"
    ((PASSED++))

    # List sessions
    RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/api/v1/langgraph/sessions \
      -H "X-API-Key: $API_KEY")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

    if [ "$HTTP_CODE" = "200" ]; then
        print_success "Session listing passed (HTTP $HTTP_CODE)"
        ((PASSED++))
    else
        print_error "Session listing failed (HTTP $HTTP_CODE)"
        ((FAILED++))
    fi
else
    print_error "Session creation failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    ((FAILED++))
fi

# ============================================================================
# Test 8: Authentication Check
# ============================================================================
print_test "API Key Authentication"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: invalid-key" \
  -d '{
    "question": "test",
    "use_rag": false
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "403" ]; then
    print_success "Authentication check passed - Invalid key rejected (HTTP 403)"
    ((PASSED++))
else
    print_error "Authentication check failed - Expected HTTP 403, got $HTTP_CODE"
    ((FAILED++))
fi

# ============================================================================
# Test Results Summary
# ============================================================================
print_header "Test Results Summary"

TOTAL=$((PASSED + FAILED))
echo -e "Total Tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ All tests passed! System is fully functional.${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    exit 0
else
    echo -e "\n${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ Some tests failed. Check logs above for details.${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    echo -e "To view logs: docker compose logs -f api"
    exit 1
fi
