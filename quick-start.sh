#!/bin/bash

# ============================================================================
# Agentic Backend v1.3 - Quick Start Script
# ============================================================================
# This script automates the setup and launch of the entire system
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================
print_header "Step 1: Checking Prerequisites"

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed: $DOCKER_VERSION"
else
    print_error "Docker is not installed"
    echo -e "\nPlease install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
    COMPOSE_CMD="docker-compose"
else
    print_error "Docker Compose is not installed"
    echo -e "\nPlease install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Set compose command
COMPOSE_CMD=${COMPOSE_CMD:-"docker compose"}

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    echo -e "\nPlease start Docker Desktop or the Docker daemon"
    exit 1
fi

print_success "All prerequisites met"

# ============================================================================
# Step 2: Environment Configuration
# ============================================================================
print_header "Step 2: Environment Configuration"

# Check if .env exists
if [ -f ".env" ]; then
    print_warning ".env file already exists"
    read -p "Do you want to reconfigure it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Using existing .env file"
        SKIP_ENV_SETUP=true
    fi
fi

if [ "$SKIP_ENV_SETUP" != "true" ]; then
    # Create .env from template
    cp .env.example .env
    print_success "Created .env file from template"

    # Prompt for OpenAI API Key
    echo -e "\n${YELLOW}Required Configuration:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    read -p "Enter your OpenAI API Key (from https://platform.openai.com): " OPENAI_KEY
    if [ -z "$OPENAI_KEY" ]; then
        print_error "OpenAI API Key is required"
        exit 1
    fi

    # Generate random Internal API Key
    INTERNAL_KEY=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 64 | head -n 1)

    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_KEY|" .env
        sed -i '' "s|INTERNAL_API_KEY=.*|INTERNAL_API_KEY=$INTERNAL_KEY|" .env
    else
        # Linux
        sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_KEY|" .env
        sed -i "s|INTERNAL_API_KEY=.*|INTERNAL_API_KEY=$INTERNAL_KEY|" .env
    fi

    print_success "OpenAI API Key configured"
    print_success "Generated Internal API Key: $INTERNAL_KEY"
    echo -e "\n${YELLOW}Save this Internal API Key - you'll need it to make API requests!${NC}\n"
fi

# ============================================================================
# Step 3: Stop Existing Services
# ============================================================================
print_header "Step 3: Stopping Existing Services"

if $COMPOSE_CMD ps --quiet | grep -q .; then
    print_info "Stopping existing containers..."
    $COMPOSE_CMD down
    print_success "Existing services stopped"
else
    print_info "No existing services to stop"
fi

# ============================================================================
# Step 4: Build and Start Services
# ============================================================================
print_header "Step 4: Building and Starting Services"

print_info "Building Docker images (this may take a few minutes on first run)..."
$COMPOSE_CMD build

print_info "Starting services: Redis, Qdrant, Jaeger, FastAPI..."
$COMPOSE_CMD up -d

print_success "Services started"

# ============================================================================
# Step 5: Wait for Services to be Healthy
# ============================================================================
print_header "Step 5: Waiting for Services to Initialize"

wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $service to be ready"
    while [ $attempt -le $max_attempts ]; do
        if $COMPOSE_CMD ps $service | grep -q "healthy\|running"; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

wait_for_service "redis"
wait_for_service "qdrant"
wait_for_service "jaeger"
wait_for_service "api"

sleep 5  # Additional grace period

# ============================================================================
# Step 6: Health Checks
# ============================================================================
print_header "Step 6: Running Health Checks"

# Check API health
echo -n "Checking FastAPI health endpoint"
MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        API_HEALTHY=true
        break
    fi
    echo -n "."
    sleep 2
    RETRY=$((RETRY + 1))
done

if [ "$API_HEALTHY" != "true" ]; then
    echo -e " ${RED}✗${NC}"
    print_error "API health check failed"
    print_info "Checking API logs..."
    $COMPOSE_CMD logs --tail=50 api
    exit 1
fi

# Check Redis
echo -n "Checking Redis connection"
if $COMPOSE_CMD exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
fi

# Check Qdrant
echo -n "Checking Qdrant health"
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC}"
fi

print_success "All health checks passed!"

# ============================================================================
# Step 7: Display Service Information
# ============================================================================
print_header "🎉 System is Ready!"

# Get the Internal API Key from .env
INTERNAL_API_KEY=$(grep "INTERNAL_API_KEY=" .env | cut -d '=' -f2)

cat << EOF

${GREEN}All services are up and running!${NC}

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BLUE}Service URLs:${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

  📡 API Server:        http://localhost:8000
  📚 API Documentation: http://localhost:8000/docs
  📊 Jaeger Tracing:    http://localhost:16686
  🔍 Qdrant Dashboard:  http://localhost:6333/dashboard
  💾 Redis:             localhost:6379

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BLUE}Quick Test Commands:${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}1. Health Check:${NC}
   curl http://localhost:8000/api/v1/health

${YELLOW}2. Simple Query (Calculator):${NC}
   curl -X POST http://localhost:8000/api/v1/langgraph/query \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: ${INTERNAL_API_KEY}" \\
     -d '{"question": "What is 25 * 47?", "use_rag": false}'

${YELLOW}3. Upload a Document:${NC}
   echo "The Eiffel Tower is in Paris, France." > test.txt
   curl -X POST http://localhost:8000/api/v1/docs/upload \\
     -H "X-API-Key: ${INTERNAL_API_KEY}" \\
     -F "file=@test.txt"

${YELLOW}4. Query with RAG:${NC}
   curl -X POST http://localhost:8000/api/v1/langgraph/query \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: ${INTERNAL_API_KEY}" \\
     -d '{"question": "Where is the Eiffel Tower?", "use_rag": true}'

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BLUE}Your API Key:${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

  ${GREEN}${INTERNAL_API_KEY}${NC}

  ${YELLOW}Save this key! You'll need it for all API requests.${NC}

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BLUE}Useful Commands:${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

  📋 View logs:        $COMPOSE_CMD logs -f api
  🔄 Restart services: $COMPOSE_CMD restart
  🛑 Stop services:    $COMPOSE_CMD down
  🔍 Service status:   $COMPOSE_CMD ps

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}Happy building with Agentic Backend! 🚀${NC}

EOF
