#!/bin/bash
#
# Docker Secrets Management Script
#
# Manages Docker Swarm secrets for production deployment
#
# Usage:
#   ./scripts/manage-secrets.sh create    # Create all secrets
#   ./scripts/manage-secrets.sh list      # List all secrets
#   ./scripts/manage-secrets.sh rotate    # Rotate all secrets
#   ./scripts/manage-secrets.sh delete    # Delete all secrets
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running in Docker Swarm mode
check_swarm() {
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        echo -e "${YELLOW}⚠️  Docker Swarm is not active${NC}"
        echo "Initialize with: docker swarm init"
        echo ""
        echo "For development without Swarm, use bind-mounted secret files instead."
        exit 1
    fi
}

# Create a secret from user input
create_secret() {
    local name=$1
    local description=$2

    echo -e "${BLUE}Creating secret: ${name}${NC}"
    echo "Description: ${description}"

    # Check if secret already exists
    if docker secret ls --format '{{.Name}}' | grep -q "^${name}$"; then
        echo -e "${YELLOW}Secret ${name} already exists. Skipping.${NC}"
        return
    fi

    # Prompt for secret value
    echo -n "Enter value (or press Enter to generate): "
    read -s secret_value
    echo ""

    # Generate random value if empty
    if [ -z "$secret_value" ]; then
        if [ "$name" = "openai_api_key" ]; then
            echo -e "${RED}OpenAI API key cannot be auto-generated!${NC}"
            echo "Get your API key from: https://platform.openai.com/api-keys"
            return 1
        fi
        secret_value=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo -e "${GREEN}Generated random value${NC}"
    fi

    # Create secret
    echo "$secret_value" | docker secret create "$name" - >/dev/null
    echo -e "${GREEN}✓ Secret ${name} created${NC}"
    echo ""
}

# Create all required secrets
create_all_secrets() {
    echo -e "${GREEN}=== Creating Docker Secrets ===${NC}"
    echo ""

    check_swarm

    create_secret "openai_api_key" "OpenAI API key from https://platform.openai.com/api-keys"
    create_secret "redis_password" "Redis authentication password"
    create_secret "qdrant_api_key" "Qdrant vector database API key (optional)"
    create_secret "grafana_admin_password" "Grafana admin dashboard password"
    create_secret "app_secret_key" "Application secret key for signing tokens"

    echo -e "${GREEN}=== Secrets created successfully! ===${NC}"
    echo ""
    echo "View secrets with: docker secret ls"
    echo "Deploy with: docker stack deploy -c docker-compose.yml -c docker-compose.secrets.yml agentic"
}

# List all secrets
list_secrets() {
    check_swarm

    echo -e "${GREEN}=== Docker Secrets ===${NC}"
    docker secret ls
}

# Rotate a secret
rotate_secret() {
    local name=$1

    echo -e "${BLUE}Rotating secret: ${name}${NC}"

    # Remove old secret
    if docker secret ls --format '{{.Name}}' | grep -q "^${name}$"; then
        echo "Removing old secret..."
        docker secret rm "$name" >/dev/null || true
    fi

    # Create new secret
    echo -n "Enter new value (or press Enter to generate): "
    read -s secret_value
    echo ""

    if [ -z "$secret_value" ]; then
        secret_value=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo -e "${GREEN}Generated random value${NC}"
    fi

    echo "$secret_value" | docker secret create "$name" - >/dev/null
    echo -e "${GREEN}✓ Secret ${name} rotated${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Update your stack to use the new secret:${NC}"
    echo "docker stack deploy -c docker-compose.yml -c docker-compose.secrets.yml agentic"
}

# Rotate all secrets
rotate_all_secrets() {
    echo -e "${YELLOW}⚠️  This will rotate ALL secrets!${NC}"
    echo -n "Are you sure? (yes/no): "
    read confirm

    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi

    check_swarm

    rotate_secret "openai_api_key"
    rotate_secret "redis_password"
    rotate_secret "qdrant_api_key"
    rotate_secret "grafana_admin_password"
    rotate_secret "app_secret_key"

    echo -e "${GREEN}=== All secrets rotated ===${NC}"
}

# Delete all secrets
delete_all_secrets() {
    echo -e "${RED}⚠️  This will DELETE ALL secrets!${NC}"
    echo -n "Are you sure? (yes/no): "
    read confirm

    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi

    check_swarm

    echo "Deleting secrets..."
    docker secret rm openai_api_key 2>/dev/null || true
    docker secret rm redis_password 2>/dev/null || true
    docker secret rm qdrant_api_key 2>/dev/null || true
    docker secret rm grafana_admin_password 2>/dev/null || true
    docker secret rm app_secret_key 2>/dev/null || true

    echo -e "${GREEN}✓ Secrets deleted${NC}"
}

# Create development secret files (without Docker Swarm)
create_dev_secrets() {
    echo -e "${GREEN}=== Creating Development Secret Files ===${NC}"
    echo ""
    echo "This creates secret files for development without Docker Swarm."
    echo ""

    mkdir -p ./secrets

    echo -n "Enter OpenAI API key: "
    read -s openai_key
    echo ""
    echo "$openai_key" > ./secrets/openai_api_key

    # Generate random values for other secrets
    python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ./secrets/redis_password
    python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ./secrets/app_secret_key
    python3 -c "import secrets; print('admin' + secrets.token_urlsafe(16))" > ./secrets/grafana_admin_password

    chmod 600 ./secrets/*

    echo -e "${GREEN}✓ Development secrets created in ./secrets/${NC}"
    echo ""
    echo "Use these in docker-compose with:"
    echo "  environment:"
    echo "    - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key"
    echo "  volumes:"
    echo "    - ./secrets:/run/secrets:ro"
}

# Main command handler
case "${1:-}" in
    create)
        create_all_secrets
        ;;
    list)
        list_secrets
        ;;
    rotate)
        rotate_all_secrets
        ;;
    delete)
        delete_all_secrets
        ;;
    dev)
        create_dev_secrets
        ;;
    *)
        echo "Docker Secrets Management"
        echo ""
        echo "Usage:"
        echo "  $0 create    # Create all secrets (requires Docker Swarm)"
        echo "  $0 list      # List all secrets"
        echo "  $0 rotate    # Rotate all secrets"
        echo "  $0 delete    # Delete all secrets"
        echo "  $0 dev       # Create development secret files (no Swarm needed)"
        echo ""
        exit 1
        ;;
esac
