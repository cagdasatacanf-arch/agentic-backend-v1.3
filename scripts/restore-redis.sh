#!/bin/bash
#
# Redis Restore Script
#
# Restores Redis data from backup.
#
# Usage:
#   ./scripts/restore-redis.sh <backup_path>
#
# Examples:
#   ./scripts/restore-redis.sh ./backups/redis/redis_backup_20250101_120000
#   ./scripts/restore-redis.sh ./backups/redis/redis_backup_20250101_120000.tar.gz
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No backup path specified${NC}"
    echo "Usage: $0 <backup_path>"
    echo ""
    echo "Available backups:"
    find ./backups/redis -name "redis_backup_*" 2>/dev/null | sort -r | head -10
    exit 1
fi

BACKUP_PATH="$1"

echo -e "${GREEN}=== Redis Restore ===${NC}"
echo "Backup: $BACKUP_PATH"
echo ""

# Check if backup exists
if [ ! -e "$BACKUP_PATH" ]; then
    echo -e "${RED}Error: Backup not found: $BACKUP_PATH${NC}"
    exit 1
fi

# Extract if compressed
TEMP_DIR=""
if [[ $BACKUP_PATH == *.tar.gz ]]; then
    echo -e "${BLUE}Extracting compressed backup...${NC}"
    TEMP_DIR=$(mktemp -d)
    tar -xzf "$BACKUP_PATH" -C "$TEMP_DIR"
    BACKUP_PATH="$TEMP_DIR/$(basename $BACKUP_PATH .tar.gz)"
    echo -e "${GREEN}✓ Backup extracted to $TEMP_DIR${NC}"
fi

# Verify backup contents
if [ ! -f "$BACKUP_PATH/dump.rdb" ] && [ ! -f "$BACKUP_PATH/appendonly.aof" ]; then
    echo -e "${RED}Error: No valid Redis backup files found${NC}"
    [ -n "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
    exit 1
fi

# Show backup info
if [ -f "$BACKUP_PATH/backup_info.txt" ]; then
    echo -e "${BLUE}Backup Information:${NC}"
    cat "$BACKUP_PATH/backup_info.txt"
    echo ""
fi

# Confirm restore
echo -e "${YELLOW}⚠️  WARNING: This will overwrite current Redis data!${NC}"
echo -n "Type 'yes' to continue: "
read confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled."
    [ -n "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
    exit 0
fi

# Stop Redis container
echo -e "${BLUE}Stopping Redis container...${NC}"
docker stop agentic-redis 2>/dev/null || echo -e "${YELLOW}⚠️  Redis container not running${NC}"

# Backup current data (just in case)
echo -e "${BLUE}Backing up current data...${NC}"
SAFETY_BACKUP="./backups/redis/pre_restore_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAFETY_BACKUP"
docker cp agentic-redis:/data/dump.rdb "$SAFETY_BACKUP/" 2>/dev/null || echo "No current RDB"
docker cp agentic-redis:/data/appendonly.aof "$SAFETY_BACKUP/" 2>/dev/null || echo "No current AOF"
echo -e "${GREEN}✓ Safety backup created at: $SAFETY_BACKUP${NC}"

# Restore RDB file
if [ -f "$BACKUP_PATH/dump.rdb" ]; then
    echo -e "${BLUE}Restoring RDB file...${NC}"
    docker cp "$BACKUP_PATH/dump.rdb" agentic-redis:/data/
    echo -e "${GREEN}✓ RDB file restored${NC}"
fi

# Restore AOF file
if [ -f "$BACKUP_PATH/appendonly.aof" ]; then
    echo -e "${BLUE}Restoring AOF file...${NC}"
    docker cp "$BACKUP_PATH/appendonly.aof" agentic-redis:/data/
    echo -e "${GREEN}✓ AOF file restored${NC}"
fi

# Start Redis container
echo -e "${BLUE}Starting Redis container...${NC}"
docker start agentic-redis

# Wait for Redis to be ready
echo -e "${BLUE}Waiting for Redis to be ready...${NC}"
for i in {1..30}; do
    if docker exec agentic-redis redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}✓ Redis is ready${NC}"
        break
    fi
    sleep 1
done

# Verify restore
echo -e "${BLUE}Verifying restore...${NC}"
echo ""
echo "Database size: $(docker exec agentic-redis redis-cli DBSIZE)"
echo "Memory used: $(docker exec agentic-redis redis-cli INFO MEMORY | grep used_memory_human | cut -d: -f2)"
echo "Uptime: $(docker exec agentic-redis redis-cli INFO SERVER | grep uptime_in_seconds | cut -d: -f2)"

# Cleanup
if [ -n "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

echo ""
echo -e "${GREEN}=== Restore Complete ===${NC}"
echo "Redis has been restored from backup."
echo ""
echo "Safety backup location: $SAFETY_BACKUP"
echo "(You can delete this after verifying the restore)"
echo ""
echo -e "${GREEN}✓ Redis restore completed successfully!${NC}"
