#!/bin/bash
#
# Redis Backup Script
#
# Creates backup of Redis data (RDB + AOF files) with timestamp.
# Supports local and remote backups.
#
# Usage:
#   ./scripts/backup-redis.sh [OPTIONS]
#
# Options:
#   -d, --destination DIR    Backup destination directory (default: ./backups/redis)
#   -r, --remote URL        Remote backup location (s3://bucket, rsync://host/path)
#   -k, --keep DAYS         Keep backups for N days (default: 30)
#   --compress              Compress backup with gzip
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
BACKUP_DIR="./backups/redis"
REMOTE_URL=""
KEEP_DAYS=30
COMPRESS=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--destination)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE_URL="$2"
            shift 2
            ;;
        -k|--keep)
            KEEP_DAYS="$2"
            shift 2
            ;;
        --compress)
            COMPRESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Redis Backup ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Destination: $BACKUP_DIR"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Check if Redis container is running
if ! docker ps | grep -q agentic-redis; then
    echo -e "${RED}Error: Redis container is not running${NC}"
    exit 1
fi

# Trigger Redis save
echo -e "${BLUE}Triggering Redis BGSAVE...${NC}"
docker exec agentic-redis redis-cli BGSAVE

# Wait for save to complete
echo -e "${BLUE}Waiting for save to complete...${NC}"
sleep 2

while docker exec agentic-redis redis-cli LASTSAVE | grep -q "$(date +%s)"; do
    sleep 1
done

echo -e "${GREEN}✓ Redis save completed${NC}"

# Create backup directory for this timestamp
BACKUP_PATH="$BACKUP_DIR/redis_backup_$TIMESTAMP"
mkdir -p "$BACKUP_PATH"

# Copy RDB file
echo -e "${BLUE}Backing up RDB file...${NC}"
docker cp agentic-redis:/data/dump.rdb "$BACKUP_PATH/" 2>/dev/null || echo -e "${YELLOW}⚠️  No RDB file found${NC}"

# Copy AOF file
echo -e "${BLUE}Backing up AOF file...${NC}"
docker cp agentic-redis:/data/appendonly.aof "$BACKUP_PATH/" 2>/dev/null || echo -e "${YELLOW}⚠️  No AOF file found${NC}"

# Copy Redis configuration
echo -e "${BLUE}Backing up Redis configuration...${NC}"
if [ -f ./redis.conf ]; then
    cp ./redis.conf "$BACKUP_PATH/"
fi

# Create backup metadata
cat > "$BACKUP_PATH/backup_info.txt" << EOF
Redis Backup Information
========================
Timestamp: $TIMESTAMP
Date: $(date)
Hostname: $(hostname)
Redis Version: $(docker exec agentic-redis redis-cli INFO SERVER | grep redis_version | cut -d: -f2)
Database Size: $(docker exec agentic-redis redis-cli DBSIZE)
Memory Used: $(docker exec agentic-redis redis-cli INFO MEMORY | grep used_memory_human | cut -d: -f2)
AOF Enabled: $(docker exec agentic-redis redis-cli CONFIG GET appendonly | tail -1)
EOF

# Compress if requested
if [ "$COMPRESS" = true ]; then
    echo -e "${BLUE}Compressing backup...${NC}"
    tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "$(basename $BACKUP_PATH)"
    rm -rf "$BACKUP_PATH"
    BACKUP_PATH="${BACKUP_PATH}.tar.gz"
    echo -e "${GREEN}✓ Backup compressed${NC}"
fi

echo -e "${GREEN}✓ Backup created: $BACKUP_PATH${NC}"

# Upload to remote if specified
if [ -n "$REMOTE_URL" ]; then
    echo -e "${BLUE}Uploading to remote: $REMOTE_URL${NC}"

    if [[ $REMOTE_URL == s3://* ]]; then
        # S3 upload
        if command -v aws &> /dev/null; then
            aws s3 cp "$BACKUP_PATH" "$REMOTE_URL/"
            echo -e "${GREEN}✓ Uploaded to S3${NC}"
        else
            echo -e "${RED}Error: AWS CLI not installed${NC}"
        fi
    elif [[ $REMOTE_URL == rsync://* ]]; then
        # Rsync upload
        rsync -avz "$BACKUP_PATH" "$REMOTE_URL/"
        echo -e "${GREEN}✓ Uploaded via rsync${NC}"
    else
        echo -e "${YELLOW}⚠️  Unknown remote URL format${NC}"
    fi
fi

# Clean up old backups
echo -e "${BLUE}Cleaning up old backups (older than $KEEP_DAYS days)...${NC}"
find "$BACKUP_DIR" -name "redis_backup_*" -type f -mtime +$KEEP_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "redis_backup_*" -type d -mtime +$KEEP_DAYS -exec rm -rf {} + 2>/dev/null || true

REMAINING=$(find "$BACKUP_DIR" -name "redis_backup_*" | wc -l)
echo -e "${GREEN}✓ Cleanup complete. $REMAINING backups remaining${NC}"

# Calculate backup size
if [ -f "$BACKUP_PATH" ]; then
    SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
else
    SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
fi

echo ""
echo -e "${GREEN}=== Backup Summary ===${NC}"
echo "Backup path: $BACKUP_PATH"
echo "Backup size: $SIZE"
echo "Retention: $KEEP_DAYS days"
echo "Backups total: $REMAINING"
echo ""
echo -e "${GREEN}✓ Redis backup completed successfully!${NC}"
