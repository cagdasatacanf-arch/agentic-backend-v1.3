# Backup and Disaster Recovery Strategy

Comprehensive backup and disaster recovery plan for the Agentic Backend.

## Overview

This system implements a multi-layered backup strategy:

1. **Redis Data** - All application state, sessions, caching, and RBAC data
2. **Qdrant Data** - Vector embeddings and document storage
3. **Application Configuration** - Environment variables, secrets, and configs
4. **Prometheus Metrics** - Historical performance data
5. **Grafana Dashboards** - Custom visualization configurations

## Redis Backup Strategy

### Persistence Configuration

Redis uses **dual persistence** for maximum reliability:

**RDB (Snapshots):**
- Full database snapshots at intervals
- Fast restores
- Compact file size
- Configuration: `save 900 1`, `save 300 10`, `save 60 10000`

**AOF (Append-Only File):**
- Logs every write operation
- Better durability (fsync every second)
- Automatic rewrite to optimize size
- Hybrid RDB+AOF format for fast loading

**Configuration:** See `redis.conf` for full settings.

### Automated Backups

#### Daily Backups

```bash
# Add to crontab
0 2 * * * /path/to/agentic-backend/scripts/backup-redis.sh --compress --keep 30
```

This runs at 2 AM daily, compresses backups, and keeps 30 days of history.

#### Hourly Backups (High-Frequency)

```bash
# For critical production systems
0 * * * * /path/to/agentic-backend/scripts/backup-redis.sh --destination /backups/hourly --keep 7
```

Keep hourly backups for 7 days.

### Manual Backup

```bash
# Basic backup
./scripts/backup-redis.sh

# Compressed backup with 90-day retention
./scripts/backup-redis.sh --compress --keep 90

# Backup to custom location
./scripts/backup-redis.sh --destination /mnt/backup-drive/redis

# Backup to S3
./scripts/backup-redis.sh --remote s3://my-bucket/redis-backups --compress
```

### Restore from Backup

```bash
# List available backups
ls -lh ./backups/redis/

# Restore from specific backup
./scripts/restore-redis.sh ./backups/redis/redis_backup_20250124_020000

# Restore from compressed backup
./scripts/restore-redis.sh ./backups/redis/redis_backup_20250124_020000.tar.gz
```

**Important:** The restore script:
1. Creates a safety backup of current data
2. Stops Redis
3. Replaces data files
4. Restarts Redis
5. Verifies the restore

## Qdrant Backup Strategy

### Snapshot Backups

Qdrant provides built-in snapshot functionality:

```bash
# Create snapshot via API
curl -X POST http://localhost:6333/collections/documents/snapshots

# Download snapshot
curl http://localhost:6333/collections/documents/snapshots/snapshot_name \
  -o qdrant_backup.snapshot

# Restore snapshot
curl -X PUT http://localhost:6333/collections/documents/snapshots/upload \
  -F 'file=@qdrant_backup.snapshot'
```

### Automated Qdrant Backups

```bash
# Create backup script
cat > scripts/backup-qdrant.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="./backups/qdrant"

mkdir -p "$BACKUP_DIR"

# Create snapshot
SNAPSHOT=$(curl -s -X POST http://localhost:6333/collections/documents/snapshots | jq -r '.result.name')

# Download snapshot
curl -s "http://localhost:6333/collections/documents/snapshots/$SNAPSHOT" \
  -o "$BACKUP_DIR/qdrant_backup_$TIMESTAMP.snapshot"

echo "Qdrant backup created: $BACKUP_DIR/qdrant_backup_$TIMESTAMP.snapshot"
EOF

chmod +x scripts/backup-qdrant.sh
```

### Volume Backups

Alternative method - backup the Docker volume:

```bash
# Stop Qdrant
docker stop agentic-qdrant

# Backup volume
docker run --rm \
  -v agentic-backend-v13_qdrant_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant_volume_$(date +%Y%m%d).tar.gz /data

# Start Qdrant
docker start agentic-qdrant
```

## Configuration Backup

### Environment Variables

```bash
# Backup .env file (NEVER commit to git!)
cp .env backups/config/env_$(date +%Y%m%d).backup

# Encrypt for secure storage
openssl enc -aes-256-cbc -salt -in .env -out backups/config/env_encrypted_$(date +%Y%m%d).enc
```

### Docker Configurations

```bash
# Backup all configuration files
tar czf backups/config/configs_$(date +%Y%m%d).tar.gz \
  docker-compose*.yml \
  nginx.conf \
  redis.conf \
  monitoring/
```

## Disaster Recovery Procedures

### Scenario 1: Redis Data Corruption

**Symptoms:**
- Redis won't start
- Error messages about corrupted AOF/RDB
- Data loss detected

**Recovery:**

```bash
# 1. Stop Redis
docker stop agentic-redis

# 2. Check AOF integrity
docker run --rm -v agentic-backend-v13_redis_data:/data redis:7-alpine \
  redis-check-aof /data/appendonly.aof

# 3. If corrupted, try to fix
docker run --rm -v agentic-backend-v13_redis_data:/data redis:7-alpine \
  redis-check-aof --fix /data/appendonly.aof

# 4. If fix fails, restore from backup
./scripts/restore-redis.sh ./backups/redis/redis_backup_latest

# 5. Verify
docker logs agentic-redis
```

### Scenario 2: Complete Data Loss

**Symptoms:**
- All volumes deleted
- Catastrophic failure
- Need to rebuild from scratch

**Recovery:**

```bash
# 1. Restore Redis
./scripts/restore-redis.sh ./backups/redis/redis_backup_20250124_020000

# 2. Restore Qdrant
docker stop agentic-qdrant
docker run --rm \
  -v agentic-backend-v13_qdrant_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/qdrant_volume_20250124.tar.gz -C /
docker start agentic-qdrant

# 3. Restore configurations
cp backups/config/env_20250124.backup .env

# 4. Restart all services
docker-compose down
docker-compose up -d

# 5. Verify all services
docker-compose ps
```

### Scenario 3: Accidental Data Deletion

**Symptoms:**
- User accidentally deleted important data
- Need to recover specific keys/collections

**Recovery:**

```bash
# Redis: Restore to temporary instance
docker run --name redis-temp -d -p 6380:6379 \
  -v ./backups/redis/redis_backup_20250124:/data \
  redis:7-alpine

# Access temporary instance
redis-cli -p 6380

# Export specific keys
redis-cli -p 6380 --scan --pattern "user:*" | \
  xargs redis-cli -p 6380 DUMP > keys_to_restore.txt

# Import to production (requires custom script)
```

## Backup Testing

### Monthly Restore Test

```bash
# 1. Create test environment
docker-compose -f docker-compose.test.yml up -d

# 2. Restore latest backup to test environment
./scripts/restore-redis.sh --target test ./backups/redis/redis_backup_latest

# 3. Verify data integrity
docker exec redis-test redis-cli DBSIZE
docker exec redis-test redis-cli GET test_key

# 4. Cleanup
docker-compose -f docker-compose.test.yml down
```

### Backup Verification Script

```bash
#!/bin/bash
# scripts/verify-backups.sh

BACKUP_DIR="./backups/redis"
LATEST=$(ls -t $BACKUP_DIR/redis_backup_* | head -1)

echo "Verifying latest backup: $LATEST"

# Check RDB integrity
if [ -f "$LATEST/dump.rdb" ]; then
  docker run --rm -v "$LATEST":/data redis:7-alpine redis-check-rdb /data/dump.rdb
fi

# Check AOF integrity
if [ -f "$LATEST/appendonly.aof" ]; then
  docker run --rm -v "$LATEST":/data redis:7-alpine redis-check-aof /data/appendonly.aof
fi

echo "Backup verification complete!"
```

## Backup Retention Policy

| Backup Type | Frequency | Retention | Location |
|-------------|-----------|-----------|----------|
| Hourly | Every hour | 7 days | Local disk |
| Daily | Daily at 2 AM | 30 days | Local + S3 |
| Weekly | Sunday 2 AM | 90 days | S3 |
| Monthly | 1st of month | 1 year | S3 Glacier |

## Off-Site Backups

### S3 Configuration

```bash
# Configure AWS CLI
aws configure

# Test upload
./scripts/backup-redis.sh --remote s3://my-bucket/backups --compress

# Verify
aws s3 ls s3://my-bucket/backups/
```

### Automated S3 Sync

```bash
# Add to crontab for daily S3 sync
0 3 * * * /path/to/scripts/backup-redis.sh --remote s3://my-bucket/redis --compress --keep 30
```

### S3 Lifecycle Policies

```json
{
  "Rules": [
    {
      "Id": "ArchiveOldBackups",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    }
  ]
}
```

## Monitoring Backups

### Backup Health Checks

```bash
# Check last backup age
LAST_BACKUP=$(ls -t ./backups/redis/redis_backup_* | head -1)
BACKUP_AGE=$(( ($(date +%s) - $(stat -f %m "$LAST_BACKUP")) / 3600 ))

if [ $BACKUP_AGE -gt 24 ]; then
  echo "WARNING: Last backup is $BACKUP_AGE hours old!"
fi
```

### Prometheus Metrics

Add custom metrics for backup monitoring:

```python
backup_last_success = Gauge('backup_last_success_timestamp', 'Last successful backup')
backup_size_bytes = Gauge('backup_size_bytes', 'Backup size in bytes')
```

### Alerts

Set up alerts for backup failures:

```yaml
# prometheus/alerts.yml
groups:
  - name: backup
    rules:
      - alert: BackupTooOld
        expr: time() - backup_last_success_timestamp > 86400
        annotations:
          summary: "Backup is older than 24 hours"
```

## Best Practices

1. **Test Restores Regularly** - Backups are useless if you can't restore
2. **3-2-1 Rule** - 3 copies, 2 different media, 1 off-site
3. **Automate Everything** - Manual backups will be forgotten
4. **Monitor Backup Health** - Set up alerts for backup failures
5. **Document Procedures** - Team should know how to restore
6. **Encrypt Sensitive Backups** - Especially for off-site storage
7. **Version Control Configs** - Track changes to configuration files

## Troubleshooting

### Backup Script Fails

```bash
# Check Docker access
docker ps

# Check disk space
df -h

# Check Redis status
docker exec agentic-redis redis-cli ping

# Run backup with verbose output
bash -x ./scripts/backup-redis.sh
```

### Restore Fails

```bash
# Check backup integrity
tar -tzf backup.tar.gz

# Check permissions
ls -la ./backups/redis/

# Manual restore
docker stop agentic-redis
docker cp backup/dump.rdb agentic-redis:/data/
docker start agentic-redis
```

## Recovery Time Objectives (RTO)

| Scenario | Target RTO | Procedure |
|----------|------------|-----------|
| Redis restart | < 1 minute | docker restart |
| Redis restore from local backup | < 5 minutes | Run restore script |
| Full system restore | < 30 minutes | Restore all volumes |
| Restore from S3 | < 60 minutes | Download + restore |

## Recovery Point Objectives (RPO)

| Data Type | Target RPO | Method |
|-----------|------------|--------|
| User sessions | 1 hour | Hourly Redis backup |
| RBAC data | 1 day | Daily Redis backup |
| Vector embeddings | 1 day | Daily Qdrant backup |
| Metrics data | 7 days | Prometheus retention |

## Further Reading

- [Redis Persistence](https://redis.io/docs/management/persistence/)
- [Qdrant Snapshots](https://qdrant.tech/documentation/concepts/snapshots/)
- [Docker Volume Backups](https://docs.docker.com/storage/volumes/#back-up-restore-or-migrate-data-volumes)
