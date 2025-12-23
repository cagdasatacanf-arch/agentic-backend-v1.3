# 🚨 ERROR GUIDELINES & TROUBLESHOOTING DOCUMENT

Complete guide to common errors, how to fix them, and how to prevent them.

---

## TABLE OF CONTENTS

1. [Docker Build Errors](#docker-build-errors)
2. [Docker Runtime Errors](#docker-runtime-errors)
3. [Application Errors](#application-errors)
4. [Configuration Errors](#configuration-errors)
5. [Network & Connection Errors](#network--connection-errors)
6. [Performance Issues](#performance-issues)
7. [Quick Diagnostic Commands](#quick-diagnostic-commands)
8. [Prevention Checklist](#prevention-checklist)

---

## DOCKER BUILD ERRORS

### Error 1: "No such file or directory: 'requirements.txt'"

**Problem:** Docker cannot find `requirements.txt` during build

**Cause:**
- File doesn't exist
- Wrong filename (e.g., `requirement.txt` instead of `requirements.txt`)
- Running docker build from wrong directory

**Fix:**
```bash
# Check file exists
ls -la requirements.txt

# Make sure you're in project root
pwd
# Should show: /path/to/agentic-backend

# Rebuild
docker compose build
```

**Prevention:**
- Verify file exists before building
- Always build from project root directory

---

### Error 2: "failed to solve: rpc error: code = Unknown desc = error processing tar file"

**Problem:** Build context has corrupted or too-large files

**Cause:**
- Large `.git` directory
- Unneeded node_modules, venv, or cache
- File permissions issue

**Fix:**
```bash
# Rebuild without cache
docker compose build --no-cache

# Or clean everything
docker system prune -a
docker compose build
```

**Prevention:**
- Use proper `.dockerignore` (provided)
- Don't include `venv/`, `.git/`, cache in Docker context

---

### Error 3: "ERROR: failed to create LLM client: API key not found"

**Problem:** `OPENAI_API_KEY` not passed to build

**Cause:**
- Key not in `.env`
- `.env` file doesn't exist
- Environment variable not set

**Fix:**
```bash
# Create .env
cp .env.example .env

# Edit and add your key
nano .env
# Find: OPENAI_API_KEY=sk-your-openai-key-here
# Replace with actual key

# Rebuild
docker compose build
```

**Prevention:**
- Always create `.env` before building
- Never commit `.env` to git (it's in `.gitignore`)

---

## DOCKER RUNTIME ERRORS

### Error 4: "Port 8000 is already allocated"

**Problem:** Port 8000 is in use by another service

**Cause:**
- Another container using port 8000
- Local service using port 8000

**Fix:**
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in docker-compose.yml:
# ports:
#   - "8001:8000"  # Change to 8001

# Restart
docker compose down
docker compose up -d
```

**Prevention:**
- Stop other services before starting Docker Compose
- Check port availability first: `lsof -i :8000`

---

### Error 5: "service is unhealthy"

**Problem:** Health check is failing

**Cause:**
- Service crashed or not responding
- API not starting properly
- Missing dependencies

**Fix:**
```bash
# Check logs
docker compose logs app

# Look for error messages
docker compose logs redis
docker compose logs qdrant

# Restart service
docker compose restart app

# If still failing
docker compose down
docker compose up --build
```

**Prevention:**
- Check logs before errors get worse
- Verify all environment variables are set

---

### Error 6: "connect: connection refused"

**Problem:** Can't connect to service (app, redis, qdrant)

**Cause:**
- Service not started
- Service crashed
- Wrong port/hostname

**Fix:**
```bash
# Check service status
docker compose ps

# Check service is running
docker compose logs <service-name>

# Restart service
docker compose restart <service-name>

# Test connection
docker compose exec app curl http://redis:6379
docker compose exec app curl http://qdrant:6333
```

**Prevention:**
- Always check `docker compose ps` first
- Wait 30+ seconds for services to fully start

---

### Error 7: "Out of memory"

**Problem:** Container killed due to memory limit

**Cause:**
- Memory limit set too low
- Memory leak in application
- Large dataset processing

**Fix:**
```bash
# Check memory usage
docker stats

# Increase limit in docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 4G    # Increase from 2G

docker compose down
docker compose up -d
```

**Prevention:**
- Monitor memory usage: `docker stats`
- Start with reasonable limits, adjust as needed

---

## APPLICATION ERRORS

### Error 8: "ModuleNotFoundError: No module named 'app'"

**Problem:** Python can't find the `app` module

**Cause:**
- `app/` directory doesn't exist
- Working directory is wrong in Dockerfile
- `PYTHONPATH` not set

**Fix:**
```bash
# Verify structure
ls -la app/
# Should show: __init__.py, main.py, etc.

# Check Dockerfile has correct PYTHONPATH
# Should have: ENV PYTHONPATH="/app"

# Rebuild
docker compose build
```

**Prevention:**
- Ensure `app/` directory structure is correct
- Check Dockerfile `PYTHONPATH` and `WORKDIR`

---

### Error 9: "OPENAI_API_KEY error: Invalid API key"

**Problem:** OpenAI API key is invalid or expired

**Cause:**
- Wrong API key format
- Expired key
- Key not properly set

**Fix:**
```bash
# Check key format (should start with sk-)
echo $OPENAI_API_KEY

# Get new key from https://platform.openai.com/api-keys
# Update .env
nano .env

# Restart app
docker compose restart app
```

**Prevention:**
- Use correct key format
- Keep key secret, never commit to git
- Check key hasn't expired

---

### Error 10: "Uvicorn server failed to start"

**Problem:** Uvicorn worker can't start under Gunicorn

**Cause:**
- Syntax error in app code
- Import error
- Missing dependency

**Fix:**
```bash
# Check logs for details
docker compose logs app -f

# Look for traceback/error message
# Fix code issue
# Rebuild if changed requirements
docker compose restart app
```

**Prevention:**
- Test app locally first: `uvicorn app.main:app --reload`
- Run tests: `pytest`

---

## CONFIGURATION ERRORS

### Error 11: "VECTOR_DB_URL connection refused"

**Problem:** Can't connect to Qdrant

**Cause:**
- Wrong URL (should be `http://qdrant:6333` in Docker)
- Qdrant service not started
- Network issue

**Fix:**
```bash
# Check Qdrant is running
docker compose ps qdrant

# Check it's healthy
docker compose logs qdrant

# Verify URL in .env
grep VECTOR_DB_URL .env
# Should be: VECTOR_DB_URL=http://qdrant:6333

# Restart
docker compose restart qdrant
```

**Prevention:**
- Use provided `.env.example` as template
- Check service names match in docker-compose.yml

---

### Error 12: "REDIS_URL connection refused"

**Problem:** Can't connect to Redis

**Cause:**
- Wrong URL (should be `redis://redis:6379` in Docker)
- Redis service not started
- Network issue

**Fix:**
```bash
# Check Redis is running
docker compose ps redis

# Verify URL in .env
grep REDIS_URL .env
# Should be: REDIS_URL=redis://redis:6379

# Test connection
docker compose exec app redis-cli -h redis ping
# Should return: PONG

# Restart
docker compose restart redis
```

**Prevention:**
- Use provided `.env.example`
- Always test after configuration changes

---

### Error 13: ".env file not found"

**Problem:** Application can't find `.env` file

**Cause:**
- `.env` doesn't exist
- `.env` in wrong location
- File is named wrong

**Fix:**
```bash
# Create .env
cp .env.example .env

# Verify it exists
ls -la .env

# Make sure it's in project root
pwd
# Should show: /path/to/agentic-backend
```

**Prevention:**
- Always create `.env` from `.env.example`
- Keep `.env` in project root, same as Dockerfile

---

## NETWORK & CONNECTION ERRORS

### Error 14: "Unable to connect to Docker daemon"

**Problem:** Docker daemon not running

**Cause:**
- Docker not installed
- Docker not started
- Wrong Docker socket permissions

**Fix:**
```bash
# Start Docker daemon
# macOS: open Docker.app
# Linux: sudo systemctl start docker
# Windows: Start Docker Desktop

# Verify Docker is running
docker ps

# Check permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

**Prevention:**
- Ensure Docker is running before building
- Check Docker status: `docker ps`

---

### Error 15: "Name or service not known" (DNS error)

**Problem:** Service name not resolving in Docker network

**Cause:**
- Service not on same network
- Network not created
- Service name misspelled

**Fix:**
```bash
# Check networks
docker network ls
docker network inspect agentic-network

# Verify service names match in docker-compose.yml
# All should have: networks: - agentic-network

docker compose down
docker compose up --build
```

**Prevention:**
- Use provided docker-compose.yml
- Check all services have correct network assignment

---

### Error 16: "Hostname resolution failed"

**Problem:** Can't resolve service hostname from host machine

**Cause:**
- Trying to use `http://qdrant:6333` from host (wrong)
- Should use `http://localhost:6333` from host

**Fix:**
```bash
# From INSIDE container (correct)
docker compose exec app curl http://qdrant:6333

# From HOST machine (correct)
curl http://localhost:6333

# From HOST machine (WRONG - don't do this)
# curl http://qdrant:6333    # Won't work!
```

**Prevention:**
- Use `localhost` when accessing from host machine
- Use service name when accessing from inside container

---

## PERFORMANCE ISSUES

### Issue 17: "Slow API responses"

**Problem:** API taking too long to respond

**Cause:**
- Only 1 worker running
- High memory usage causing swapping
- Database queries too slow

**Fix:**
```bash
# Check worker count in Dockerfile
# Should have: --workers 4

# Check memory usage
docker stats

# Check CPU usage
docker stats

# Increase workers if CPU is low
# Edit Dockerfile: --workers 8 (from 4)
docker compose build
docker compose up -d
```

**Prevention:**
- Use 4+ workers for production
- Monitor with `docker stats`

---

### Issue 18: "High memory usage"

**Problem:** Container using too much memory

**Cause:**
- Memory limit too high
- Memory leak in code
- Too many workers

**Fix:**
```bash
# Check memory usage
docker stats

# Reduce workers if needed
# Edit Dockerfile: --workers 2 (from 4)

# Reduce timeout if safe
# Edit Dockerfile: --timeout 30 (from 60)

# Restart
docker compose build
docker compose up -d
```

**Prevention:**
- Monitor regularly: `docker stats`
- Set reasonable limits in docker-compose.yml
- Profile code for memory leaks

---

### Issue 19: "Disk space full"

**Problem:** Docker images/containers using all disk space

**Cause:**
- Many image layers
- Log files growing unbounded
- Old containers not removed

**Fix:**
```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a

# Clean up volumes
docker volume prune

# Remove old logs
docker compose logs --tail=0 > /dev/null
```

**Prevention:**
- Log rotation configured in docker-compose.yml
- Regularly clean up: `docker system prune`

---

## QUICK DIAGNOSTIC COMMANDS

Use these to diagnose issues quickly:

```bash
# 1. Check all services running
docker compose ps

# 2. Check app logs
docker compose logs app -f

# 3. Check all logs
docker compose logs

# 4. Health check
curl http://localhost:8000/api/v1/health

# 5. Check memory/CPU
docker stats

# 6. Test Redis connection
docker compose exec app redis-cli -h redis ping

# 7. Test Qdrant connection
curl http://localhost:6333/health

# 8. Test from inside app container
docker compose exec app bash
curl http://qdrant:6333/health
exit

# 9. Check environment variables
docker compose exec app env | grep OPENAI

# 10. Network info
docker network inspect agentic-network

# 11. Full system diagnostics
docker system df
docker system events

# 12. Rebuild from scratch
docker compose down -v
docker system prune -a
docker compose build --no-cache
docker compose up -d
```

---

## PREVENTION CHECKLIST

### Before Deploying

- [ ] All 6 critical files present (Dockerfile, docker-compose.yml, requirements.txt, .dockerignore, .env.example, .gitignore)
- [ ] `.env` created from `.env.example`
- [ ] `OPENAI_API_KEY` set in `.env`
- [ ] Docker installed and running
- [ ] Port 8000 not in use
- [ ] `.env` NOT in git
- [ ] Dockerfile tested locally

### During First Run

- [ ] `docker compose build` completes without errors
- [ ] `docker compose up -d` starts all services
- [ ] Wait 30 seconds for startup
- [ ] `docker compose ps` shows all services as "Up"
- [ ] Health check passes: `curl http://localhost:8000/api/v1/health`
- [ ] App logs show no errors: `docker compose logs app`

### Before Production

- [ ] All services healthy
- [ ] Health checks passing
- [ ] Memory usage reasonable (< 70% of limit)
- [ ] CPU usage reasonable
- [ ] No error messages in logs
- [ ] API responds correctly
- [ ] Database connections working
- [ ] Monitoring/tracing working (Jaeger)

### Ongoing Maintenance

- [ ] Monitor logs daily
- [ ] Check `docker stats` weekly
- [ ] Clean up with `docker system prune` monthly
- [ ] Update dependencies quarterly
- [ ] Backup important data
- [ ] Test rollback procedures

---

## EMERGENCY PROCEDURES

### Complete Restart (Nuclear Option)

```bash
# Stop everything
docker compose down

# Remove all data
docker volume prune -a
docker system prune -a

# Rebuild from scratch
docker compose build --no-cache

# Start fresh
docker compose up -d

# Verify
docker compose ps
curl http://localhost:8000/api/v1/health
```

### Rollback to Previous State

```bash
# If you have a backup:
docker compose down
# Restore data from backup
docker compose up -d
```

### Emergency Logs Export

```bash
# Save all logs to file
docker compose logs > logs-backup-$(date +%Y%m%d-%H%M%S).txt

# Export logs from container
docker logs <container-id> > container-logs.txt
```

---

## ERROR CODES QUICK REFERENCE

| Code | Meaning | Check |
|------|---------|-------|
| 1 | General error | Logs for details |
| 2 | Misuse of command | Syntax/arguments |
| 127 | Command not found | Path/binary missing |
| 137 | OOM killed | Memory limit too low |
| 139 | Segmentation fault | Memory corruption |
| 143 | Terminated signal | Service was killed |

---

## SUPPORT & HELP

### If Something Goes Wrong

1. **Check logs first:** `docker compose logs`
2. **Search this document** for the error message
3. **Try diagnostic commands** from "Quick Diagnostic Commands"
4. **Restart services:** `docker compose restart`
5. **Last resort:** Full rebuild (see "Emergency Procedures")

### Getting Help

- Error message not listed? Check:
  - `docker compose logs -f app` (current errors)
  - `docker system df` (disk/space issues)
  - `docker stats` (memory/CPU issues)

### Reporting Issues

Include when reporting:
- Full error message
- Output of `docker compose ps`
- Output of `docker compose logs app --tail=50`
- Output of `docker --version`
- OS/platform you're using

---

**Version:** 1.0.0  
**Last Updated:** 2025-12-16  
**Status:** Complete Error Guidelines ✅
