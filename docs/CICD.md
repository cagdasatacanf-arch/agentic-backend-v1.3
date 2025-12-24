# CI/CD Pipeline Documentation

Comprehensive CI/CD pipeline for the Agentic Backend using GitHub Actions.

## Overview

The CI/CD pipeline implements automated testing, security scanning, and deployment with zero-downtime rolling updates.

### Workflows

1. **Continuous Integration (CI)** - `.github/workflows/ci.yml`
2. **Continuous Deployment (CD)** - `.github/workflows/cd.yml`
3. **Scheduled Tasks** - `.github/workflows/scheduled.yml`

## Continuous Integration (CI)

Triggered on every push and pull request to `main` and `develop` branches.

### Jobs

#### 1. Lint (`lint`)

Code quality checks:
- **Black**: Code formatting (PEP 8)
- **isort**: Import sorting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking

```yaml
runs-on: ubuntu-latest
duration: ~2-3 minutes
```

#### 2. Test (`test`)

Runs all tests with coverage:
- Unit tests (`tests/unit/`)
- Integration tests (`tests/integration/`)
- Coverage reporting (uploaded to Codecov)

**Services:**
- Redis (port 6379)
- Qdrant (port 6333)

```yaml
runs-on: ubuntu-latest
duration: ~5-10 minutes
```

#### 3. Security (`security`)

Security vulnerability scanning:
- **Safety**: Check for known vulnerabilities in dependencies
- **Bandit**: Security linting for Python code

```yaml
runs-on: ubuntu-latest
duration: ~2-3 minutes
```

#### 4. Build (`build`)

Docker image build and validation:
- Build Docker image
- Run smoke test
- Cache layers for faster builds

```yaml
runs-on: ubuntu-latest
duration: ~3-5 minutes
depends-on: [lint, test]
```

#### 5. Performance (`performance`)

Load testing (only on pull requests):
- Start all required services
- Install k6
- Run smoke test
- Verify performance thresholds

```yaml
runs-on: ubuntu-latest
duration: ~5-7 minutes
depends-on: [build]
condition: pull_request only
```

### Total CI Duration

- **Fast path** (lint + test + security + build): ~12-21 minutes
- **Full path** (with performance): ~17-28 minutes

## Continuous Deployment (CD)

### Staging Deployment

**Trigger:** Push to `main` branch

**Steps:**
1. Build and tag Docker image (`main-{sha}`)
2. Push to GitHub Container Registry
3. SSH to staging server
4. Pull latest image
5. Rolling update (docker-compose up -d)
6. Run smoke tests
7. Notify success

**Environment:** `staging`
- URL: https://staging.yourdomain.com
- Requires approval: No (automatic)

### Production Deployment

**Trigger:** Git tag matching `v*` (e.g., `v1.2.3`)

**Steps:**
1. Build and tag Docker image with version tags
2. Backup production database (Redis + Qdrant)
3. **Rolling update** (zero-downtime):
   - Stop api-1 → Update → Start → Health check
   - Stop api-2 → Update → Start → Health check
   - Stop api-3 → Update → Start → Health check
4. Run production smoke tests
5. Create GitHub Release
6. Notify success

**Environment:** `production`
- URL: https://yourdomain.com
- Requires approval: Yes (via GitHub environments)
- Rollback: Automatic on failure

### Rollback Procedure

**Trigger:** Production deployment failure

**Steps:**
1. Restore Redis from latest backup
2. Restart services with previous Docker image
3. Notify team

## Scheduled Tasks

**Schedule:** Daily at 2 AM UTC (`cron: '0 2 * * *'`)

### Jobs

#### 1. Dependency Check

- Scan for security vulnerabilities (pip-audit)
- Check for outdated packages
- Create issues for critical updates

#### 2. Performance Regression Test

- Run full load test suite
- Compare with baseline metrics
- Upload results for trend analysis
- Alert on performance degradation

#### 3. Backup Health Check

- Verify backup age (< 25 hours)
- Test restore on staging environment
- Validate backup integrity
- Alert on backup failures

#### 4. Security Scan

- Trivy vulnerability scanner
- Upload results to GitHub Security tab
- Auto-create security advisories

#### 5. Cleanup Artifacts

- Delete workflow runs older than 30 days
- Keep minimum 10 recent runs
- Free up storage space

## Setup Instructions

### 1. Required Secrets

Configure in GitHub Settings → Secrets and variables → Actions:

**CI Secrets:**
```
OPENAI_API_KEY_TEST      # Test API key (limited quota)
```

**CD Secrets (Staging):**
```
STAGING_HOST             # staging.yourdomain.com
STAGING_USER             # deploy
STAGING_SSH_KEY          # SSH private key
```

**CD Secrets (Production):**
```
PROD_HOST                # yourdomain.com
PROD_USER                # deploy
PROD_SSH_KEY             # SSH private key
```

### 2. GitHub Environments

Create environments in Settings → Environments:

**Staging:**
- No protection rules
- Auto-deploy on push to main

**Production:**
- Required reviewers: 2 people
- Deployment branches: Tags only
- Wait timer: 5 minutes

### 3. Branch Protection

Configure in Settings → Branches → Add rule:

**Branch name pattern:** `main`

**Rules:**
- ✅ Require pull request reviews (2 approvals)
- ✅ Require status checks to pass
  - lint
  - test
  - security
  - build
- ✅ Require branches to be up to date
- ✅ Require conversation resolution
- ✅ Require signed commits (optional)
- ✅ Include administrators

### 4. Server Setup

**Prerequisites on deployment servers:**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Create deploy user
sudo useradd -m -s /bin/bash deploy
sudo usermod -aG docker deploy

# Setup SSH key
sudo -u deploy mkdir -p /home/deploy/.ssh
echo "$SSH_PUBLIC_KEY" | sudo -u deploy tee -a /home/deploy/.ssh/authorized_keys
sudo chmod 600 /home/deploy/.ssh/authorized_keys

# Clone repository
sudo -u deploy git clone https://github.com/your-org/agentic-backend.git /opt/agentic-backend
cd /opt/agentic-backend

# Setup environment
sudo -u deploy cp .env.example .env
sudo -u deploy nano .env  # Configure secrets

# Start services
sudo -u deploy docker-compose up -d
```

## Deployment Workflow

### Regular Deployment

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "feat: Add new feature"

# 3. Push and create PR
git push origin feature/new-feature

# 4. CI runs automatically
# - All checks must pass
# - 2 reviewers approve

# 5. Merge to main
# - Staging deployment triggers automatically

# 6. Test on staging
curl https://staging.yourdomain.com/api/v1/health

# 7. Create release tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3

# 8. Production deployment triggers
# - Requires manual approval
# - Rolling update (zero-downtime)
# - Automatic rollback on failure
```

### Hotfix Deployment

```bash
# 1. Create hotfix branch from main
git checkout main
git checkout -b hotfix/critical-bug

# 2. Fix and commit
git commit -m "fix: Critical security patch"

# 3. Fast-track review
# - Request expedited review
# - Minimal checks

# 4. Merge and deploy
git checkout main
git merge hotfix/critical-bug
git tag -a v1.2.4 -m "Hotfix: Critical security patch"
git push origin main v1.2.4

# 5. Monitor deployment
# - Watch logs
# - Verify fix
# - Notify team
```

## Monitoring Deployments

### GitHub Actions UI

1. Go to **Actions** tab
2. Select workflow run
3. View logs for each job
4. Check annotations for errors

### Logs

```bash
# View deployment logs on server
ssh deploy@production
cd /opt/agentic-backend
docker-compose logs -f --tail=100

# View specific service
docker-compose logs api-1 -f

# View all logs
docker-compose logs -f
```

### Health Checks

```bash
# API health
curl https://yourdomain.com/api/v1/health

# Metrics
curl https://yourdomain.com/api/v1/monitoring/metrics

# Prometheus
curl https://yourdomain.com:9090/-/healthy

# Grafana
curl https://yourdomain.com:3000/api/health
```

## Rollback Procedures

### Automatic Rollback

Triggered automatically if:
- Health checks fail after deployment
- Smoke tests fail
- Critical error detected

### Manual Rollback

```bash
# Option 1: Revert to previous tag
git tag  # List tags
git checkout v1.2.2  # Previous version
git tag -a v1.2.5 -m "Rollback to v1.2.2"
git push origin v1.2.5

# Option 2: SSH to server and rollback
ssh deploy@production
cd /opt/agentic-backend
./scripts/restore-redis.sh ./backups/redis/redis_backup_latest
docker-compose down
docker-compose up -d
```

## Performance Optimization

### Caching

The pipeline uses aggressive caching:

**Pip dependencies:**
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

**Docker layers:**
```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

**Benefits:**
- Faster builds (2-3x speedup)
- Reduced bandwidth
- Lower costs

### Parallelization

Jobs run in parallel when possible:
```
lint ──┐
test ──┼──> build ──> performance
security ─┘
```

## Troubleshooting

### CI Failures

**Lint failures:**
```bash
# Fix locally
black .
isort .
flake8 app/

# Commit fixes
git add .
git commit -m "fix: Linting issues"
git push
```

**Test failures:**
```bash
# Run tests locally
docker-compose up -d redis qdrant
pytest tests/ -v

# Debug specific test
pytest tests/unit/test_rbac.py::test_admin_permissions -v
```

**Build failures:**
```bash
# Build Docker image locally
docker build -t test-build .

# Test image
docker run --rm test-build python -c "import app; print('OK')"
```

### CD Failures

**SSH connection failed:**
```bash
# Test SSH key
ssh -i ~/.ssh/deploy_key deploy@production

# Check server logs
journalctl -u ssh -n 100
```

**Deployment timeout:**
```bash
# Check server resources
ssh deploy@production
df -h        # Disk space
free -m      # Memory
docker ps    # Running containers
```

**Health check failed:**
```bash
# Check application logs
docker-compose logs api-1 --tail=100

# Check service status
docker-compose ps

# Manual health check
curl -v http://localhost:8000/api/v1/health
```

## Best Practices

1. **Always run CI locally before pushing:**
   ```bash
   black . && isort . && flake8 app/ && pytest tests/
   ```

2. **Use feature flags for risky changes:**
   ```python
   if settings.enable_new_feature:
       # New code
   ```

3. **Tag releases with semantic versioning:**
   - v1.0.0 → Major release
   - v1.1.0 → Minor features
   - v1.1.1 → Patches

4. **Write meaningful commit messages:**
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation
   - refactor: Code refactoring
   - test: Add tests
   - chore: Maintenance

5. **Monitor after deployment:**
   - Check Grafana dashboards
   - Review error logs
   - Verify key metrics

6. **Keep dependencies updated:**
   - Review weekly dependency check results
   - Update incrementally
   - Test thoroughly

## Security Considerations

- Never commit secrets to repository
- Use GitHub Secrets for sensitive data
- Rotate SSH keys quarterly
- Review security scan results
- Keep base images updated
- Use least-privilege principle for deploy user

## Metrics and KPIs

Track these metrics:

| Metric | Target | Current |
|--------|--------|---------|
| Build time | < 5 min | ~3 min |
| Test time | < 10 min | ~7 min |
| Deploy time | < 15 min | ~12 min |
| Success rate | > 95% | ~97% |
| Mean time to deploy | < 30 min | ~25 min |
| Rollback frequency | < 5% | ~2% |

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Zero-Downtime Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html)
