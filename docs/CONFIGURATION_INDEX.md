# Configuration Guide Index

Complete reference for configuring and customizing your agentic backend.

## Quick Links

| Guide | What You'll Learn | Time |
|-------|-------------------|------|
| **[Custom Tools](CUSTOM_TOOLS.md)** | Create custom tools for your agents | 30 min |
| **[LangSmith Setup](LANGSMITH_SETUP.md)** | Monitor and debug your agents | 15 min |
| **[RAG Optimization](RAG_OPTIMIZATION.md)** | Improve retrieval quality and performance | 45 min |

---

## Environment Variables Reference

Complete `.env` configuration options:

### Required Settings

```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-your-key-here          # Get from https://platform.openai.com

# Authentication (Required)
INTERNAL_API_KEY=your-secure-key         # Generate with: openssl rand -hex 32
```

### AI Models

```bash
# Chat Model
OPENAI_CHAT_MODEL=gpt-4o                 # Options: gpt-4o, gpt-4, gpt-3.5-turbo

# Embedding Model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large
```

### RAG Configuration

```bash
# Retrieval Settings
RAG_TOP_K=5                              # Number of documents to retrieve (3-15)
RAG_SCORE_THRESHOLD=0.7                  # Similarity threshold (0.0-1.0)

# Chunking Settings
CHUNK_SIZE=1000                          # Characters per chunk (500-2000)
CHUNK_OVERLAP=200                        # Overlap between chunks (100-400)
```

**See [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md) for detailed tuning guide.**

### Agent Configuration

```bash
# LangGraph Settings
ENABLE_LANGGRAPH_PERSISTENCE=true        # Enable conversation memory
MAX_AGENT_ITERATIONS=10                  # Max reasoning loops (5-20)
```

### Database Configuration

```bash
# Qdrant (Vector Database)
QDRANT_HOST=localhost                    # or qdrant (in Docker)
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
QDRANT_API_KEY=                          # Optional, for cloud deployment

# Redis (Session Storage)
REDIS_HOST=localhost                     # or redis (in Docker)
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=                          # Optional, for production
REDIS_TTL_SECONDS=604800                 # Session expiration (7 days)
```

### Observability (Optional)

```bash
# LangSmith
LANGCHAIN_TRACING_V2=false               # Set to true to enable
LANGCHAIN_API_KEY=                       # Get from https://smith.langchain.com
LANGCHAIN_PROJECT=agentic-backend        # Project name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# OpenTelemetry / Jaeger
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

**See [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md) for setup guide.**

### Rate Limiting

```bash
RATE_LIMIT_PER_MINUTE=60                 # Requests per minute
RATE_LIMIT_PER_HOUR=1000                 # Requests per hour
```

### Environment

```bash
ENVIRONMENT=local                        # Options: local, dev, staging, prod
```

---

## Configuration by Use Case

### Development Environment

```bash
# .env.development
ENVIRONMENT=local
OPENAI_CHAT_MODEL=gpt-3.5-turbo         # Cheaper for dev
RAG_TOP_K=3                             # Faster
LANGCHAIN_TRACING_V2=true               # Debug with LangSmith
RATE_LIMIT_PER_MINUTE=100               # Higher for testing
```

### Production Environment

```bash
# .env.production
ENVIRONMENT=prod
OPENAI_CHAT_MODEL=gpt-4o                # Best quality
RAG_TOP_K=5                             # Balanced
LANGCHAIN_TRACING_V2=true               # Monitor production
RATE_LIMIT_PER_MINUTE=60                # Protect from abuse
REDIS_PASSWORD=strong-password-here     # Secure Redis
QDRANT_API_KEY=your-qdrant-cloud-key   # Use Qdrant Cloud
```

### High-Volume Production

```bash
# .env.high-volume
OPENAI_CHAT_MODEL=gpt-3.5-turbo         # Cost optimization
RAG_TOP_K=3                             # Speed optimization
CHUNK_SIZE=800                          # Smaller chunks, faster
RATE_LIMIT_PER_MINUTE=120               # Higher throughput
REDIS_TTL_SECONDS=86400                 # 1 day (shorter for memory)
```

### Research/Analysis

```bash
# .env.research
OPENAI_CHAT_MODEL=gpt-4o                # Best reasoning
RAG_TOP_K=10                            # More context
RAG_SCORE_THRESHOLD=0.65                # Broader retrieval
CHUNK_SIZE=1500                         # Larger context
MAX_AGENT_ITERATIONS=15                 # More reasoning steps
```

---

## Quick Configuration Tasks

### Task 1: Enable LangSmith Monitoring

**Time**: 5 minutes

```bash
# 1. Get API key from https://smith.langchain.com
# 2. Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxx...
LANGCHAIN_PROJECT=my-project

# 3. Restart
docker compose restart api
```

**Guide**: [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md)

### Task 2: Add a Custom Weather Tool

**Time**: 15 minutes

```bash
# 1. Create app/tools/weather.py
# 2. Implement get_weather(city: str) -> str
# 3. Register in app/services/agent_service.py
# 4. Test via API
```

**Guide**: [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md)

### Task 3: Optimize RAG for Your Documents

**Time**: 30 minutes

```bash
# 1. Upload test documents
# 2. Try queries and check relevance
# 3. Adjust RAG_TOP_K and RAG_SCORE_THRESHOLD
# 4. Test again
# 5. Iterate until satisfied
```

**Guide**: [RAG_OPTIMIZATION.md](RAG_OPTIMIZATION.md)

### Task 4: Switch to GPT-4 for Better Quality

**Time**: 2 minutes

```bash
# In .env:
OPENAI_CHAT_MODEL=gpt-4

# Restart:
docker compose restart api
```

**Note**: GPT-4 is ~10x more expensive but significantly better at reasoning.

### Task 5: Reduce Costs

**Time**: 10 minutes

**Option 1 - Use GPT-3.5:**
```bash
OPENAI_CHAT_MODEL=gpt-3.5-turbo
```

**Option 2 - Reduce RAG retrieval:**
```bash
RAG_TOP_K=3                  # Less context = fewer tokens
```

**Option 3 - Cache embeddings:**
```bash
# Add Redis caching for embeddings (requires code changes)
# See RAG_OPTIMIZATION.md for implementation
```

---

## Advanced Configuration

### Multi-Environment Setup

**Project structure:**
```
.env.local
.env.staging
.env.production
docker-compose.yml
docker-compose.prod.yml
```

**Switch environments:**
```bash
# Development
docker compose up

# Staging
docker compose --env-file .env.staging up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Environment-Specific Overrides

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  api:
    environment:
      - ENVIRONMENT=production
      - OPENAI_CHAT_MODEL=gpt-4o
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Secrets Management

**For production, use Docker secrets:**

```bash
# Create secrets
echo "sk-xxx" | docker secret create openai_api_key -
echo "strong-password" | docker secret create redis_password -

# Use in compose:
services:
  api:
    secrets:
      - openai_api_key
      - redis_password
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
```

---

## Configuration Validation

### Check Current Configuration

```bash
# View all environment variables
docker compose exec api env | grep -E "OPENAI|RAG|REDIS|QDRANT"

# Test OpenAI connection
curl http://localhost:8000/api/v1/metadata

# Test Redis connection
docker compose exec redis redis-cli ping

# Test Qdrant connection
curl http://localhost:6333/healthz
```

### Validate Before Deploy

**Checklist:**
- [ ] OPENAI_API_KEY is set and valid
- [ ] INTERNAL_API_KEY is strong (>32 chars)
- [ ] Redis is accessible
- [ ] Qdrant is accessible
- [ ] RAG settings are tuned for your use case
- [ ] Rate limits are appropriate
- [ ] LangSmith is configured (for production)

---

## Troubleshooting

### Common Issues

**1. "OpenAI API key is invalid"**
```bash
# Check key in .env
cat .env | grep OPENAI_API_KEY

# Verify it's being passed to container
docker compose exec api env | grep OPENAI_API_KEY

# Test key manually
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer sk-your-key"
```

**2. "No documents found in RAG"**
```bash
# Check Qdrant collection
curl http://localhost:6333/collections/documents

# Should show: "points_count": > 0

# If 0, upload documents:
curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@test.pdf"
```

**3. "Slow responses"**
```bash
# Check current settings
cat .env | grep RAG_TOP_K

# Reduce for speed:
RAG_TOP_K=3
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# Restart
docker compose restart api
```

**4. "High OpenAI costs"**
```bash
# Switch to cheaper model
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# Reduce context
RAG_TOP_K=3
CHUNK_SIZE=800

# Monitor in LangSmith
# See LANGSMITH_SETUP.md
```

---

## Configuration Best Practices

### 1. Version Control

```bash
# DO commit:
.env.example

# DON'T commit:
.env
.env.local
.env.production
```

**Add to .gitignore:**
```
.env
.env.*
!.env.example
```

### 2. Documentation

**Document your custom settings:**

```bash
# .env
# ==================================================
# CUSTOM CONFIGURATION for XYZ Use Case
# ==================================================
# We use higher RAG_TOP_K (10) because our documents
# are technical and require more context.
RAG_TOP_K=10

# We use gpt-4 because accuracy is critical for
# our legal compliance use case.
OPENAI_CHAT_MODEL=gpt-4
```

### 3. Regular Reviews

**Monthly checklist:**
- [ ] Review OpenAI costs in LangSmith
- [ ] Check if RAG settings still optimal
- [ ] Update to newer models if available
- [ ] Review and rotate API keys
- [ ] Check for configuration drift across environments

---

## Migration Guides

### Upgrading OpenAI Models

**GPT-3.5 to GPT-4o:**
```bash
# Before:
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# After:
OPENAI_CHAT_MODEL=gpt-4o

# Expect:
# - 10x cost increase
# - 2x better quality
# - Similar latency
```

### Changing Embedding Models

**Small to Large embeddings:**
```bash
# Before:
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# After:
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Note: You must re-index all documents!
# 1. Clear Qdrant collection
# 2. Re-upload all documents
```

---

## Getting Help

**Configuration Questions:**
1. Check this guide first
2. Review specific guide (CUSTOM_TOOLS.md, etc.)
3. Check logs: `docker compose logs -f api`
4. Search LangChain/LangGraph docs

**Issues:**
- Create issue in repository
- Include relevant `.env` settings (redact API keys!)
- Include error logs

---

## Next Steps

1. ✅ Review current `.env` settings
2. 📋 Choose configuration for your use case
3. 🔧 Apply recommended settings
4. 🧪 Test with your workload
5. 📊 Monitor in LangSmith
6. 🔄 Iterate and optimize

**Happy configuring! ⚙️**
