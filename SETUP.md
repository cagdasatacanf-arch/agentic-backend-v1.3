# Agentic Backend v1.3 - Setup Guide

Complete guide to get the Agentic Backend system up and running in minutes.

## Quick Start (Recommended)

### For Linux/macOS:
```bash
./quick-start.sh
```

### For Windows:
```cmd
quick-start.bat
```

That's it! The script will:
1. Check all prerequisites (Docker, Docker Compose)
2. Configure your environment (OpenAI API key, Internal API key)
3. Build and start all services
4. Run health checks
5. Display service URLs and test commands

---

## What You Need

### Required:
- **Docker Desktop** or Docker Engine + Docker Compose
  - Download: https://docs.docker.com/get-docker/
- **OpenAI API Key**
  - Get one at: https://platform.openai.com/api-keys
  - Ensure you have credits available

### Optional:
- **Python 3.11+** (for local development without Docker)
- **Node.js 18+** (to run the React frontend)
- **curl** or **Postman** (for testing APIs)

---

## Installation Methods

### Method 1: Automated Setup (Easiest) ⭐

**Linux/macOS:**
```bash
# Run the quick-start script
./quick-start.sh

# Test the system
./test-system.sh
```

**Windows:**
```cmd
REM Run the quick-start script
quick-start.bat
```

The script will prompt you for:
- Your OpenAI API key
- It will auto-generate an internal API key for you

### Method 2: Manual Docker Setup

```bash
# 1. Create .env file
cp .env.example .env

# 2. Edit .env and set required values
nano .env  # or use any text editor
# Set: OPENAI_API_KEY and INTERNAL_API_KEY

# 3. Start all services
docker compose up -d --build

# 4. Check logs
docker compose logs -f api

# 5. Test the API
curl http://localhost:8000/api/v1/health
```

### Method 3: Local Python Development

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure services only
docker compose up -d redis qdrant jaeger

# 4. Create and configure .env
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, INTERNAL_API_KEY
# Set: QDRANT_HOST=localhost, REDIS_HOST=localhost

# 5. Run the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Testing Your Installation

### Automated Testing
```bash
./test-system.sh
```

This runs 8 comprehensive tests:
1. Health check endpoint
2. Metadata endpoint
3. LangGraph agent health
4. Calculator query (no RAG)
5. Document upload
6. RAG query
7. Session management
8. Authentication check

### Manual Testing

**1. Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**2. Simple Query (Calculator):**
```bash
# Replace YOUR_API_KEY with the key from .env
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is 25 * 47?",
    "use_rag": false
  }'
```

**3. Upload a Document:**
```bash
echo "The Eiffel Tower is located in Paris, France." > test.txt

curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@test.txt"
```

**4. Query with RAG:**
```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "Where is the Eiffel Tower?",
    "use_rag": true
  }'
```

---

## Service URLs

Once running, access these services:

| Service | URL | Description |
|---------|-----|-------------|
| **API Server** | http://localhost:8000 | Main FastAPI application |
| **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **Jaeger UI** | http://localhost:16686 | Distributed tracing dashboard |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Vector database UI |
| **Redis** | localhost:6379 | Key-value store (no web UI) |

---

## Configuration

### Environment Variables (.env)

**Required:**
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
INTERNAL_API_KEY=your-secure-random-string
```

**Common Customizations:**
```bash
# AI Models
OPENAI_CHAT_MODEL=gpt-4o              # or gpt-4, gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Settings
RAG_TOP_K=5                           # Number of documents to retrieve
RAG_SCORE_THRESHOLD=0.7               # Similarity threshold (0.0-1.0)
CHUNK_SIZE=1000                       # Document chunk size
CHUNK_OVERLAP=200                     # Chunk overlap

# Agent Settings
MAX_AGENT_ITERATIONS=10               # Max reasoning loops
ENABLE_LANGGRAPH_PERSISTENCE=true     # Persistent conversations

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Session Expiration
REDIS_TTL_SECONDS=604800              # 7 days
```

**Optional - LangSmith Observability:**
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=agentic-backend
```
Get LangSmith API key: https://smith.langchain.com

---

## Common Commands

### Docker Management
```bash
# View logs
docker compose logs -f api          # API logs only
docker compose logs -f              # All services

# Restart services
docker compose restart

# Stop services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Check service status
docker compose ps

# Rebuild after code changes
docker compose up -d --build
```

### Python Development
```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_health.py -v

# Run with coverage
pytest --cov=app tests/

# Format code
black app/
isort app/
```

---

## Troubleshooting

### Services won't start

**Check Docker is running:**
```bash
docker info
```

**Check logs:**
```bash
docker compose logs -f api
docker compose logs -f redis
docker compose logs -f qdrant
```

**Clean restart:**
```bash
docker compose down -v
docker compose up -d --build
```

### API returns 500 errors

**Most common causes:**
1. Invalid OpenAI API key
2. No OpenAI credits remaining
3. Services not fully initialized

**Solutions:**
```bash
# 1. Verify API key in .env
cat .env | grep OPENAI_API_KEY

# 2. Check OpenAI credits
# Visit: https://platform.openai.com/account/usage

# 3. Wait 30 seconds for initialization
sleep 30
curl http://localhost:8000/api/v1/health
```

### Port conflicts

If ports 8000, 6379, 6333, or 16686 are already in use:

**Option 1 - Kill conflicting process:**
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

**Option 2 - Change ports in docker-compose.yml:**
```yaml
services:
  api:
    ports:
      - "8001:8000"  # Change 8001 to any free port
```

### Redis connection errors

```bash
# Test Redis connection
docker compose exec redis redis-cli ping
# Should return: PONG

# Clear Redis data
docker compose exec redis redis-cli FLUSHALL
```

### Qdrant issues

```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# View collections
curl http://localhost:6333/collections

# Recreate Qdrant volume
docker compose down -v
docker compose up -d --build
```

### OpenAI rate limits

If you see "Rate limit exceeded" errors:
1. Wait a few minutes
2. Upgrade your OpenAI plan
3. Reduce `RATE_LIMIT_PER_MINUTE` in .env

---

## Advanced Setup

### Running Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev

# Frontend available at http://localhost:5173
```

Configure frontend API endpoint in `frontend/src/config.ts` if needed.

### Using LangSmith for Observability

1. Sign up at https://smith.langchain.com
2. Get your API key
3. Update .env:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-key
   LANGCHAIN_PROJECT=agentic-backend
   ```
4. Restart: `docker compose restart`
5. View traces at https://smith.langchain.com

### Custom Tool Development

Add custom tools in `app/services/agent_service.py`:

```python
def my_custom_tool(input: str) -> str:
    """Description of what this tool does."""
    # Your logic here
    return result

# Register in TOOLS_REGISTRY
TOOLS_REGISTRY["my_custom_tool"] = my_custom_tool
```

### Production Deployment

**Security checklist:**
- [ ] Change `INTERNAL_API_KEY` to a strong random value
- [ ] Set `ENVIRONMENT=prod`
- [ ] Enable HTTPS/TLS
- [ ] Use managed Redis (AWS ElastiCache, etc.)
- [ ] Use managed Qdrant (Qdrant Cloud)
- [ ] Set up proper firewall rules
- [ ] Enable LangSmith monitoring
- [ ] Configure log aggregation
- [ ] Set up alerting

**Production docker-compose:**
```yaml
# Use production-grade configurations
services:
  api:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Getting Help

- **Documentation**: Check `README.md`, `LANGGRAPH_INTEGRATION.md`
- **API Docs**: http://localhost:8000/docs (when running)
- **Logs**: `docker compose logs -f api`
- **Issues**: Create an issue in the repository

---

## Next Steps

1. ✅ Complete setup using `./quick-start.sh`
2. ✅ Run tests using `./test-system.sh`
3. 📚 Explore API docs at http://localhost:8000/docs
4. 🔍 Upload documents and test RAG functionality
5. 🎨 Customize tools and agents for your use case
6. 📊 Monitor with Jaeger at http://localhost:16686
7. 🚀 Deploy to production

Happy building! 🎉
