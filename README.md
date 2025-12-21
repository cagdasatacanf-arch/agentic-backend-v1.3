# agentic-backend-v1.3

Production-ready FastAPI backend for agentic AI systems with RAG (Retrieval-Augmented Generation) and multi-agent orchestration.

## Features

✅ **FastAPI Framework**
- High-performance async API
- Automatic API documentation (Swagger UI)
- Type validation with Pydantic
- CORS support

✅ **AI/ML Capabilities**
- OpenAI integration (GPT-4o for chat, text-embedding-3-small for embeddings)
- LangChain & LangGraph for agent orchestration
- Vector embeddings with Qdrant
- Semantic search and RAG pipeline
- 13+ built-in tools (calculator, web search, file operations, code execution, etc.)

✅ **Production Ready**
- Docker containerization
- Multi-service orchestration (FastAPI + Redis + Qdrant + Jaeger)
- Persistent conversation memory via Redis
- Health checks for all services
- Distributed tracing via OpenTelemetry + Jaeger
- Structured logging with Loguru

✅ **Caching & Performance**
- Redis for session management and caching
- Request throttling (SlowAPI)
- Optimized vector search
- Configurable timeouts & resource limits

---

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- OpenAI API key (https://platform.openai.com)
- Python 3.11+ (optional, for local dev)

### Automated Setup (Recommended) ⭐

**Linux/macOS:**
```bash
./quick-start.sh
```

**Windows:**
```cmd
quick-start.bat
```

The script will automatically:
1. Check prerequisites
2. Configure environment variables
3. Build and start all services
4. Run health checks
5. Display service URLs and test commands

See [SETUP.md](SETUP.md) for detailed setup instructions.

---

## Documentation

Comprehensive guides for configuration and customization:

| Guide | Description | Time |
|-------|-------------|------|
| **[Setup Guide](SETUP.md)** | Complete installation and troubleshooting | 15 min |
| **[API Testing](docs/API_TESTING.md)** | Test all endpoints with curl, Python, Postman | 20 min |
| **[Configuration Index](docs/CONFIGURATION_INDEX.md)** | Environment variables reference | 10 min |
| **[Custom Tools](docs/CUSTOM_TOOLS.md)** | Create custom tools for your agents | 30 min |
| **[LangSmith Setup](docs/LANGSMITH_SETUP.md)** | Monitor and debug with LangSmith | 15 min |
| **[RAG Optimization](docs/RAG_OPTIMIZATION.md)** | Improve retrieval quality | 45 min |
| **[LangGraph Integration](LANGGRAPH_INTEGRATION.md)** | Multi-agent system architecture | 20 min |

**Browse all guides**: [docs/README.md](docs/README.md)

---

## Service URLs

When running, access these services:

| Service | URL | Description |
|---------|-----|-------------|
| **API Server** | http://localhost:8000 | Main FastAPI application |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Jaeger** | http://localhost:16686 | Distributed tracing |
| **Qdrant** | http://localhost:6333/dashboard | Vector database UI |

---

## API Examples

### Simple Query (Calculator)
```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is 25 * 47?",
    "use_rag": false
  }'
```

### Upload Document for RAG
```bash
curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf"
```

### Query with RAG
```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What information is in my documents about X?",
    "use_rag": true
  }'
```

### Streaming Response
```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "Explain how agents work",
    "use_rag": false
  }'
```

---

## Testing

### Quick API Test

Run the interactive test client to verify all endpoints:

```bash
# Interactive API test client (tests all endpoints)
python3 test_api_client.py
```

### System Tests

Run comprehensive tests:

```bash
# Automated system tests
./test-system.sh

# Unit tests
pytest tests/ -v

# Specific test file
pytest tests/test_health.py -v

# With coverage
pytest --cov=app tests/
```

### Manual Testing

- **Postman**: Import `postman_collection.json`
- **curl**: See examples in [API Testing Guide](docs/API_TESTING.md)
- **Swagger UI**: http://localhost:8000/docs

---

## Project Structure

```
agentic-backend-v1.3/
├── app/                          # Main application
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration settings
│   ├── rag.py                    # RAG implementation
│   ├── api/                      # API routes
│   │   ├── routes_query.py
│   │   ├── routes_query_langgraph.py
│   │   └── routes_docs.py
│   ├── services/                 # Business logic
│   │   ├── agent_service.py      # Agent orchestration
│   │   ├── graph_agent.py        # LangGraph agents
│   │   └── redis_checkpointer.py # Session persistence
│   └── utils/                    # Utilities
│       └── chunking.py           # Document chunking
├── frontend/                     # React UI
├── tests/                        # Test suite
├── docs/                         # Documentation
├── docker-compose.yml            # Service orchestration
├── quick-start.sh                # Automated setup (Linux/macOS)
├── quick-start.bat               # Automated setup (Windows)
├── test-system.sh                # System tests
└── README.md                     # This file
```

---

## Common Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Restart after code changes
docker compose restart api

# Stop all services
docker compose down

# Clean restart (removes data)
docker compose down -v && docker compose up -d --build

# Run tests
pytest tests/ -v

# Check service status
docker compose ps
```

---

## Configuration

Key environment variables (`.env`):

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
INTERNAL_API_KEY=your-secure-key

# Models
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Settings
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional: LangSmith Monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

See [Configuration Index](docs/CONFIGURATION_INDEX.md) for complete reference.

---

## Troubleshooting

**Services won't start:**
```bash
docker compose logs -f
docker compose down -v
docker compose up -d --build
```

**API returns 500 errors:**
- Check OpenAI API key is valid
- Verify you have OpenAI credits
- Check logs: `docker compose logs -f api`

**Port conflicts:**
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or change port in docker-compose.yml
```

See [SETUP.md](SETUP.md) for detailed troubleshooting.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

[Add your license here]

---

## Support

- **Documentation**: [docs/README.md](docs/README.md)
- **Issues**: Create an issue in the repository
- **API Docs**: http://localhost:8000/docs (when running)

---

**Built with ❤️ using FastAPI, LangChain, and OpenAI**
