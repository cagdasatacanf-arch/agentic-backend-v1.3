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

### Manual Setup

#### 1. Clone & Configure
