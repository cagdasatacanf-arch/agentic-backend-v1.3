# Documentation

Comprehensive guides for configuring and extending your agentic backend.

## 📚 Available Guides

### Getting Started
- **[../SETUP.md](../SETUP.md)** - Complete setup and installation guide
- **[../README.md](../README.md)** - Project overview and quick start

### Configuration Guides

| Guide | Description | Time | Difficulty |
|-------|-------------|------|------------|
| **[Configuration Index](CONFIGURATION_INDEX.md)** | Complete environment variable reference | 10 min | ⭐ Beginner |
| **[API Testing Guide](API_TESTING.md)** | Test all endpoints with examples | 20 min | ⭐ Beginner |
| **[Custom Tools Development](CUSTOM_TOOLS.md)** | Create custom tools for your agents | 30 min | ⭐⭐ Intermediate |
| **[LangSmith Setup](LANGSMITH_SETUP.md)** | Monitor and debug with LangSmith | 15 min | ⭐ Beginner |
| **[RAG Optimization](RAG_OPTIMIZATION.md)** | Improve retrieval quality and performance | 45 min | ⭐⭐⭐ Advanced |

### Advanced Topics

| Guide | Description | Time | Difficulty |
|-------|-------------|------|------------|
| **[Agentic AI Enhancements](AGENTIC_AI_ENHANCEMENTS.md)** | Research-backed roadmap for advanced features | 60 min | ⭐⭐⭐ Advanced |
| **[Phase 1 Implementation](PHASE1_IMPLEMENTATION.md)** | ✅ Completed: Tool metrics, quality scoring, multi-hop RAG | 30 min | ⭐⭐ Intermediate |

### Integration Guides
- **[../LANGGRAPH_INTEGRATION.md](../LANGGRAPH_INTEGRATION.md)** - LangGraph multi-agent system
- **[../ERROR_GUIDELINES.md](../ERROR_GUIDELINES.md)** - Error handling standards
- **[../architecture-diagram.md](../architecture-diagram.md)** - System architecture overview

---

## Quick Links

### I want to...

**...get started quickly**
→ Run `../quick-start.sh` (Linux/macOS) or `../quick-start.bat` (Windows)

**...test the API**
→ Run `../test_api_client.py` or see [API Testing Guide](API_TESTING.md)

**...add a custom tool**
→ See [Custom Tools Development](CUSTOM_TOOLS.md)

**...monitor my agents**
→ See [LangSmith Setup](LANGSMITH_SETUP.md)

**...improve RAG quality**
→ See [RAG Optimization](RAG_OPTIMIZATION.md)

**...configure environment variables**
→ See [Configuration Index](CONFIGURATION_INDEX.md)

**...understand the architecture**
→ See [Architecture Diagram](../architecture-diagram.md)

**...handle errors properly**
→ See [Error Guidelines](../ERROR_GUIDELINES.md)

**...explore advanced AI techniques**
→ See [Agentic AI Enhancements](AGENTIC_AI_ENHANCEMENTS.md)

---

## Configuration by Goal

### 🎯 Maximize Quality
```bash
OPENAI_CHAT_MODEL=gpt-4o
RAG_TOP_K=7
RAG_SCORE_THRESHOLD=0.75
MAX_AGENT_ITERATIONS=15
```
**Cost**: $$$, **Speed**: Slower, **Quality**: Excellent

### ⚡ Maximize Speed
```bash
OPENAI_CHAT_MODEL=gpt-3.5-turbo
RAG_TOP_K=3
CHUNK_SIZE=800
MAX_AGENT_ITERATIONS=8
```
**Cost**: $, **Speed**: Fast, **Quality**: Good

### 💰 Minimize Cost
```bash
OPENAI_CHAT_MODEL=gpt-3.5-turbo
RAG_TOP_K=3
RAG_SCORE_THRESHOLD=0.75
# Enable embedding caching (see RAG_OPTIMIZATION.md)
```
**Cost**: $, **Speed**: Good, **Quality**: Good

### 🎓 Research & Analysis
```bash
OPENAI_CHAT_MODEL=gpt-4o
RAG_TOP_K=10
RAG_SCORE_THRESHOLD=0.65
MAX_AGENT_ITERATIONS=15
CHUNK_SIZE=1500
```
**Cost**: $$$, **Speed**: Slower, **Quality**: Excellent for complex tasks

---

## Common Tasks

### 1. Enable Monitoring (5 min)
1. Sign up at https://smith.langchain.com
2. Get API key
3. Add to `.env`:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_...
   ```
4. Restart: `docker compose restart api`

→ Full guide: [LangSmith Setup](LANGSMITH_SETUP.md)

### 2. Add Custom Tool (15 min)
1. Create tool function in `app/tools/`
2. Register in `app/services/agent_service.py`
3. Add to LangGraph in `app/services/graph_agent.py`
4. Test via API

→ Full guide: [Custom Tools](CUSTOM_TOOLS.md)

### 3. Optimize RAG (30 min)
1. Upload test documents
2. Run test queries
3. Adjust `RAG_TOP_K` and `RAG_SCORE_THRESHOLD`
4. Measure improvement
5. Iterate

→ Full guide: [RAG Optimization](RAG_OPTIMIZATION.md)

---

## Support

- **Issues**: Create issue in repository
- **Questions**: Check documentation first
- **API Reference**: http://localhost:8000/docs (when running)

---

## Contributing

Found an error in the docs? Have a suggestion?
- Create an issue
- Submit a pull request
- Suggest improvements

---

**Happy building! 🚀**
