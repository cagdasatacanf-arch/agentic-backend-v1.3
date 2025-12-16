# LangGraph Integration Guide

## Step-by-Step Implementation

### 1. Install Dependencies
```bash
# Add to requirements.txt
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-core>=0.3.0

# Install
pip install -r requirements.txt
```

### 2. Update Your Project Structure
```
app/
├── services/
│   ├── agent_service.py         # Your existing agent (keep for now)
│   ├── graph_agent.py           # New: Basic LangGraph agent
│   └── multi_agent_system.py    # New: Advanced multi-agent
├── api/
│   ├── routes_query.py          # Your existing routes (keep)
│   └── routes_query_langgraph.py # New: LangGraph routes
└── rag.py                       # Your existing RAG service
```

### 3. Connect Your Existing RAG Service

Edit `graph_agent.py` to use your real RAG implementation:
```python
# In graph_agent.py, replace the mock search_documents tool:

from app.rag import RAGService  # Import your existing service

@tool
def search_documents(query: str) -> str:
    """Search the vector database for relevant documents."""
    # Use your actual Qdrant implementation
    rag_service = RAGService()  # Or however you initialize it
    results = rag_service.search(query, top_k=5)
    
    # Format results as JSON string
    formatted = [
        {
            "text": r.text,
            "score": r.score,
            "id": r.id,
            "metadata": r.metadata
        }
        for r in results
    ]
    
    return json.dumps(formatted, indent=2)
```

### 4. Update Your Main FastAPI App

Edit `app/main.py` to include the new routes:
```python
from app.api import routes_query  # Existing
from app.api import routes_query_langgraph  # New

# ... existing app setup ...

# Add both routers
app.include_router(routes_query.router)  # Keep existing
app.include_router(routes_query_langgraph.router)  # Add new
```

### 5. Environment Configuration

Add to your `.env`:
```bash
# Existing
OPENAI_API_KEY=sk-...
QDRANT_HOST=localhost
QDRANT_PORT=6333

# New (optional)
LANGGRAPH_TRACING=true  # Enable LangSmith tracing
LANGCHAIN_API_KEY=ls_...  # For LangSmith (optional)
```

### 6. Test the Integration
```bash
# Start your services
docker compose up -d

# Test basic LangGraph agent
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What documents do we have about AI agents?",
    "stream": false
  }'

# Test streaming
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Calculate 25 * 4 and explain the result"
  }'
```

## Migration Strategy

### Option A: Gradual Migration (Recommended)

Keep both systems running:

1. **Week 1**: Deploy LangGraph alongside existing agent
2. **Week 2**: Route 10% of traffic to LangGraph, monitor
3. **Week 3**: Increase to 50% if stable
4. **Week 4**: Full migration, deprecate old agent
```python
# In routes_query.py
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() == "true"

if USE_LANGGRAPH:
    agent = LangGraphAgent()
else:
    agent = YourExistingAgent()
```

### Option B: Feature Flagging
```python
# In config.py
class Settings(BaseSettings):
    use_langgraph: bool = Field(False, env="USE_LANGGRAPH")
    langgraph_rollout_percent: int = Field(0, env="LANGGRAPH_ROLLOUT")

# In routes
if settings.use_langgraph:
    # Use LangGraph
    pass
else:
    # Use existing agent
    pass
```

## Key Features You Get

### 1. Tool Calling with Validation
```python
@tool
def search_documents(query: str) -> str:
    """Search docs. Query must be a specific question."""
    if len(query) < 5:
        return "Error: Query too short"
    # ... actual search
```

### 2. Stateful Conversations
```python
# Conversations persist across requests
result1 = agent.query("What's in our Q3 report?", session_id="user123")
result2 = agent.query("Summarize the key points", session_id="user123")
# Agent remembers the context!
```

### 3. Conditional Logic
```python
# Agent can make decisions
def should_search_web(state):
    if "recent" in state["messages"][-1].content.lower():
        return "web_search"
    return "rag_search"
```

### 4. Human-in-the-Loop
```python
# Pause for approval before expensive operations
workflow.add_node("approval_gate", wait_for_human)
```

## Monitoring & Debugging

### Enable LangSmith Tracing
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "agentic-backend"
```

View traces at: https://smith.langchain.com

### Add Logging
```python
# In graph_agent.py
import logging

logger = logging.getLogger(__name__)

def call_model(state):
    logger.info(f"Agent reasoning with {len(state['messages'])} messages")
    # ... model call
    logger.info(f"Generated response: {response.content[:100]}...")
```

### Monitor Token Usage
```python
# Track costs
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent.query("...")
    print(f"Cost: ${cb.total_cost:.4f}")
```

## Performance Optimization

### 1. Cache LLM Responses
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### 2. Parallel Tool Execution
```python
# LangGraph automatically parallelizes independent tool calls
# No extra code needed!
```

### 3. Streaming for Long Operations
```python
# Already implemented in routes_query_langgraph.py
# Use /api/v1/query/stream endpoint
```

## Common Issues & Solutions

### Issue: "Tool not found"

**Solution**: Make sure tools are properly bound to the LLM:
```python
llm_with_tools = llm.bind_tools([tool1, tool2, tool3])
```

### Issue: "State not persisting"

**Solution**: Use checkpointer with thread IDs:
```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Always pass thread_id
config = {"configurable": {"thread_id": session_id}}
```

### Issue: "Infinite loops"

**Solution**: Add max iteration limits:
```python
def should_continue(state):
    if state["iteration"] > 10:
        return "end"
    # ... normal routing
```

## Next Steps

1. ✅ Copy the 3 new files to your project
2. ✅ Update imports in `main.py`
3. ✅ Connect your RAG service
4. ✅ Test with example queries
5. ✅ Set up LangSmith for monitoring
6. 🚀 Deploy and monitor performance

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com)
- [Example Notebooks](https://github.com/langchain-ai/langgraph/tree/main/examples)

## Quick Reference Commands
```bash
# Run tests
pytest tests/test_langgraph_agent.py -v

# Run with coverage
pytest tests/test_langgraph_agent.py --cov=app/services

# Start services
docker compose up -d

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

## File Checklist

- [ ] `app/services/graph_agent.py` - Basic LangGraph agent
- [ ] `app/api/routes_query_langgraph.py` - FastAPI routes
- [ ] `app/services/multi_agent_system.py` - Multi-agent system
- [ ] `tests/test_langgraph_agent.py` - Test suite
- [ ] `requirements.txt` - Updated dependencies
- [ ] `app/main.py` - Updated with new routes

## Support

If you encounter issues:
1. Check the error logs
2. Verify environment variables are set
3. Ensure all dependencies are installed
4. Test with the provided test suite
5. Review LangSmith traces if enabled

Good luck with your implementation! 🚀