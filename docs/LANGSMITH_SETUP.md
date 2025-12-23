# LangSmith Observability Setup

Complete guide to setting up LangSmith for monitoring, debugging, and optimizing your agentic backend.

## What is LangSmith?

LangSmith is LangChain's observability platform that provides:
- **Tracing**: See every step of your agent's reasoning process
- **Debugging**: Identify failures and performance bottlenecks
- **Monitoring**: Track costs, latency, and success rates
- **Evaluation**: Test and improve your agents systematically
- **Datasets**: Create test cases for regression testing

**Official site**: https://smith.langchain.com

---

## Quick Setup (5 minutes)

### Step 1: Create LangSmith Account

1. Go to https://smith.langchain.com
2. Sign up with your email or GitHub
3. Create a new organization (or use default)

### Step 2: Get API Key

1. Click your profile icon → **Settings**
2. Navigate to **API Keys**
3. Click **Create API Key**
4. Give it a name (e.g., "agentic-backend-prod")
5. Copy the key (you'll only see it once!)

### Step 3: Configure Environment

Edit your `.env` file:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=agentic-backend-production

# Optional: Set endpoint (usually not needed)
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Step 4: Restart Services

```bash
docker compose restart api

# Or if running locally:
# Just restart your uvicorn/gunicorn server
```

### Step 5: Verify It's Working

1. Make a test query:
```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is 2 + 2?",
    "use_rag": false
  }'
```

2. Go to https://smith.langchain.com
3. Click **Projects** → Select your project
4. You should see the trace appear within seconds!

---

## Understanding Traces

### What Gets Traced?

Every agent interaction creates a trace showing:

1. **Input**: User's question
2. **Agent Reasoning**: Each thought and decision
3. **Tool Calls**: When tools are invoked, with inputs/outputs
4. **LLM Calls**: Prompts sent to OpenAI and responses
5. **Final Answer**: What the agent returns

### Trace Structure

```
Trace: "User Query"
├── Agent Executor
│   ├── LLM Call #1 (Planning)
│   │   ├── Input: User question + system prompt
│   │   └── Output: Agent decides to use calculator
│   ├── Tool Call: calculator
│   │   ├── Input: "25 * 47"
│   │   └── Output: "1175"
│   ├── LLM Call #2 (Synthesis)
│   │   ├── Input: Calculator result
│   │   └── Output: Final answer
│   └── Final Output
```

---

## Key Features

### 1. Real-time Monitoring

**View live traces:**
- Go to your project in LangSmith
- See requests appear in real-time
- Click any trace to drill down

**What to look for:**
- ⏱️ **Latency**: Total time and breakdown by step
- 💰 **Costs**: Token usage and estimated cost
- ❌ **Errors**: Failed tool calls or LLM errors
- 🔄 **Iterations**: How many reasoning steps

### 2. Debugging Failed Queries

When something goes wrong:

1. **Find the trace** in LangSmith
2. **Expand each step** to see what happened
3. **Check for**:
   - Incorrect tool selection
   - Malformed tool inputs
   - LLM hallucinations
   - Timeout errors
   - RAG retrieval failures

**Example debug flow:**
```
❌ Error: Agent gave wrong answer
├── Step 1: Check LLM planning call
│   └── Issue: Agent chose wrong tool
├── Step 2: Review system prompt
│   └── Fix: Add clearer tool descriptions
└── Step 3: Re-test and verify fix
```

### 3. Performance Optimization

**Identify bottlenecks:**

1. Click **Analytics** in your project
2. View metrics:
   - Average latency per endpoint
   - Token usage trends
   - Error rates
   - Cost breakdown

**Common optimizations:**

| Issue | Solution |
|-------|----------|
| High latency | Use streaming, reduce RAG top_k |
| High costs | Switch to GPT-3.5 for simple queries |
| Too many tool calls | Improve prompts, add examples |
| Low accuracy | Add more context, improve RAG chunking |

### 4. Datasets & Testing

Create regression test suites:

#### Create Dataset in LangSmith UI:

1. Go to **Datasets** → **Create Dataset**
2. Name it (e.g., "calculator-tests")
3. Add test cases:

```json
[
  {
    "input": {"question": "What is 5 + 3?"},
    "expected": {"contains": "8"}
  },
  {
    "input": {"question": "Calculate 100 / 4"},
    "expected": {"contains": "25"}
  }
]
```

#### Run Tests from Code:

```python
from langsmith import Client
from app.services.graph_agent import create_agent_graph

client = Client()

# Load dataset
dataset = client.read_dataset(dataset_name="calculator-tests")

# Run evaluations
results = client.run_on_dataset(
    dataset_name="calculator-tests",
    llm_or_chain_factory=create_agent_graph,
    evaluation=my_evaluation_function
)

print(f"Accuracy: {results['accuracy']}")
```

---

## Advanced Configuration

### Different Projects per Environment

```bash
# .env.production
LANGCHAIN_PROJECT=agentic-backend-production

# .env.staging
LANGCHAIN_PROJECT=agentic-backend-staging

# .env.development
LANGCHAIN_PROJECT=agentic-backend-dev
```

Benefits:
- Separate production from test data
- Different teammates can have their own projects
- Easy to compare across environments

### Custom Tags and Metadata

Add custom metadata to traces:

**In `app/services/graph_agent.py`:**

```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable(
    run_type="chain",
    name="Agent Query",
    tags=["production", "v1.3"],
    metadata={"version": "1.3.0"}
)
async def process_query(question: str, session_id: str):
    # Get current trace
    run_tree = get_current_run_tree()

    # Add custom metadata
    if run_tree:
        run_tree.add_tags(["user-query"])
        run_tree.add_metadata({
            "session_id": session_id,
            "question_length": len(question),
            "timestamp": datetime.now().isoformat()
        })

    # Your logic here
    result = await agent.ainvoke({"input": question})
    return result
```

Now you can filter by tags in LangSmith UI!

### Sampling for Cost Control

Only trace a percentage of requests:

```python
import random
from app.config import settings

# In app/config.py
class Settings(BaseSettings):
    LANGCHAIN_TRACING_SAMPLE_RATE: float = 1.0  # 1.0 = 100%, 0.1 = 10%

# In your code
def should_trace() -> bool:
    return random.random() < settings.LANGCHAIN_TRACING_SAMPLE_RATE

# Conditionally enable tracing
if should_trace():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### Disable Tracing for Specific Functions

```python
from langchain_core.tracers.context import tracing_v2_enabled

# Disable tracing for this block
with tracing_v2_enabled(False):
    # This won't be traced
    result = expensive_preprocessing()

# Re-enable for main logic
result = agent.invoke(input)
```

---

## Monitoring Best Practices

### 1. Set Up Alerts

In LangSmith (coming soon):
- Alert on error rate > 5%
- Alert on latency > 10s
- Alert on cost spike

### 2. Regular Review Schedule

**Daily**: Check error logs
**Weekly**: Review performance metrics
**Monthly**: Analyze trends and optimize

### 3. Track Key Metrics

Create a dashboard to monitor:

```python
# Custom metrics tracking
from langsmith import Client
import pandas as pd

client = Client()

# Fetch runs from last 24 hours
runs = client.list_runs(
    project_name="agentic-backend-production",
    start_time=datetime.now() - timedelta(days=1)
)

# Calculate metrics
df = pd.DataFrame([
    {
        "latency": r.total_time,
        "tokens": r.total_tokens,
        "cost": r.total_cost,
        "success": r.status == "success"
    }
    for r in runs
])

print(f"Average Latency: {df['latency'].mean():.2f}s")
print(f"Success Rate: {df['success'].mean()*100:.1f}%")
print(f"Total Cost: ${df['cost'].sum():.2f}")
```

---

## Integration with Existing Monitoring

### Combine with Jaeger (OpenTelemetry)

You can use both LangSmith and Jaeger:

- **LangSmith**: AI-specific insights (LLM calls, reasoning, costs)
- **Jaeger**: Infrastructure insights (API latency, DB queries, caching)

They complement each other!

**Example setup:**

```python
from opentelemetry import trace
from langsmith import traceable

tracer = trace.get_tracer(__name__)

@traceable(name="Query Processing")
@tracer.start_as_current_span("process_query")
async def process_query(question: str):
    # Both LangSmith and Jaeger will trace this
    result = await agent.ainvoke({"input": question})
    return result
```

### Export to Custom Analytics

```python
from langsmith import Client
import json

client = Client()

# Export traces to your analytics system
runs = client.list_runs(project_name="agentic-backend-production")

for run in runs:
    # Send to your analytics
    analytics.track({
        "event": "agent_query",
        "user_id": run.metadata.get("user_id"),
        "latency": run.total_time,
        "tokens": run.total_tokens,
        "success": run.status == "success"
    })
```

---

## Troubleshooting

### Traces Not Appearing

**Check 1**: Verify environment variables
```bash
docker compose exec api env | grep LANGCHAIN
```

Should show:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=agentic-backend-production
```

**Check 2**: Check API key is valid
```bash
curl -H "x-api-key: YOUR_LANGSMITH_API_KEY" \
  https://api.smith.langchain.com/info
```

**Check 3**: Check logs for errors
```bash
docker compose logs api | grep -i langsmith
```

### High LangSmith Costs

LangSmith pricing (as of 2024):
- Free tier: 5K traces/month
- Developer: $39/mo for 50K traces
- Team: $199/mo for 500K traces

**Reduce costs:**
1. Use sampling (trace 10% of requests)
2. Delete old projects you don't need
3. Use shorter retention periods
4. Upgrade to team plan if you need more traces

### Network Errors

If you see "Failed to send trace to LangSmith":

1. Check firewall allows outbound to `api.smith.langchain.com`
2. Verify no proxy issues
3. Try setting explicit endpoint:
   ```bash
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

---

## Cost Estimation

### LangSmith Costs

- **Free Tier**: 5,000 traces/month
- **Beyond**: ~$0.001 per trace

Example:
- 1,000 queries/day = 30,000/month = ~$25/month
- 100 queries/day = 3,000/month = Free!

### OpenAI Costs (visible in LangSmith)

LangSmith shows token usage per trace:

| Model | Input | Output | Avg Cost/Query |
|-------|-------|--------|----------------|
| GPT-4o | $2.50/1M | $10/1M | $0.01-0.05 |
| GPT-4 | $30/1M | $60/1M | $0.10-0.30 |
| GPT-3.5 Turbo | $0.50/1M | $1.50/1M | $0.001-0.01 |

---

## Example Queries to Monitor

### Find Expensive Queries
In LangSmith, filter by:
- Total tokens > 10,000
- Sort by cost descending

### Find Slow Queries
Filter by:
- Latency > 10s
- Check which step is slow

### Find Failed Queries
Filter by:
- Status = "error"
- Review error messages

### Find Queries with Many Iterations
Filter by:
- Runs with > 5 LLM calls
- May indicate prompt issues

---

## Next Steps

1. ✅ Sign up for LangSmith
2. ✅ Add API key to `.env`
3. ✅ Restart services
4. ✅ Make test queries
5. 📊 Explore traces in LangSmith
6. 🎯 Set up evaluation datasets
7. 📈 Create monitoring dashboard
8. 🔔 Configure alerts

**Resources:**
- LangSmith Docs: https://docs.smith.langchain.com
- API Reference: https://api.smith.langchain.com/redoc
- Tutorials: https://docs.smith.langchain.com/tutorials

Happy monitoring! 📊
