# Custom Tools Development Guide

Learn how to create and integrate custom tools into your agentic backend.

## What are Tools?

Tools are functions that your AI agent can call to perform specific tasks. The system comes with 13+ built-in tools, but you can easily add your own.

---

## Built-in Tools Reference

Your agent has access to these tools by default:

| Tool | Purpose | Example |
|------|---------|---------|
| `calculator` | Math operations | "What is 25 * 47?" |
| `web_search` | DuckDuckGo search | "Search for latest Python news" |
| `search_documents` | RAG vector search | "Find info about Paris in my docs" |
| `read_file` | Read file contents | "Read the config.json file" |
| `write_file` | Write to file | "Save this data to output.txt" |
| `list_directory` | List files in dir | "What files are in /data?" |
| `run_python` | Execute Python code | "Run this Python script" |
| `make_http_request` | HTTP API calls | "GET https://api.example.com/data" |
| `create_agent` | Create sub-agent | "Create a specialized research agent" |
| `delegate_to_agent` | Delegate to agent | "Ask the research agent about AI" |
| `deep_research` | Perplexity AI research | "Do deep research on quantum computing" |
| `ask_human` | Request user input | "Ask the user for their preference" |
| `finish` | Complete task | "Task completed" |

---

## Creating a Custom Tool

### Step 1: Define Your Tool Function

Create a new file or add to existing service file:

**Example: `app/tools/custom_tools.py`**

```python
from typing import Dict, Any
import httpx
from loguru import logger

def get_weather(city: str) -> str:
    """
    Get current weather for a specified city.

    Args:
        city: Name of the city (e.g., "London", "New York")

    Returns:
        str: Weather information including temperature and conditions
    """
    try:
        # Using OpenWeatherMap API (replace with your API key)
        api_key = "your-openweathermap-api-key"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()

        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]

        return f"Weather in {city}: {temp}°C, {description}"

    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return f"Could not fetch weather for {city}: {str(e)}"


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of given text.

    Args:
        text: Text to analyze

    Returns:
        dict: Sentiment analysis result with score and classification
    """
    # Simple example - in production, use a proper sentiment analysis model
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        sentiment = "positive"
        score = 0.7
    elif neg_count > pos_count:
        sentiment = "negative"
        score = -0.7
    else:
        sentiment = "neutral"
        score = 0.0

    return {
        "sentiment": sentiment,
        "score": score,
        "text_length": len(text)
    }


def database_query(query: str) -> str:
    """
    Execute a safe database query.

    Args:
        query: SQL query to execute (SELECT only for safety)

    Returns:
        str: Query results as formatted string
    """
    # Example with SQLite - adapt for your database
    import sqlite3

    # Safety check - only allow SELECT queries
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety"

    try:
        conn = sqlite3.connect("/app/data/database.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "Query returned no results"

        # Format results
        formatted = "\n".join([str(row) for row in results])
        return f"Query results:\n{formatted}"

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return f"Database error: {str(e)}"


def send_notification(message: str, channel: str = "slack") -> str:
    """
    Send notification to external service.

    Args:
        message: Notification message
        channel: Channel to send to (slack, email, discord)

    Returns:
        str: Confirmation message
    """
    # Example: Slack webhook
    if channel == "slack":
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

        try:
            response = httpx.post(
                webhook_url,
                json={"text": message},
                timeout=10.0
            )
            response.raise_for_status()
            return f"Notification sent to {channel}: {message}"

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return f"Failed to send notification: {str(e)}"

    return f"Channel {channel} not supported"
```

---

### Step 2: Register Your Tool

Add your tool to the registry in `app/services/agent_service.py`:

```python
# At the top of the file
from app.tools.custom_tools import (
    get_weather,
    analyze_sentiment,
    database_query,
    send_notification
)

# In the TOOLS_REGISTRY dictionary
TOOLS_REGISTRY = {
    # ... existing tools ...

    # Your custom tools
    "get_weather": get_weather,
    "analyze_sentiment": analyze_sentiment,
    "database_query": database_query,
    "send_notification": send_notification,
}
```

---

### Step 3: Convert to LangChain Tool (for LangGraph)

Update `app/services/graph_agent.py` to include your tools:

```python
from langchain.tools import Tool

# Create LangChain tool wrappers
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a specified city. Input should be a city name.",
    func=get_weather
)

sentiment_tool = Tool(
    name="analyze_sentiment",
    description="Analyze sentiment of text. Input should be the text to analyze.",
    func=lambda text: str(analyze_sentiment(text))
)

# Add to tools list in create_agent_graph()
def create_agent_graph():
    tools = [
        # ... existing tools ...
        weather_tool,
        sentiment_tool,
    ]
    # ... rest of the function
```

---

### Step 4: Test Your Tool

```bash
# Test via API
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is the weather in London?",
    "use_rag": false
  }'
```

---

## Best Practices

### 1. Clear Documentation
```python
def my_tool(param: str) -> str:
    """
    Brief description of what the tool does.

    Args:
        param: Describe the parameter - be specific!
              Example: "City name (e.g., 'Paris', 'Tokyo')"

    Returns:
        str: Describe what gets returned
    """
```

The docstring is crucial - the AI uses it to understand when and how to use your tool!

### 2. Error Handling
```python
def robust_tool(input: str) -> str:
    try:
        # Your logic
        result = do_something(input)
        return result

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return f"Error: Invalid input - {str(e)}"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Error: {str(e)}"
```

### 3. Input Validation
```python
def validated_tool(url: str) -> str:
    """Fetch data from URL."""

    # Validate input
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    if "dangerous-site.com" in url:
        return "Error: This domain is blocked"

    # Proceed with validated input
    return fetch_url(url)
```

### 4. Timeouts and Limits
```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def external_api_call(endpoint: str) -> str:
    """Call external API with retry logic."""

    response = httpx.get(
        endpoint,
        timeout=10.0,  # 10 second timeout
        follow_redirects=True
    )
    response.raise_for_status()
    return response.json()
```

### 5. Type Hints
```python
from typing import Dict, List, Optional, Any

def typed_tool(
    required_param: str,
    optional_param: Optional[int] = None
) -> Dict[str, Any]:
    """Tools with proper type hints are easier to maintain."""

    result = {
        "input": required_param,
        "count": optional_param or 0
    }
    return result
```

---

## Advanced: Async Tools

For I/O-bound operations, use async tools:

```python
import httpx
from typing import Dict

async def async_weather_tool(city: str) -> str:
    """Async version for better performance."""

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}",
            timeout=10.0
        )
        data = response.json()
        return f"Weather: {data['main']['temp']}°C"


# Register async tool
from langchain.tools import Tool

async_weather = Tool(
    name="get_weather",
    description="Get weather for a city",
    func=async_weather_tool,
    coroutine=async_weather_tool  # Register async version
)
```

---

## Tool Categories

### Data Retrieval Tools
- External APIs
- Database queries
- File system access
- Web scraping

### Data Processing Tools
- Text analysis
- Image processing
- Data transformation
- Calculations

### Integration Tools
- Slack/Discord notifications
- Email sending
- CRM updates
- Payment processing

### System Tools
- File operations
- System commands
- Docker management
- Process monitoring

---

## Security Considerations

### 1. Input Sanitization
```python
import re

def safe_file_reader(filename: str) -> str:
    """Read file with path validation."""

    # Prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return "Error: Invalid file path"

    # Only allow alphanumeric and safe chars
    if not re.match(r'^[a-zA-Z0-9_\-./]+$', filename):
        return "Error: Invalid filename characters"

    # Read from safe directory only
    safe_path = f"/app/data/{filename}"
    with open(safe_path, 'r') as f:
        return f.read()
```

### 2. Rate Limiting
```python
from datetime import datetime, timedelta
from collections import defaultdict

# Simple in-memory rate limiter
_rate_limits = defaultdict(list)

def rate_limited_tool(input: str, limit: int = 10, window: int = 60) -> str:
    """Tool with rate limiting (10 calls per 60 seconds)."""

    tool_name = "rate_limited_tool"
    now = datetime.now()
    cutoff = now - timedelta(seconds=window)

    # Clean old entries
    _rate_limits[tool_name] = [
        t for t in _rate_limits[tool_name] if t > cutoff
    ]

    # Check limit
    if len(_rate_limits[tool_name]) >= limit:
        return f"Error: Rate limit exceeded. Try again in {window} seconds."

    # Add this call
    _rate_limits[tool_name].append(now)

    # Your tool logic
    return f"Processed: {input}"
```

### 3. API Key Management
```python
from app.config import settings

def secure_api_tool(query: str) -> str:
    """Use API keys from environment, not hardcoded."""

    api_key = settings.EXTERNAL_SERVICE_API_KEY

    if not api_key:
        return "Error: API key not configured"

    # Use the key securely
    response = httpx.get(
        "https://api.example.com/data",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    return response.text
```

---

## Testing Your Tools

Create tests in `tests/test_custom_tools.py`:

```python
import pytest
from app.tools.custom_tools import get_weather, analyze_sentiment

def test_weather_tool():
    """Test weather tool with valid city."""
    result = get_weather("London")
    assert "Weather in London" in result
    assert "°C" in result

def test_weather_invalid_city():
    """Test weather tool with invalid city."""
    result = get_weather("InvalidCityXYZ123")
    assert "Error" in result or "Could not fetch" in result

def test_sentiment_positive():
    """Test sentiment analysis with positive text."""
    result = analyze_sentiment("I love this! It's great!")
    assert result["sentiment"] == "positive"
    assert result["score"] > 0

def test_sentiment_negative():
    """Test sentiment analysis with negative text."""
    result = analyze_sentiment("This is terrible and awful")
    assert result["sentiment"] == "negative"
    assert result["score"] < 0

@pytest.mark.asyncio
async def test_async_tool():
    """Test async tools."""
    from app.tools.custom_tools import async_weather_tool
    result = await async_weather_tool("Paris")
    assert "Weather" in result
```

Run tests:
```bash
pytest tests/test_custom_tools.py -v
```

---

## Examples by Use Case

### Example 1: Stripe Payment Tool
```python
import stripe
from app.config import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

def create_payment_intent(amount: int, currency: str = "usd") -> str:
    """
    Create a Stripe payment intent.

    Args:
        amount: Amount in cents (e.g., 1000 = $10.00)
        currency: Currency code (default: usd)

    Returns:
        str: Payment intent ID or error message
    """
    try:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            payment_method_types=["card"],
        )
        return f"Payment intent created: {intent.id}"

    except stripe.error.StripeError as e:
        return f"Payment error: {str(e)}"
```

### Example 2: Image Analysis Tool
```python
from PIL import Image
import base64
from io import BytesIO

def analyze_image(image_path: str) -> str:
    """
    Analyze image and return metadata.

    Args:
        image_path: Path to image file

    Returns:
        str: Image analysis results
    """
    try:
        img = Image.open(image_path)

        return f"""Image Analysis:
- Format: {img.format}
- Size: {img.size[0]}x{img.size[1]} pixels
- Mode: {img.mode}
- File: {image_path}
"""
    except Exception as e:
        return f"Error analyzing image: {str(e)}"
```

### Example 3: Translation Tool
```python
from googletrans import Translator

translator = Translator()

def translate_text(text: str, target_lang: str = "es") -> str:
    """
    Translate text to target language.

    Args:
        text: Text to translate
        target_lang: Target language code (es, fr, de, etc.)

    Returns:
        str: Translated text
    """
    try:
        result = translator.translate(text, dest=target_lang)
        return f"Translation ({target_lang}): {result.text}"

    except Exception as e:
        return f"Translation error: {str(e)}"
```

---

## Next Steps

1. **Create your tool** in `app/tools/custom_tools.py`
2. **Register it** in `app/services/agent_service.py`
3. **Add to LangGraph** in `app/services/graph_agent.py`
4. **Test it** via API or write unit tests
5. **Deploy** by restarting the service

Need help? Check the existing tools in:
- `app/services/agent_service.py` (tool registry)
- `app/services/graph_agent.py` (LangGraph integration)

Happy tool building! 🛠️
