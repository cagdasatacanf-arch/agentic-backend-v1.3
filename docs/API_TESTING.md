# API Testing Guide

Complete guide to testing your agentic backend API endpoints.

## Quick Reference

### Health & Metadata Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# System metadata
curl http://localhost:8000/api/v1/metadata

# Agent health
curl http://localhost:8000/api/v1/langgraph/health/agent
```

---

## Authentication

All protected endpoints require the `X-API-Key` header:

```bash
# Get your API key from .env file
API_KEY=$(grep "INTERNAL_API_KEY=" .env | cut -d '=' -f2)

# Use in requests
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/langgraph/sessions
```

---

## Query Endpoints

### 1. Simple Query (No RAG)

Test basic agent functionality with calculator:

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is 25 * 47?",
    "use_rag": false
  }'
```

**Expected Response:**
```json
{
  "answer": "The result is 1175.",
  "session_id": "auto-generated-uuid",
  "sources": [],
  "metadata": {
    "iterations": 2,
    "tools_used": ["calculator"],
    "total_time": 1.23
  }
}
```

### 2. Web Search Query

Test web search tool:

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What are the latest developments in AI?",
    "use_rag": false
  }'
```

### 3. Streaming Query

Get real-time responses:

```bash
curl -N -X POST http://localhost:8000/api/v1/langgraph/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "Explain quantum computing in simple terms",
    "use_rag": false
  }'
```

**Expected Response (Server-Sent Events):**
```
data: {"type": "thought", "content": "I need to explain quantum computing..."}

data: {"type": "chunk", "content": "Quantum computing is..."}

data: {"type": "chunk", "content": " based on principles of..."}

data: {"type": "complete", "answer": "Quantum computing is based on..."}
```

### 4. Query with Custom Session

Maintain conversation history:

```bash
# First message
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "My name is Alice",
    "session_id": "user-123-session",
    "use_rag": false
  }'

# Follow-up (agent remembers context)
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is my name?",
    "session_id": "user-123-session",
    "use_rag": false
  }'
```

---

## Document & RAG Endpoints

### 1. Upload Document

```bash
# Upload a text file
echo "The capital of France is Paris. Paris is known for the Eiffel Tower." > test.txt

curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@test.txt"
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Document uploaded and indexed successfully",
  "filename": "test.txt",
  "chunks_created": 1,
  "total_size": 68
}
```

### 2. Upload PDF

```bash
curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf"
```

### 3. Query with RAG

```bash
# After uploading documents
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is the capital of France?",
    "use_rag": true
  }'
```

**Expected Response:**
```json
{
  "answer": "According to the documents, the capital of France is Paris.",
  "session_id": "uuid",
  "sources": [
    {
      "content": "The capital of France is Paris...",
      "score": 0.89,
      "metadata": {"filename": "test.txt"}
    }
  ],
  "metadata": {
    "retrieved_docs": 1,
    "tools_used": ["search_documents"]
  }
}
```

### 4. RAG with Custom Parameters

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "Tell me about Paris",
    "use_rag": true,
    "top_k": 3,
    "score_threshold": 0.8
  }'
```

---

## Session Management

### 1. Create Session

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "session_id": "my-custom-session",
    "metadata": {
      "user_id": "user-123",
      "context": "customer-support"
    }
  }'
```

### 2. List All Sessions

```bash
curl http://localhost:8000/api/v1/langgraph/sessions \
  -H "X-API-Key: YOUR_API_KEY"
```

**Expected Response:**
```json
{
  "sessions": [
    {
      "session_id": "my-custom-session",
      "created_at": "2024-01-15T10:30:00Z",
      "message_count": 5,
      "metadata": {"user_id": "user-123"}
    }
  ],
  "total": 1
}
```

### 3. Get Session Details

```bash
curl http://localhost:8000/api/v1/langgraph/sessions/my-custom-session \
  -H "X-API-Key: YOUR_API_KEY"
```

### 4. Get Session History

```bash
curl http://localhost:8000/api/v1/langgraph/sessions/my-custom-session/history \
  -H "X-API-Key: YOUR_API_KEY"
```

**Expected Response:**
```json
{
  "session_id": "my-custom-session",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?",
      "timestamp": "2024-01-15T10:30:01Z"
    }
  ],
  "total_messages": 2
}
```

### 5. Delete Session

```bash
curl -X DELETE http://localhost:8000/api/v1/langgraph/sessions/my-custom-session \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## Error Testing

### 1. Test Invalid API Key

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: invalid-key" \
  -d '{"question": "test"}'
```

**Expected Response: 403 Forbidden**
```json
{
  "detail": "Invalid API key"
}
```

### 2. Test Missing API Key

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

**Expected Response: 403 Forbidden**

### 3. Test Malformed JSON

```bash
curl -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d 'invalid json'
```

**Expected Response: 422 Unprocessable Entity**

### 4. Test Invalid File Upload

```bash
# Try to upload unsupported file type
echo "test" > test.exe

curl -X POST http://localhost:8000/api/v1/docs/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@test.exe"
```

---

## Performance Testing

### 1. Measure Response Time

```bash
curl -w "\nTime: %{time_total}s\n" \
  -X POST http://localhost:8000/api/v1/langgraph/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is 2+2?",
    "use_rag": false
  }'
```

### 2. Concurrent Requests Test

```bash
# Install Apache Bench
sudo apt-get install apache2-utils  # Ubuntu
brew install httpd  # macOS

# Run 100 requests with 10 concurrent
ab -n 100 -c 10 \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -p query.json \
  http://localhost:8000/api/v1/langgraph/query
```

Where `query.json` contains:
```json
{"question": "What is 2+2?", "use_rag": false}
```

### 3. Rate Limit Testing

```bash
# Send requests until rate limited
for i in {1..100}; do
  curl -X POST http://localhost:8000/api/v1/langgraph/query \
    -H "Content-Type: application/json" \
    -H "X-API-Key: YOUR_API_KEY" \
    -d '{"question": "test", "use_rag": false}'
  echo "Request $i"
  sleep 0.1
done
```

---

## Python Test Client

Create `test_client.py`:

```python
import httpx
import json
from typing import Dict, Any

class AgenticClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

    def query(self, question: str, use_rag: bool = False, session_id: str = None) -> Dict[str, Any]:
        """Send a query to the agent."""
        data = {
            "question": question,
            "use_rag": use_rag
        }
        if session_id:
            data["session_id"] = session_id

        response = httpx.post(
            f"{self.base_url}/api/v1/langgraph/query",
            json=data,
            headers=self.headers,
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()

    def query_stream(self, question: str, use_rag: bool = False):
        """Stream query responses."""
        data = {
            "question": question,
            "use_rag": use_rag
        }

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/v1/langgraph/query/stream",
            json=data,
            headers=self.headers,
            timeout=60.0
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload a document for RAG."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            headers = {"X-API-Key": self.api_key}
            response = httpx.post(
                f"{self.base_url}/api/v1/docs/upload",
                files=files,
                headers=headers,
                timeout=60.0
            )
        response.raise_for_status()
        return response.json()

    def list_sessions(self) -> Dict[str, Any]:
        """List all sessions."""
        response = httpx.get(
            f"{self.base_url}/api/v1/langgraph/sessions",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get session conversation history."""
        response = httpx.get(
            f"{self.base_url}/api/v1/langgraph/sessions/{session_id}/history",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session."""
        response = httpx.delete(
            f"{self.base_url}/api/v1/langgraph/sessions/{session_id}",
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # Get API key from environment or .env
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("INTERNAL_API_KEY")
    client = AgenticClient(api_key=api_key)

    # Test 1: Simple query
    print("Test 1: Calculator")
    result = client.query("What is 15 * 23?", use_rag=False)
    print(f"Answer: {result['answer']}\n")

    # Test 2: Streaming
    print("Test 2: Streaming response")
    for chunk in client.query_stream("Explain AI in one sentence", use_rag=False):
        if chunk.get("type") == "chunk":
            print(chunk["content"], end="", flush=True)
    print("\n")

    # Test 3: Upload and RAG
    print("Test 3: RAG")

    # Create test document
    with open("test_doc.txt", "w") as f:
        f.write("Python is a high-level programming language.")

    # Upload
    upload_result = client.upload_document("test_doc.txt")
    print(f"Uploaded: {upload_result}")

    # Query with RAG
    rag_result = client.query("What is Python?", use_rag=True)
    print(f"RAG Answer: {rag_result['answer']}")
    print(f"Sources: {len(rag_result.get('sources', []))}\n")

    # Test 4: Session management
    print("Test 4: Sessions")
    sessions = client.list_sessions()
    print(f"Active sessions: {sessions['total']}")
```

Run it:
```bash
pip install httpx python-dotenv
python test_client.py
```

---

## Integration Tests

Create `tests/test_integration.py`:

```python
import pytest
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("INTERNAL_API_KEY")

@pytest.fixture
def client():
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=60.0
    )

def test_health(client):
    """Test health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_metadata(client):
    """Test metadata endpoint."""
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "chat_model" in data
    assert "embedding_model" in data

def test_calculator_query(client):
    """Test calculator tool."""
    response = client.post(
        "/api/v1/langgraph/query",
        json={
            "question": "What is 7 * 8?",
            "use_rag": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "56" in data["answer"]

def test_document_upload_and_rag(client):
    """Test document upload and RAG query."""

    # Create test document
    test_content = "The sky is blue because of Rayleigh scattering."
    with open("test_upload.txt", "w") as f:
        f.write(test_content)

    # Upload document
    with open("test_upload.txt", "rb") as f:
        response = client.post(
            "/api/v1/docs/upload",
            files={"file": f}
        )
    assert response.status_code == 200

    # Wait for indexing
    import time
    time.sleep(2)

    # Query with RAG
    response = client.post(
        "/api/v1/langgraph/query",
        json={
            "question": "Why is the sky blue?",
            "use_rag": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "scatter" in data["answer"].lower()

    # Cleanup
    os.remove("test_upload.txt")

def test_session_persistence(client):
    """Test conversation memory."""
    session_id = "test-session-123"

    # First message
    response = client.post(
        "/api/v1/langgraph/query",
        json={
            "question": "My favorite color is purple",
            "session_id": session_id,
            "use_rag": False
        }
    )
    assert response.status_code == 200

    # Follow-up
    response = client.post(
        "/api/v1/langgraph/query",
        json={
            "question": "What is my favorite color?",
            "session_id": session_id,
            "use_rag": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "purple" in data["answer"].lower()

    # Cleanup
    client.delete(f"/api/v1/langgraph/sessions/{session_id}")

def test_invalid_api_key():
    """Test authentication."""
    bad_client = httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": "invalid-key"}
    )
    response = bad_client.post(
        "/api/v1/langgraph/query",
        json={"question": "test"}
    )
    assert response.status_code == 403

def test_rate_limiting(client):
    """Test rate limiting (should fail after limit)."""
    # This depends on your RATE_LIMIT_PER_MINUTE setting
    # Adjust the range based on your limits
    for i in range(70):  # Exceeds default 60/min
        response = client.post(
            "/api/v1/langgraph/query",
            json={"question": f"test {i}", "use_rag": False}
        )
        if response.status_code == 429:
            # Rate limited as expected
            return

    # If we got here, rate limiting might not be working
    pytest.fail("Rate limiting did not trigger")
```

Run integration tests:
```bash
pytest tests/test_integration.py -v
```

---

## Postman Collection

Save as `agentic-backend.postman_collection.json`:

```json
{
  "info": {
    "name": "Agentic Backend API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "auth": {
    "type": "apikey",
    "apikey": [
      {"key": "key", "value": "X-API-Key", "type": "string"},
      {"key": "value", "value": "{{API_KEY}}", "type": "string"},
      {"key": "in", "value": "header", "type": "string"}
    ]
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{BASE_URL}}/api/v1/health"
      }
    },
    {
      "name": "Simple Query",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "url": "{{BASE_URL}}/api/v1/langgraph/query",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"question\": \"What is 5 + 3?\",\n  \"use_rag\": false\n}"
        }
      }
    },
    {
      "name": "RAG Query",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "url": "{{BASE_URL}}/api/v1/langgraph/query",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"question\": \"Tell me about my documents\",\n  \"use_rag\": true\n}"
        }
      }
    },
    {
      "name": "Upload Document",
      "request": {
        "method": "POST",
        "url": "{{BASE_URL}}/api/v1/docs/upload",
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file", "src": ""}
          ]
        }
      }
    },
    {
      "name": "List Sessions",
      "request": {
        "method": "GET",
        "url": "{{BASE_URL}}/api/v1/langgraph/sessions"
      }
    }
  ],
  "variable": [
    {"key": "BASE_URL", "value": "http://localhost:8000"},
    {"key": "API_KEY", "value": "your-api-key-here"}
  ]
}
```

Import into Postman and set variables:
- `BASE_URL`: http://localhost:8000
- `API_KEY`: Your internal API key

---

## Troubleshooting Tests

### Connection Refused
```bash
# Check services are running
docker compose ps

# Check API is responding
curl http://localhost:8000/api/v1/health
```

### 403 Forbidden
```bash
# Verify API key
echo $API_KEY

# Check .env file
cat .env | grep INTERNAL_API_KEY
```

### Slow Responses
```bash
# Check if using GPT-4 (slower but better)
cat .env | grep OPENAI_CHAT_MODEL

# Consider switching to GPT-3.5 for faster responses
```

### RAG Returns No Results
```bash
# Check documents are indexed
curl http://localhost:6333/collections/documents

# Verify points_count > 0
```

---

## Next Steps

1. ✅ Start services: `./quick-start.sh`
2. ✅ Run health check
3. ✅ Test calculator query
4. ✅ Upload test document
5. ✅ Test RAG query
6. ✅ Try streaming
7. ✅ Run integration tests

Happy testing! 🧪
