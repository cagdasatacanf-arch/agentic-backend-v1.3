"""
Test suite for LangGraph agent integration.
Place in: tests/test_langgraph_agent.py
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.services.graph_agent import LangGraphAgent, create_graph
from app.services.multi_agent_system import MultiAgentSystem


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("langchain_openai.ChatOpenAI") as mock:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = "This is a test response"
        mock_response.tool_calls = []
        
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_instance.bind_tools.return_value = mock_instance
        
        mock.return_value = mock_instance
        yield mock


# ============================================================================
# UNIT TESTS - Graph Agent
# ============================================================================

def test_graph_creation():
    """Test that the graph compiles without errors."""
    graph = create_graph()
    assert graph is not None


def test_agent_initialization():
    """Test that the agent initializes correctly."""
    agent = LangGraphAgent()
    assert agent.graph is not None


@patch("app.services.graph_agent.ChatOpenAI")
def test_simple_query(mock_openai):
    """Test a simple query through the agent."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.content = "Hello! I'm working correctly."
    mock_response.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    # Run query
    agent = LangGraphAgent()
    result = agent.query("Hello")
    
    assert result["success"] is True
    assert "answer" in result
    assert result["iterations"] >= 0


@patch("app.services.graph_agent.ChatOpenAI")
def test_query_with_tools(mock_openai):
    """Test that the agent can call tools."""
    # Setup mock to simulate tool calling
    mock_tool_call = MagicMock()
    mock_tool_call.content = ""
    mock_tool_call.tool_calls = [{
        "name": "calculator",
        "args": {"expression": "2+2"},
        "id": "call_123"
    }]
    
    mock_final = MagicMock()
    mock_final.content = "The answer is 4"
    mock_final.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.side_effect = [mock_tool_call, mock_final]
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    # Run query
    agent = LangGraphAgent()
    result = agent.query("What is 2+2?")
    
    assert result["success"] is True


def test_query_error_handling():
    """Test that errors are handled gracefully."""
    agent = LangGraphAgent()
    
    # Simulate error by passing invalid state
    with patch.object(agent.graph, "stream", side_effect=Exception("Test error")):
        result = agent.query("test")
        assert result["success"] is False
        assert "error" in result


# ============================================================================
# INTEGRATION TESTS - FastAPI Endpoints
# ============================================================================

@patch("app.services.graph_agent.ChatOpenAI")
def test_query_endpoint(mock_openai, client):
    """Test the /api/v1/query endpoint."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.content = "Test answer"
    mock_response.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    # Make request
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is AI?",
            "session_id": "test-123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "iterations" in data


@patch("app.services.graph_agent.ChatOpenAI")
def test_streaming_endpoint(mock_openai, client):
    """Test the /api/v1/query/stream endpoint."""
    mock_response = MagicMock()
    mock_response.content = "Streaming response"
    mock_response.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    response = client.post(
        "/api/v1/query/stream",
        json={"question": "Test streaming"}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_query_validation(client):
    """Test request validation."""
    # Missing required field
    response = client.post(
        "/api/v1/query",
        json={}
    )
    assert response.status_code == 422  # Validation error
    
    # Invalid type
    response = client.post(
        "/api/v1/query",
        json={"question": 123}  # Should be string
    )
    assert response.status_code == 422


@patch("app.services.graph_agent.ChatOpenAI")
def test_agent_health_check(mock_openai, client):
    """Test the agent health check endpoint."""
    mock_response = MagicMock()
    mock_response.content = "pong"
    mock_response.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    response = client.get("/api/v1/health/agent")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["framework"] == "langgraph"


# ============================================================================
# INTEGRATION TESTS - Multi-Agent System
# ============================================================================

@patch("app.services.multi_agent_system.ChatOpenAI")
def test_multi_agent_initialization(mock_openai):
    """Test multi-agent system initialization."""
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    
    system = MultiAgentSystem()
    assert system.graph is not None


@patch("app.services.multi_agent_system.ChatOpenAI")
def test_multi_agent_routing(mock_openai):
    """Test that supervisor routes to correct agents."""
    # Mock supervisor decision
    supervisor_response = MagicMock()
    supervisor_response.content = "research"
    
    # Mock agent response
    agent_response = MagicMock()
    agent_response.content = "Research completed"
    agent_response.tool_calls = []
    
    # Mock finish decision
    finish_response = MagicMock()
    finish_response.content = "FINISH"
    
    mock_instance = MagicMock()
    mock_instance.invoke.side_effect = [
        supervisor_response,  # First routing
        agent_response,       # Research agent
        finish_response       # Final routing
    ]
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    system = MultiAgentSystem()
    result = system.query("Research AI trends", thread_id="test-multi")
    
    assert result["success"] is True


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
@patch("app.services.graph_agent.ChatOpenAI")
def test_concurrent_queries(mock_openai):
    """Test handling multiple concurrent queries."""
    import concurrent.futures
    
    mock_response = MagicMock()
    mock_response.content = "Concurrent response"
    mock_response.tool_calls = []
    
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response
    mock_instance.bind_tools.return_value = mock_instance
    mock_openai.return_value = mock_instance
    
    agent = LangGraphAgent()
    
    def run_query(i):
        return agent.query(f"Test query {i}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_query, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    assert len(results) == 10
    assert all(r["success"] for r in results)


@pytest.mark.slow
def test_max_iterations_limit():
    """Test that infinite loops are prevented."""
    with patch("app.services.graph_agent.ChatOpenAI") as mock_openai:
        # Simulate agent that keeps calling tools
        mock_tool_response = MagicMock()
        mock_tool_response.content = ""
        mock_tool_response.tool_calls = [{
            "name": "calculator",
            "args": {"expression": "1+1"},
            "id": "call_loop"
        }]
        
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_tool_response
        mock_instance.bind_tools.return_value = mock_instance
        mock_openai.return_value = mock_instance
        
        agent = LangGraphAgent()
        
        # Should not hang - graph should have max iterations
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Query took too long")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        try:
            result = agent.query("Loop test")
            # If we get here, iteration limit worked
            assert True
        except TimeoutError:
            pytest.fail("Query exceeded timeout - no iteration limit?")
        finally:
            signal.alarm(0)


# ============================================================================
# TOOL TESTS
# ============================================================================

def test_calculator_tool():
    """Test the calculator tool directly."""
    from app.services.graph_agent import calculator
    
    result = calculator.invoke({"expression": "2 + 2"})
    assert result == "4"
    
    result = calculator.invoke({"expression": "sqrt(16)"})
    assert result == "4.0"
    
    result = calculator.invoke({"expression": "invalid"})
    assert "Error" in result


def test_search_documents_tool():
    """Test the search_documents tool."""
    from app.services.graph_agent import search_documents
    
    result = search_documents.invoke({"query": "test query"})
    
    # Should return JSON
    data = json.loads(result)
    assert isinstance(data, list)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])