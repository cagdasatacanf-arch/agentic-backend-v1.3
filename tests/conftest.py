"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient

from app.main import app
from app.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create test client for API testing"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_session():
    """Create test session for conversation testing"""
    from app.services.graph_agent import LangGraphAgent
    
    agent = LangGraphAgent(enable_persistence=False)
    if agent.conversation_manager:
        session_id = agent.conversation_manager.create_session(user_id="test_user")
        yield session_id
        # Cleanup
        agent.conversation_manager.delete_session(session_id)
    else:
        yield "test_session"


@pytest.fixture(autouse=True)
async def reset_monitoring_systems():
    """Reset all monitoring systems before each test"""
    from app.services.performance import clear_metrics, clear_cache
    from app.services.circuit_breaker import reset_all_circuit_breakers
    from app.services.error_recovery import clear_dead_letter_queue
    
    # Reset before test
    await clear_metrics()
    await clear_cache()
    reset_all_circuit_breakers()
    await clear_dead_letter_queue()
    
    yield
    
    # Cleanup after test
    await clear_metrics()
    await clear_cache()
    reset_all_circuit_breakers()
    await clear_dead_letter_queue()


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def sample_document():
    """Sample document for RAG testing"""
    return {
        "id": "doc_test_123",
        "text": "This is a sample document for testing RAG functionality.",
        "metadata": {
            "source": "test",
            "created_at": "2025-12-23T00:00:00Z"
        }
    }
