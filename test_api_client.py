#!/usr/bin/env python3
"""
Interactive API Test Client for Agentic Backend
Run this script to test all API endpoints interactively.
"""

import httpx
import json
import os
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class AgenticAPIClient:
    """Client for testing Agentic Backend API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

    def test_connection(self) -> bool:
        """Test if API is reachable."""
        try:
            response = httpx.get(f"{self.base_url}/api/v1/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"{Colors.FAIL}Connection failed: {e}{Colors.ENDC}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        response = httpx.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

    def metadata(self) -> Dict[str, Any]:
        """Get system metadata."""
        response = httpx.get(f"{self.base_url}/api/v1/metadata")
        response.raise_for_status()
        return response.json()

    def query(
        self,
        question: str,
        use_rag: bool = False,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Send a query to the agent."""
        data = {
            "question": question,
            "use_rag": use_rag
        }
        if session_id:
            data["session_id"] = session_id
        if top_k:
            data["top_k"] = top_k
        if score_threshold:
            data["score_threshold"] = score_threshold

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
        data = {"question": question, "use_rag": use_rag}

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
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        pass

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload a document for RAG."""
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
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
            headers={"X-API-Key": self.api_key},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details."""
        response = httpx.get(
            f"{self.base_url}/api/v1/langgraph/sessions/{session_id}",
            headers={"X-API-Key": self.api_key},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get session conversation history."""
        response = httpx.get(
            f"{self.base_url}/api/v1/langgraph/sessions/{session_id}/history",
            headers={"X-API-Key": self.api_key},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session."""
        response = httpx.delete(
            f"{self.base_url}/api/v1/langgraph/sessions/{session_id}",
            headers={"X-API-Key": self.api_key},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def run_tests():
    """Run all API tests."""
    # Load environment
    load_dotenv()
    api_key = os.getenv("INTERNAL_API_KEY")

    if not api_key:
        print_error("INTERNAL_API_KEY not found in .env file")
        print_info("Please run ./quick-start.sh or create .env file")
        sys.exit(1)

    client = AgenticAPIClient(api_key=api_key)

    print_header("Agentic Backend API Test Suite")

    # Test 1: Connection
    print_header("Test 1: Connection & Health Check")
    try:
        if not client.test_connection():
            print_error("Cannot connect to API")
            print_info("Make sure services are running: docker compose ps")
            sys.exit(1)

        health = client.health_check()
        print_success("API is reachable")
        print(f"  Status: {health.get('status')}")
        print(f"  Services: {', '.join(health.get('services', {}).keys())}")

        metadata = client.metadata()
        print_success("Metadata retrieved")
        print(f"  Chat Model: {metadata.get('chat_model')}")
        print(f"  Embedding Model: {metadata.get('embedding_model')}")

    except Exception as e:
        print_error(f"Health check failed: {e}")
        sys.exit(1)

    # Test 2: Simple Query (Calculator)
    print_header("Test 2: Simple Query (Calculator Tool)")
    try:
        result = client.query("What is 25 * 47? Just give me the number.", use_rag=False)
        print_success("Query successful")
        print(f"  Question: What is 25 * 47?")
        print(f"  Answer: {result['answer']}")
        print(f"  Session ID: {result.get('session_id')}")

        if '1175' in result['answer']:
            print_success("Correct answer!")
        else:
            print_error("Unexpected answer")

    except Exception as e:
        print_error(f"Simple query failed: {e}")

    # Test 3: Streaming Response
    print_header("Test 3: Streaming Response")
    try:
        print_info("Streaming response...")
        print("  Response: ", end="", flush=True)

        full_response = ""
        for chunk in client.query_stream("Explain AI in one short sentence", use_rag=False):
            if chunk.get("type") == "chunk":
                content = chunk.get("content", "")
                print(content, end="", flush=True)
                full_response += content

        print()  # New line
        if full_response:
            print_success("Streaming successful")
        else:
            print_error("No streaming data received")

    except Exception as e:
        print_error(f"Streaming failed: {e}")

    # Test 4: Document Upload & RAG
    print_header("Test 4: Document Upload & RAG")
    try:
        # Create test document
        test_file = "test_api_doc.txt"
        test_content = """
        The capital of France is Paris.
        Paris is known for the Eiffel Tower, which was built in 1889.
        The Louvre Museum is located in Paris.
        """

        with open(test_file, "w") as f:
            f.write(test_content)

        print_info(f"Created test document: {test_file}")

        # Upload
        upload_result = client.upload_document(test_file)
        print_success("Document uploaded")
        print(f"  Filename: {upload_result.get('filename')}")
        print(f"  Chunks: {upload_result.get('chunks_created')}")

        # Wait for indexing
        print_info("Waiting 2 seconds for indexing...")
        import time
        time.sleep(2)

        # Query with RAG
        rag_result = client.query("What is the capital of France?", use_rag=True)
        print_success("RAG query successful")
        print(f"  Question: What is the capital of France?")
        print(f"  Answer: {rag_result['answer']}")

        sources = rag_result.get('sources', [])
        if sources:
            print(f"  Sources found: {len(sources)}")
            print(f"  Best match score: {sources[0].get('score', 0):.3f}")

        if 'paris' in rag_result['answer'].lower():
            print_success("RAG returned correct answer!")
        else:
            print_error("RAG answer unexpected")

        # Cleanup
        os.remove(test_file)

    except Exception as e:
        print_error(f"RAG test failed: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)

    # Test 5: Session Management
    print_header("Test 5: Session Management & Memory")
    try:
        session_id = "test-api-session-123"

        # First message
        result1 = client.query(
            "My favorite programming language is Python",
            session_id=session_id,
            use_rag=False
        )
        print_success("First message sent")
        print(f"  Session ID: {session_id}")

        # Second message (test memory)
        result2 = client.query(
            "What is my favorite programming language?",
            session_id=session_id,
            use_rag=False
        )
        print_success("Follow-up message sent")
        print(f"  Answer: {result2['answer']}")

        if 'python' in result2['answer'].lower():
            print_success("Session memory working!")
        else:
            print_error("Session memory might not be working")

        # Get session history
        history = client.get_session_history(session_id)
        print_success("Session history retrieved")
        print(f"  Total messages: {history.get('total_messages', 0)}")

        # List all sessions
        sessions = client.list_sessions()
        print_success("Sessions listed")
        print(f"  Total sessions: {sessions.get('total', 0)}")

        # Cleanup
        client.delete_session(session_id)
        print_success("Session deleted")

    except Exception as e:
        print_error(f"Session management failed: {e}")

    # Test 6: Error Handling
    print_header("Test 6: Error Handling")
    try:
        # Test invalid API key
        bad_client = AgenticAPIClient(api_key="invalid-key")
        try:
            bad_client.query("test")
            print_error("Invalid API key was accepted (should have failed!)")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print_success("Invalid API key correctly rejected (403)")
            else:
                print_error(f"Unexpected status code: {e.response.status_code}")

    except Exception as e:
        print_error(f"Error handling test failed: {e}")

    # Summary
    print_header("Test Summary")
    print_success("All tests completed!")
    print_info("\nNext steps:")
    print("  1. View API docs: http://localhost:8000/docs")
    print("  2. Check Jaeger traces: http://localhost:16686")
    print("  3. View Qdrant dashboard: http://localhost:6333/dashboard")
    print("  4. Run integration tests: pytest tests/ -v")
    print()


if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
