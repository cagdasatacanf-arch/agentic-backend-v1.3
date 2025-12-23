"""
Streaming API Routes

Server-Sent Events (SSE) endpoints for real-time streaming:
- Token-by-token response streaming
- Real-time tool execution visibility
- Better perceived performance (< 1s vs 5-30s)

Usage:
    POST /api/v1/stream/query      # Stream general query
    POST /api/v1/stream/math        # Stream math solution
    POST /api/v1/stream/code        # Stream code generation
    POST /api/v1/stream/rag         # Stream RAG query
    POST /api/v1/stream/vision      # Stream vision analysis
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
import base64

from app.services.streaming_service import get_streaming_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


# ============================================================================
# Request Models
# ============================================================================

class StreamQueryRequest(BaseModel):
    """Request model for streaming query"""
    query: str = Field(..., min_length=1, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    agent_type: str = Field(default="general", pattern="^(general|math|code|rag|vision)$")


class StreamMathRequest(BaseModel):
    """Request model for streaming math"""
    problem: str = Field(..., min_length=1, description="Math problem")
    show_steps: bool = Field(default=True, description="Include step-by-step reasoning")
    session_id: Optional[str] = None


class StreamCodeRequest(BaseModel):
    """Request model for streaming code"""
    request: str = Field(..., min_length=1, description="Code generation request")
    language: Optional[str] = Field(None, description="Target language")
    include_tests: bool = Field(default=False, description="Generate unit tests")
    session_id: Optional[str] = None


class StreamRAGRequest(BaseModel):
    """Request model for streaming RAG"""
    question: str = Field(..., min_length=1, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources")
    session_id: Optional[str] = None


class StreamVisionRequest(BaseModel):
    """Request model for streaming vision"""
    image_base64: str = Field(..., description="Base64-encoded image")
    query: Optional[str] = Field(None, description="Specific question about image")
    detail: str = Field(default="auto", pattern="^(auto|low|high)$")
    session_id: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/query")
async def stream_query(request: StreamQueryRequest):
    """
    Stream response to general query.

    Returns SSE stream with events:
    - start: Query processing started
    - token: Individual response tokens
    - tool_start: Tool execution started
    - tool_end: Tool execution completed
    - metadata: Additional information
    - done: Query processing completed
    - error: Error occurred

    Example using JavaScript:
        const eventSource = new EventSource('/api/v1/stream/query');

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'token') {
                // Append token to UI
                responseDiv.textContent += data.content;
            }
            else if (data.type === 'done') {
                // Processing complete
                eventSource.close();
            }
            else if (data.type === 'error') {
                // Handle error
                console.error(data.content);
                eventSource.close();
            }
        };

    Example using curl:
        curl -N -X POST "http://localhost:8000/api/v1/stream/query" \\
          -H "Content-Type: application/json" \\
          -d '{"query": "What is 2+2?", "agent_type": "math"}'
    """
    try:
        logger.info(f"Stream query request: {request.query[:60]}... (agent={request.agent_type})")

        service = get_streaming_service()

        return StreamingResponse(
            service.stream_response(
                query=request.query,
                agent_type=request.agent_type,
                session_id=request.session_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Stream query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/math")
async def stream_math(request: StreamMathRequest):
    """
    Stream math problem solution.

    Returns real-time stream of solution steps and answer.

    Example:
        POST /api/v1/stream/math
        {
            "problem": "Calculate (25 * 4) + sqrt(144)",
            "show_steps": true
        }

        Stream events:
        data: {"type": "start", "metadata": {"agent_type": "math"}}
        data: {"type": "metadata", "metadata": {"status": "solving"}}
        data: {"type": "token", "content": "1"}
        data: {"type": "token", "content": "1"}
        data: {"type": "token", "content": "2"}
        data: {"type": "done", "metadata": {"success": true}}
    """
    try:
        logger.info(f"Stream math request: {request.problem[:60]}...")

        service = get_streaming_service()

        return StreamingResponse(
            service.stream_response(
                query=request.problem,
                agent_type="math",
                session_id=request.session_id,
                show_steps=request.show_steps
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logger.error(f"Stream math failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/code")
async def stream_code(request: StreamCodeRequest):
    """
    Stream code generation.

    Returns real-time stream of code as it's generated.

    Example:
        POST /api/v1/stream/code
        {
            "request": "Write a Python function to calculate factorial",
            "language": "python",
            "include_tests": false
        }

        Stream events:
        data: {"type": "start", "metadata": {"agent_type": "code"}}
        data: {"type": "metadata", "metadata": {"status": "generating"}}
        data: {"type": "metadata", "metadata": {"section": "explanation"}}
        data: {"type": "token", "content": "This"}
        data: {"type": "token", "content": " function"}
        ...
        data: {"type": "metadata", "metadata": {"section": "code", "language": "python"}}
        data: {"type": "token", "content": "def"}
        data: {"type": "token", "content": " factorial"}
        ...
        data: {"type": "done", "metadata": {"success": true}}
    """
    try:
        logger.info(f"Stream code request: {request.request[:60]}...")

        service = get_streaming_service()

        return StreamingResponse(
            service.stream_response(
                query=request.request,
                agent_type="code",
                session_id=request.session_id,
                language=request.language,
                include_tests=request.include_tests
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logger.error(f"Stream code failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag")
async def stream_rag(request: StreamRAGRequest):
    """
    Stream RAG query response.

    Shows document retrieval and answer generation in real-time.

    Example:
        POST /api/v1/stream/rag
        {
            "question": "What is machine learning?",
            "top_k": 5
        }

        Stream events:
        data: {"type": "start", "metadata": {"agent_type": "rag"}}
        data: {"type": "tool_start", "metadata": {"tool": "document_retrieval"}}
        data: {"type": "tool_end", "metadata": {"sources_found": 5}}
        data: {"type": "token", "content": "Machine"}
        data: {"type": "token", "content": " learning"}
        ...
        data: {"type": "metadata", "metadata": {"sources": [...]}}
        data: {"type": "done", "metadata": {"success": true}}
    """
    try:
        logger.info(f"Stream RAG request: {request.question[:60]}...")

        service = get_streaming_service()

        return StreamingResponse(
            service.stream_response(
                query=request.question,
                agent_type="rag",
                session_id=request.session_id,
                top_k=request.top_k
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logger.error(f"Stream RAG failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision")
async def stream_vision(request: StreamVisionRequest):
    """
    Stream vision analysis.

    Returns real-time stream of image analysis.

    Example:
        POST /api/v1/stream/vision
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
            "query": "What's in this image?",
            "detail": "high"
        }

        Stream events:
        data: {"type": "start", "metadata": {"agent_type": "vision"}}
        data: {"type": "metadata", "metadata": {"status": "analyzing"}}
        data: {"type": "token", "content": "This"}
        data: {"type": "token", "content": " image"}
        ...
        data: {"type": "metadata", "metadata": {"objects": [...], "confidence": 0.95}}
        data: {"type": "done", "metadata": {"success": true}}
    """
    try:
        logger.info(f"Stream vision request: query={request.query is not None}")

        service = get_streaming_service()

        # Decode image
        image_data = base64.b64decode(request.image_base64)

        return StreamingResponse(
            service.stream_response(
                query=request.query or "Analyze this image",
                agent_type="vision",
                session_id=request.session_id,
                image_data=image_data,
                detail=request.detail
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logger.error(f"Stream vision failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for streaming system.

    Returns:
        Status of streaming capabilities
    """
    try:
        return {
            "status": "healthy",
            "streaming": "enabled",
            "supported_agents": [
                "general",
                "math",
                "code",
                "rag",
                "vision"
            ],
            "protocol": "SSE"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================================
# Example HTML Client (for testing)
# ============================================================================

EXAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Streaming Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        #response {
            border: 1px solid #ccc;
            padding: 15px;
            min-height: 200px;
            background: #f9f9f9;
            white-space: pre-wrap;
        }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #status { margin: 10px 0; color: #666; }
    </style>
</head>
<body>
    <h1>Streaming Demo</h1>

    <div>
        <label>Query:</label><br>
        <input type="text" id="query" value="What is 2+2?" style="width: 100%; padding: 8px;">
    </div>

    <div style="margin-top: 10px;">
        <label>Agent:</label>
        <select id="agent" style="padding: 8px;">
            <option value="math">Math</option>
            <option value="code">Code</option>
            <option value="rag">RAG</option>
            <option value="general">General</option>
        </select>
        <button onclick="streamQuery()">Stream Response</button>
    </div>

    <div id="status"></div>
    <div id="response"></div>

    <script>
        let eventSource = null;

        function streamQuery() {
            const query = document.getElementById('query').value;
            const agent = document.getElementById('agent').value;
            const responseDiv = document.getElementById('response');
            const statusDiv = document.getElementById('status');

            // Clear previous response
            responseDiv.textContent = '';
            statusDiv.textContent = 'Connecting...';

            // Close previous connection
            if (eventSource) {
                eventSource.close();
            }

            // Create new SSE connection
            const url = `/api/v1/stream/query`;

            fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, agent_type: agent})
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                function read() {
                    reader.read().then(({done, value}) => {
                        if (done) {
                            statusDiv.textContent = 'Stream complete';
                            return;
                        }

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = JSON.parse(line.substring(6));

                                if (data.type === 'token') {
                                    responseDiv.textContent += data.content;
                                } else if (data.type === 'start') {
                                    statusDiv.textContent = 'Streaming...';
                                } else if (data.type === 'done') {
                                    statusDiv.textContent = 'Complete!';
                                } else if (data.type === 'error') {
                                    statusDiv.textContent = 'Error: ' + data.content;
                                    statusDiv.style.color = 'red';
                                }
                            }
                        }

                        read();
                    });
                }

                read();
            });
        }
    </script>
</body>
</html>
"""

@router.get("/demo", response_class=StreamingResponse)
async def streaming_demo():
    """
    Interactive demo page for testing streaming.

    Open in browser: http://localhost:8000/api/v1/stream/demo
    """
    return StreamingResponse(
        iter([EXAMPLE_HTML]),
        media_type="text/html"
    )
