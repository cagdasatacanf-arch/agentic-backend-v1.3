"""
FastAPI routes with LangGraph integration, session management, and full observability.
Place this in: app/api/routes_query_langgraph.py
"""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import json
import logging
import time
import uuid

from app.services.graph_agent import LangGraphAgent
from app.services.observability import (
    langsmith_manager,
    trace_agent_call,
    RequestContext,
    CostTracker,
    get_current_run_id
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["langgraph"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for agent queries."""
    question: str = Field(..., description="The question to ask the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    stream: bool = Field(False, description="Whether to stream the response")
    metadata: dict = Field(default_factory=dict, description="Custom metadata for tracing")


class Source(BaseModel):
    """Document source metadata."""
    id: str
    text: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    answer: str
    sources: List[Source] = []
    iterations: int
    session_id: Optional[str] = None
    metadata: dict = {}
    observability: dict = {}  # NEW: Observability data


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    user_id: Optional[str] = Field(None, description="User ID to associate with session")


class SessionResponse(BaseModel):
    """Session metadata response."""
    session_id: str
    user_id: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    message_count: int = 0
    status: str = "active"


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: dict = {}


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    messages: List[ConversationMessage]
    total_messages: int


class FeedbackRequest(BaseModel):
    """User feedback for a query."""
    run_id: str = Field(..., description="LangSmith run ID to attach feedback to")
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    feedback_type: str = Field("user_rating", description="Type of feedback")
    comment: Optional[str] = Field(None, description="Optional text comment")


class RunStatsResponse(BaseModel):
    """Statistics for a LangSmith run."""
    run_id: str
    status: str
    latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

# Initialize once at startup
agent = LangGraphAgent(enable_persistence=settings.enable_langgraph_persistence)


# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================

def create_request_context(request: Request, user_id: Optional[str] = None) -> RequestContext:
    """Create a request context for tracing."""
    request_id = str(uuid.uuid4())
    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        endpoint=request.url.path
    )


# ============================================================================
# QUERY ENDPOINTS WITH OBSERVABILITY
# ============================================================================

@router.post("/query", response_model=QueryResponse)
@trace_agent_call(
    name="langgraph_query",
    tags=["query", "langgraph", "production"]
)
async def query_agent(request: QueryRequest, http_request: Request):
    """
    Query the LangGraph agent with a question.
    
    The agent will:
    1. Retrieve relevant documents from RAG
    2. Use available tools (calculator, web search)
    3. Reason through the answer with persistent memory
    4. Return response with sources
    
    All interactions are traced in LangSmith for observability.
    """
    
    start_time = time.time()
    
    # Create request context for tracing
    req_context = create_request_context(http_request, request.user_id)
    req_context.add_metadata("question", request.question[:100])  # First 100 chars
    req_context.add_metadata("session_id", request.session_id)
    req_context.add_metadata("custom_metadata", request.metadata)
    
    try:
        # Run the agent with tracing
        result = agent.query(
            question=request.question,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Agent error: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Get current run ID from LangSmith
        run_id = get_current_run_id()
        
        # Track cost if enabled
        cost_data = {}
        if settings.langsmith_enable_cost_tracking and run_id:
            run_stats = langsmith_manager.get_run_stats(run_id)
            if run_stats:
                cost_data = {
                    "total_tokens": run_stats.get("total_tokens"),
                    "prompt_tokens": run_stats.get("prompt_tokens"),
                    "completion_tokens": run_stats.get("completion_tokens"),
                    "estimated_cost_usd": run_stats.get("total_cost")
                }
        
        # Log slow requests
        if latency_ms > settings.slow_request_threshold_ms:
            logger.warning(
                f"Slow request detected: {latency_ms:.0f}ms for question: {request.question[:50]}"
            )
        
        # Format sources
        sources = [
            Source(
                id=s.get("id", "unknown"),
                text=s.get("text", ""),
                score=s.get("score", 0.0),
                metadata=s.get("metadata", {})
            )
            for s in result.get("sources", [])
        ]
        
        # Build observability data
        observability = {
            "run_id": run_id,
            "request_id": req_context.request_id,
            "latency_ms": round(latency_ms, 2),
            "tracing_enabled": langsmith_manager.enabled,
            **cost_data
        }
        
        if langsmith_manager.enabled:
            observability["langsmith_url"] = (
                f"https://smith.langchain.com/o/default/projects/p/{settings.langchain_project}/r/{run_id}"
                if run_id else None
            )
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            iterations=result["iterations"],
            session_id=result.get("session_id"),
            metadata={
                "model": settings.openai_chat_model,
                "framework": "langgraph",
                "persistence_enabled": settings.enable_langgraph_persistence,
                **request.metadata
            },
            observability=observability
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/query/stream")
@trace_agent_call(name="langgraph_query_stream", tags=["query", "stream"])
async def query_agent_stream(request: QueryRequest, http_request: Request):
    """
    Stream responses from the agent in real-time.
    
    Returns Server-Sent Events (SSE) stream.
    Maintains conversation context if session_id is provided.
    All streaming traced in LangSmith.
    """
    
    req_context = create_request_context(http_request, request.user_id)
    
    async def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            start_time = time.time()
            run_id = get_current_run_id()
            
            # Send initial metadata
            initial_event = {
                "type": "start",
                "run_id": run_id,
                "request_id": req_context.request_id,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(initial_event)}\n\n"
            
            # Run the agent stream
            for state in agent.query_stream(request.question, request.session_id):
                # Extract the current node and data
                node_name = list(state.keys())[0]
                node_data = state[node_name]
                
                # Format as SSE
                event_data = {
                    "type": "node",
                    "node": node_name,
                    "iteration": node_data.get("iteration", 0),
                    "timestamp": time.time()
                }
                
                # Add message if available
                if "messages" in node_data and node_data["messages"]:
                    last_msg = node_data["messages"][-1]
                    if hasattr(last_msg, "content"):
                        event_data["content"] = last_msg.content
                
                # Send the event
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Small delay for client processing
                await asyncio.sleep(0.01)
            
            # Calculate total latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Send completion event
            completion_event = {
                "type": "done",
                "latency_ms": round(latency_ms, 2),
                "run_id": run_id
            }
            yield f"data: {json.dumps(completion_event)}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new conversation session."""
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        session_id = agent.conversation_manager.create_session(
            user_id=request.user_id
        )
        
        metadata = agent.conversation_manager.get_session(session_id)
        
        return SessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            created_at=metadata.get("created_at", ""),
            status="active",
            message_count=0
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get metadata for a specific session."""
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        metadata = agent.conversation_manager.get_session(session_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return SessionResponse(
            session_id=session_id,
            user_id=metadata.get("user_id"),
            created_at=metadata.get("created_at", ""),
            updated_at=metadata.get("updated_at"),
            message_count=metadata.get("message_count", 0),
            status=metadata.get("status", "active")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}"
        )


@router.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    limit: int = Query(50, description="Maximum number of messages to return")
):
    """Get conversation history for a session."""
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        metadata = agent.conversation_manager.get_session(session_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        history = agent.get_conversation_history(session_id)
        
        messages = []
        for msg in history[:limit]:
            messages.append(ConversationMessage(
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", ""),
                metadata=msg.get("metadata", {})
            ))
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(history)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session and all its history."""
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        deleted = agent.delete_conversation(session_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.get("/sessions")
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, description="Maximum number of sessions to return")
):
    """List all active sessions."""
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        sessions = agent.list_conversations(user_id=user_id)
        
        result = []
        for session in sessions[:limit]:
            result.append({
                "session_id": session.get("thread_id"),
                "user_id": session.get("last_metadata", {}).get("user_id"),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "message_count": session.get("message_count", 0),
                "status": session.get("status", "active")
            })
        
        return {
            "sessions": result,
            "total": len(result),
            "filtered_by_user": user_id is not None
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


# ============================================================================
# OBSERVABILITY ENDPOINTS (NEW)
# ============================================================================

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback for a query.
    
    Attaches feedback to a specific LangSmith run for quality tracking.
    """
    if not langsmith_manager.enabled:
        raise HTTPException(
            status_code=503,
            detail="Feedback collection not available (LangSmith disabled)"
        )
    
    try:
        success = langsmith_manager.create_feedback(
            run_id=feedback.run_id,
            key=feedback.feedback_type,
            score=feedback.score,
            comment=feedback.comment
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback"
            )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "run_id": feedback.run_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/runs/{run_id}/stats", response_model=RunStatsResponse)
async def get_run_stats(run_id: str):
    """
    Get detailed statistics for a specific LangSmith run.
    
    Includes tokens, cost, latency, and error information.
    """
    if not langsmith_manager.enabled:
        raise HTTPException(
            status_code=503,
            detail="Run statistics not available (LangSmith disabled)"
        )
    
    try:
        stats = langsmith_manager.get_run_stats(run_id)
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )
        
        return RunStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get run statistics: {str(e)}"
        )


@router.get("/runs/recent")
async def list_recent_runs(
    limit: int = Query(20, description="Maximum number of runs to return"),
    filter_str: Optional[str] = Query(None, description="Filter string for runs")
):
    """
    List recent LangSmith runs.
    
    Useful for monitoring recent agent activity.
    """
    if not langsmith_manager.enabled:
        raise HTTPException(
            status_code=503,
            detail="Run listing not available (LangSmith disabled)"
        )
    
    try:
        runs = langsmith_manager.list_recent_runs(
            limit=limit,
            filter_str=filter_str
        )
        
        return {
            "runs": runs,
            "total": len(runs),
            "project": settings.langchain_project
        }
        
    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health/agent")
async def agent_health():
    """
    Check if the LangGraph agent is operational.
    
    Tests agent, persistence, and observability.
    """
    try:
        # Test basic agent functionality
        result = agent.query("ping")
        
        # Check persistence
        persistence_status = "enabled" if settings.enable_langgraph_persistence else "disabled"
        persistence_operational = False
        
        if agent.conversation_manager:
            try:
                test_session = agent.conversation_manager.create_session(user_id="health_check")
                agent.conversation_manager.delete_session(test_session)
                persistence_operational = True
            except Exception as e:
                logger.warning(f"Persistence health check failed: {e}")
        
        # Check observability
        observability_status = {
            "langsmith_enabled": langsmith_manager.enabled,
            "tracing_project": settings.langchain_project if langsmith_manager.enabled else None,
            "feedback_enabled": settings.langsmith_enable_feedback,
            "cost_tracking_enabled": settings.langsmith_enable_cost_tracking
        }
        
        return {
            "status": "ok",
            "agent": "operational",
            "framework": "langgraph",
            "persistence": {
                "status": persistence_status,
                "operational": persistence_operational
            },
            "observability": observability_status,
            "model": settings.openai_chat_model
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "agent": "down",
            "error": str(e)
        }