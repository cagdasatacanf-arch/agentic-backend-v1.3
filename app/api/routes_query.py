"""
FastAPI routes with LangGraph integration and session management.
Place this in: app/api/routes_query_langgraph.py
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import json
import logging

from app.services.graph_agent import LangGraphAgent
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


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

# Initialize once at startup
agent = LangGraphAgent(enable_persistence=settings.enable_langgraph_persistence)


# ============================================================================
# QUERY ENDPOINTS
# ============================================================================

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the LangGraph agent with a question.
    
    The agent will:
    1. Retrieve relevant documents from RAG
    2. Use available tools (calculator, web search)
    3. Reason through the answer with persistent memory
    4. Return response with sources
    
    If session_id is not provided, a new session will be created.
    """
    
    try:
        # Run the agent
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
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            iterations=result["iterations"],
            session_id=result.get("session_id"),
            metadata={
                "model": settings.openai_chat_model,
                "framework": "langgraph",
                "persistence_enabled": settings.enable_langgraph_persistence
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Stream responses from the agent in real-time.
    
    Returns Server-Sent Events (SSE) stream.
    Maintains conversation context if session_id is provided.
    """
    
    async def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            # Run the agent stream
            for state in agent.query_stream(request.question, request.session_id):
                # Extract the current node and data
                node_name = list(state.keys())[0]
                node_data = state[node_name]
                
                # Format as SSE
                event_data = {
                    "type": "node",
                    "node": node_name,
                    "iteration": node_data.get("iteration", 0)
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
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new conversation session.
    
    Returns a session_id that can be used for subsequent queries
    to maintain conversation context.
    """
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        session_id = agent.conversation_manager.create_session(
            user_id=request.user_id
        )
        
        # Get session metadata
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
    """
    Get metadata for a specific session.
    
    Returns session information including message count and timestamps.
    """
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
    """
    Get conversation history for a session.
    
    Returns all messages in chronological order.
    """
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        # Check if session exists
        metadata = agent.conversation_manager.get_session(session_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Get history
        history = agent.get_conversation_history(session_id)
        
        # Format messages
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
    """
    Delete a conversation session and all its history.
    
    This action is permanent and cannot be undone.
    """
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
    """
    List all active sessions.
    
    Optionally filter by user_id to get sessions for a specific user.
    """
    if not agent.conversation_manager:
        raise HTTPException(
            status_code=503,
            detail="Session management not available (persistence disabled)"
        )
    
    try:
        sessions = agent.list_conversations(user_id=user_id)
        
        # Format response
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
# HEALTH CHECK
# ============================================================================

@router.get("/health/agent")
async def agent_health():
    """
    Check if the LangGraph agent is operational.
    
    Tests both the agent and persistence layer.
    """
    try:
        # Test basic agent functionality
        result = agent.query("ping")
        
        # Check persistence
        persistence_status = "enabled" if settings.enable_langgraph_persistence else "disabled"
        persistence_operational = False
        
        if agent.conversation_manager:
            try:
                # Try to create and delete a test session
                test_session = agent.conversation_manager.create_session(user_id="health_check")
                agent.conversation_manager.delete_session(test_session)
                persistence_operational = True
            except Exception as e:
                logger.warning(f"Persistence health check failed: {e}")
        
        return {
            "status": "ok",
            "agent": "operational",
            "framework": "langgraph",
            "persistence": {
                "status": persistence_status,
                "operational": persistence_operational
            },
            "model": settings.openai_chat_model
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "agent": "down",
            "error": str(e)
        }