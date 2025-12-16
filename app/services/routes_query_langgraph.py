"""
FastAPI routes with LangGraph integration.
Place this in: app/api/routes_query_langgraph.py
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import json

from app.services.graph_agent import LangGraphAgent

router = APIRouter(prefix="/api/v1", tags=["query"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for agent queries."""
    question: str = Field(..., description="The question to ask the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    stream: bool = Field(False, description="Whether to stream the response")
    max_iterations: int = Field(10, description="Maximum reasoning iterations", ge=1, le=20)


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


# ============================================================================
# GLOBAL AGENT INSTANCE (or use dependency injection)
# ============================================================================

# Initialize once at startup
agent = LangGraphAgent()


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the LangGraph agent with a question.
    
    The agent will:
    1. Retrieve relevant documents from RAG
    2. Use available tools (calculator, web search)
    3. Reason through the answer
    4. Return response with sources
    """
    
    try:
        # Run the agent
        result = agent.query(
            question=request.question,
            session_id=request.session_id
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
            session_id=request.session_id,
            metadata={
                "model": "gpt-4o",
                "framework": "langgraph"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Stream responses from the agent in real-time.
    
    Returns Server-Sent Events (SSE) stream.
    """
    
    async def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            # Run the agent stream
            for state in agent.query_stream(request.question):
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
        }
    )


@router.get("/query/status/{session_id}")
async def get_session_status(session_id: str):
    """
    Get the status of a conversation session.
    
    TODO: Implement session storage with checkpointing.
    """
    # This would query your session store
    return {
        "session_id": session_id,
        "status": "active",
        "message_count": 0
    }


# ============================================================================
# ADVANCED: Multi-Agent with Different Personalities
# ============================================================================

class AgentType(BaseModel):
    """Available agent types."""
    type: str = Field(..., description="Agent type: 'researcher', 'coder', 'writer'")


@router.post("/query/multi-agent")
async def query_multi_agent(request: QueryRequest, agent_type: AgentType):
    """
    Query with a specialized agent personality.
    
    Different agents have different tools and system prompts:
    - researcher: Focus on RAG and web search
    - coder: Focus on code generation and debugging
    - writer: Focus on creative content generation
    """
    
    # TODO: Create different graph configurations per agent type
    # For now, use the default agent
    
    result = agent.query(
        question=request.question,
        session_id=request.session_id
    )
    
    return {
        "agent_type": agent_type.type,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "iterations": result["iterations"]
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health/agent")
async def agent_health():
    """Check if the LangGraph agent is operational."""
    try:
        # Simple test query
        result = agent.query("ping")
        return {
            "status": "ok",
            "agent": "operational",
            "framework": "langgraph"
        }
    except Exception as e:
        return {
            "status": "error",
            "agent": "down",
            "error": str(e)
        }