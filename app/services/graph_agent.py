"""
LangGraph Agent implementation with persistent memory and conversation management.
"""
from typing import Dict, List, Optional, Any, Generator
import logging
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation sessions and history."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        import uuid
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        if self.redis_client:
            metadata = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "message_count": 0
            }
            self.redis_client.set(
                f"session:{session_id}:metadata",
                json.dumps(metadata),
                ex=settings.redis_ttl_seconds
            )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session metadata."""
        if not self.redis_client:
            return None
        
        data = self.redis_client.get(f"session:{session_id}:metadata")
        if data:
            return json.loads(data)
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history."""
        if not self.redis_client:
            return False
        
        # Delete metadata and history
        self.redis_client.delete(f"session:{session_id}:metadata")
        self.redis_client.delete(f"session:{session_id}:history")
        return True


class LangGraphAgent:
    """
    LangGraph-based agent with persistent memory and tool integration.
    """
    
    def __init__(self, enable_persistence: bool = True):
        self.enable_persistence = enable_persistence
        self.conversation_manager = None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )
        
        # Initialize checkpointer for persistence
        if enable_persistence:
            try:
                import redis
                redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True
                )
                # Use MemorySaver for now - Redis checkpointer not available in this version
                self.checkpointer = MemorySaver()
                self.conversation_manager = ConversationManager(redis_client)
                logger.info("LangGraph persistence enabled with in-memory checkpointer and Redis conversation manager")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.checkpointer = MemorySaver()
                logger.info("Using in-memory persistence")
        else:
            self.checkpointer = MemorySaver()
            logger.info("Using in-memory persistence")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        from langgraph.graph import StateGraph
        
        # Define the state
        from typing import TypedDict
        class AgentState(TypedDict):
            messages: List[Any]
            iteration: int
            
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _agent_node(self, state: Dict) -> Dict:
        """Agent reasoning node."""
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0)
        
        # Call LLM
        response = self.llm.invoke(messages)
        
        # Update state
        messages.append(response)
        
        return {
            "messages": messages,
            "iteration": iteration + 1
        }
    
    def _tools_node(self, state: Dict) -> Dict:
        """Tool execution node."""
        # Placeholder for tool execution
        # In a full implementation, this would execute tools based on agent's decision
        return state
    
    def _should_continue(self, state: Dict) -> str:
        """Determine if we should continue or end."""
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0)
        
        # Check if we've hit max iterations
        if iteration >= settings.max_agent_iterations:
            return "end"
        
        # Check if last message has tool calls
        if messages and hasattr(messages[-1], "tool_calls"):
            if messages[-1].tool_calls:
                return "continue"
        
        return "end"
    
    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation continuity
            user_id: Optional user ID for tracking
            
        Returns:
            Dict with answer, sources, iterations, and metadata
        """
        try:
            # Create or get session
            if not session_id and self.conversation_manager:
                session_id = self.conversation_manager.create_session(user_id)
            
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "iteration": 0
            }
            
            # Configure thread
            config = {
                "configurable": {
                    "thread_id": session_id or "default"
                }
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract answer
            messages = final_state.get("messages", [])
            answer = messages[-1].content if messages else "No response generated"
            
            return {
                "success": True,
                "answer": answer,
                "sources": [],  # Placeholder for RAG sources
                "iterations": final_state.get("iteration", 0),
                "session_id": session_id,
                "metadata": {
                    "model": settings.openai_chat_model,
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error in agent query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": [],
                "iterations": 0
            }
    
    def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """
        Stream responses from the agent.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation continuity
            
        Yields:
            State updates as the agent processes the query
        """
        try:
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "iteration": 0
            }
            
            # Configure thread
            config = {
                "configurable": {
                    "thread_id": session_id or "default"
                }
            }
            
            # Stream the graph execution
            for state in self.graph.stream(initial_state, config):
                yield state
                
        except Exception as e:
            logger.error(f"Error in agent stream: {e}", exc_info=True)
            yield {"error": str(e)}
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        # Placeholder - would retrieve from checkpointer
        return []
    
    def list_conversations(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all conversations, optionally filtered by user."""
        # Placeholder - would query Redis for session metadata
        return []
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if self.conversation_manager:
            return self.conversation_manager.delete_session(session_id)
        return False
