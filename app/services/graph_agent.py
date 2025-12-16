"""
LangGraph-based multi-agent system with RAG, tools, and persistent memory.
Place this in: app/services/graph_agent.py
"""

from typing import Annotated, Literal, TypedDict, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import operator
import json
import logging
import redis

from app.services.redis_checkpointer import RedisCheckpointSaver, ConversationManager
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# REDIS CONNECTION
# ============================================================================

def get_redis_client() -> redis.Redis:
    """Get Redis client for checkpointing."""
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password if settings.redis_password else None,
        decode_responses=False  # We handle encoding/decoding
    )


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str  # RAG context
    next: str  # Next node to route to
    iteration: int  # Track iterations for loop control
    sources: list  # Document sources


# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def search_documents(query: str) -> str:
    """
    Search the vector database for relevant documents.
    Use this when you need information from the knowledge base.
    """
    # TODO: Replace with your actual Qdrant RAG implementation
    # from app.rag import RAGService
    # rag_service = RAGService()
    # results = rag_service.search(query, top_k=5)
    # formatted = [{"text": r.text, "score": r.score, "id": r.id} for r in results]
    # return json.dumps(formatted, indent=2)
    
    # Mock response for now
    results = [
        {"text": f"Document about {query}", "score": 0.9, "id": "doc-1"},
        {"text": f"Related info on {query}", "score": 0.8, "id": "doc-2"}
    ]
    
    return json.dumps(results, indent=2)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for calculations like "2 + 2" or "sqrt(16)".
    """
    try:
        # Safe eval for basic math
        import math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use when the knowledge base doesn't have the answer or for recent events.
    """
    # TODO: Integrate with Tavily, SerpAPI, or similar
    return f"Web search results for: {query}\n[Mock - integrate real API]"


# Tool collection
tools = [search_documents, calculator, web_search]


# ============================================================================
# AGENT NODES
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Router function - decides whether to use tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check iteration limit to prevent infinite loops
    if state.get("iteration", 0) >= 10:
        logger.warning(f"Max iterations reached for conversation")
        return "end"
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise end
    return "end"


def call_model(state: AgentState) -> AgentState:
    """
    Main agent node - calls the LLM with tools.
    """
    messages = state["messages"]
    
    # Initialize LLM with tools
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        api_key=settings.openai_api_key
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # Add system message if first turn
    if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(content="""You are a helpful AI assistant with access to:
- A document knowledge base (search_documents)
- A calculator (calculator)
- Web search (web_search)

Use these tools when needed to provide accurate, helpful responses.
Always cite sources when using search_documents.
If you've already answered the question, don't call tools again - just respond.""")
        messages = [system_msg] + list(messages)
    
    # Call the model
    response = llm_with_tools.invoke(messages)
    
    # Update iteration counter
    current_iteration = state.get("iteration", 0) + 1
    
    logger.info(f"Agent iteration {current_iteration}, response length: {len(response.content) if response.content else 0}")
    
    return {
        "messages": [response],
        "iteration": current_iteration
    }


def rag_node(state: AgentState) -> AgentState:
    """
    Dedicated RAG node - retrieves context before agent reasoning.
    Use this for always-on RAG retrieval.
    """
    messages = state["messages"]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )
    
    # TODO: Replace with actual RAG retrieval
    # from app.rag import RAGService
    # rag_service = RAGService()
    # context_results = rag_service.search(last_user_msg, top_k=5)
    
    # Mock context
    context_results = [
        {"text": "Relevant document passage", "score": 0.9, "id": "doc-1"}
    ]
    
    context = "\n\n".join([r["text"] for r in context_results])
    
    logger.info(f"RAG retrieved {len(context_results)} documents")
    
    # Add context as system message
    context_msg = SystemMessage(
        content=f"Here is relevant context from the knowledge base:\n\n{context}"
    )
    
    return {
        "messages": [context_msg],
        "context": context,
        "sources": context_results
    }


# ============================================================================
# GRAPH CONSTRUCTION WITH PERSISTENCE
# ============================================================================

def create_graph(with_checkpointer: bool = True):
    """
    Create the LangGraph state machine with optional persistence.
    
    Args:
        with_checkpointer: If True, enable Redis-based persistence
    
    Flow:
    1. START → rag_node (optional: retrieve context)
    2. rag_node → agent (reasoning with tools)
    3. agent → should_continue (router)
    4. should_continue → tools OR end
    5. tools → agent (loop back for more reasoning)
    """
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("rag", rag_node)  # RAG retrieval
    workflow.add_node("agent", call_model)  # Main reasoning
    workflow.add_node("tools", ToolNode(tools))  # Tool execution
    
    # Define edges
    workflow.set_entry_point("rag")  # Start with RAG
    workflow.add_edge("rag", "agent")  # RAG → agent
    
    # Conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph with optional checkpointer
    if with_checkpointer:
        try:
            redis_client = get_redis_client()
            checkpointer = RedisCheckpointSaver(redis_client)
            app = workflow.compile(checkpointer=checkpointer)
            logger.info("Graph compiled with Redis checkpointer")
        except Exception as e:
            logger.error(f"Failed to initialize Redis checkpointer: {e}")
            logger.warning("Falling back to in-memory graph (no persistence)")
            app = workflow.compile()
    else:
        app = workflow.compile()
        logger.info("Graph compiled without persistence")
    
    return app


# ============================================================================
# USAGE INTERFACE WITH PERSISTENCE
# ============================================================================

class LangGraphAgent:
    """High-level interface for the LangGraph agent with persistent memory."""
    
    def __init__(self, enable_persistence: bool = True):
        """
        Initialize the agent.
        
        Args:
            enable_persistence: Enable Redis-based conversation memory
        """
        self.graph = create_graph(with_checkpointer=enable_persistence)
        self.enable_persistence = enable_persistence
        
        if enable_persistence:
            try:
                redis_client = get_redis_client()
                self.conversation_manager = ConversationManager(redis_client)
                logger.info("Agent initialized with persistent memory")
            except Exception as e:
                logger.error(f"Failed to initialize conversation manager: {e}")
                self.conversation_manager = None
                self.enable_persistence = False
        else:
            self.conversation_manager = None
    
    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> dict:
        """
        Query the agent with a question.
        
        Args:
            question: User's question
            session_id: Optional session ID for conversation continuity
            user_id: Optional user ID for tracking
            
        Returns:
            dict with answer, sources, and metadata
        """
        
        # Create or use existing session
        if self.enable_persistence and not session_id:
            session_id = self.conversation_manager.create_session(user_id=user_id)
            logger.info(f"Created new session: {session_id}")
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "context": "",
            "iteration": 0,
            "sources": []
        }
        
        # Prepare config for persistence
        config = None
        if self.enable_persistence and session_id:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id
                }
            }
        
        # Run the graph
        try:
            # Stream through the graph
            final_state = None
            for state in self.graph.stream(initial_state, config):
                final_state = state
            
            # Extract final response
            if not final_state:
                raise ValueError("No state returned from graph")
            
            messages = final_state[list(final_state.keys())[-1]]["messages"]
            last_message = messages[-1]
            
            # Get the AI's final answer
            answer = last_message.content if hasattr(last_message, "content") else str(last_message)
            
            result = {
                "answer": answer,
                "sources": final_state[list(final_state.keys())[-1]].get("sources", []),
                "iterations": final_state[list(final_state.keys())[-1]].get("iteration", 0),
                "session_id": session_id,
                "success": True
            }
            
            logger.info(f"Query completed successfully for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "iterations": 0,
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    def query_stream(self, question: str, session_id: Optional[str] = None):
        """
        Stream responses from the agent.
        Useful for real-time UI updates.
        """
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "context": "",
            "iteration": 0,
            "sources": []
        }
        
        config = None
        if self.enable_persistence and session_id:
            config = {"configurable": {"thread_id": session_id}}
        
        for state in self.graph.stream(initial_state, config):
            yield state
    
    def get_conversation_history(self, session_id: str) -> list:
        """Get full conversation history for a session."""
        if not self.conversation_manager:
            return []
        
        return self.conversation_manager.get_conversation_history(session_id)
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if not self.conversation_manager:
            return False
        
        return self.conversation_manager.delete_session(session_id)
    
    def list_conversations(self, user_id: Optional[str] = None) -> list:
        """List all conversations, optionally filtered by user."""
        if not self.conversation_manager:
            return []
        
        return self.conversation_manager.list_sessions(user_id=user_id)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create agent with persistence
    agent = LangGraphAgent(enable_persistence=True)
    
    # First message
    result1 = agent.query("What is 25 * 4?")
    print("Q1:", result1["answer"])
    print("Session:", result1["session_id"])
    
    # Follow-up in same conversation
    result2 = agent.query(
        "Add 100 to that result",
        session_id=result1["session_id"]
    )
    print("Q2:", result2["answer"])
    
    # Get history
    history = agent.get_conversation_history(result1["session_id"])
    print("History:", len(history), "messages")