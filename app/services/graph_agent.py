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
from app.services.tool_metrics import ToolMetricsCollector, ToolExecution, initialize_metrics_collector
from app.config import settings
from datetime import datetime
import time

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
# INSTRUMENTED TOOL NODE (with metrics tracking)
# ============================================================================

class InstrumentedToolNode:
    """
    Tool node wrapper that tracks execution metrics.

    Wraps LangGraph's ToolNode to log:
    - Success/failure rate
    - Execution latency
    - Error messages

    Based on ToolLLM research for tool execution adaptation.
    """

    def __init__(self, tools: list, metrics_collector: Optional[ToolMetricsCollector] = None):
        """
        Initialize instrumented tool node.

        Args:
            tools: List of LangChain tools
            metrics_collector: Optional metrics collector (if None, no tracking)
        """
        self.tools = tools
        self.metrics = metrics_collector
        self.base_node = ToolNode(tools)

    def __call__(self, state: AgentState) -> AgentState:
        """
        Execute tools with metrics tracking.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]

        # If no tool calls, pass through
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return self.base_node(state)

        # Track each tool call
        session_id = state.get("session_id", "unknown")

        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})

            start_time = time.perf_counter()
            success = True
            error_msg = None
            output = None

            try:
                # Execute the tool via base node
                result_state = self.base_node(state)

                # Extract output from result messages
                if "messages" in result_state:
                    tool_messages = result_state["messages"]
                    if tool_messages:
                        output = str(tool_messages[-1].content) if hasattr(tool_messages[-1], "content") else str(tool_messages[-1])

            except Exception as e:
                success = False
                error_msg = str(e)
                logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
                raise

            finally:
                # Always log metrics (even on failure)
                latency_ms = (time.perf_counter() - start_time) * 1000

                if self.metrics:
                    try:
                        execution = ToolExecution(
                            tool_name=tool_name,
                            session_id=session_id,
                            timestamp=datetime.now(),
                            success=success,
                            latency_ms=latency_ms,
                            error_message=error_msg,
                            input_params=tool_args,
                            output=output[:500] if output else None  # Truncate long outputs
                        )
                        self.metrics.log_execution(execution)
                    except Exception as metrics_error:
                        # Never let metrics collection break the tool execution
                        logger.warning(f"Failed to log metrics: {metrics_error}")

        # Return the result from base node execution
        try:
            return self.base_node(state)
        except Exception as e:
            # If we already raised above, this won't execute
            # This handles edge cases
            logger.error(f"Tool execution failed: {e}")
            return state


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

def create_graph(with_checkpointer: bool = True, metrics_collector: Optional[ToolMetricsCollector] = None):
    """
    Create the LangGraph state machine with optional persistence and metrics.

    Args:
        with_checkpointer: If True, enable Redis-based persistence
        metrics_collector: Optional tool metrics collector for tracking

    Flow:
    1. START → rag_node (optional: retrieve context)
    2. rag_node → agent (reasoning with tools)
    3. agent → should_continue (router)
    4. should_continue → tools OR end
    5. tools → agent (loop back for more reasoning)
    """

    # Create the graph
    workflow = StateGraph(AgentState)

    # Create instrumented tool node (with metrics if provided)
    tool_node = InstrumentedToolNode(tools, metrics_collector=metrics_collector)

    # Add nodes
    workflow.add_node("rag", rag_node)  # RAG retrieval
    workflow.add_node("agent", call_model)  # Main reasoning
    workflow.add_node("tools", tool_node)  # Tool execution with metrics
    
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
    """High-level interface for the LangGraph agent with persistent memory and metrics."""

    def __init__(self, enable_persistence: bool = True, enable_metrics: bool = True):
        """
        Initialize the agent.

        Args:
            enable_persistence: Enable Redis-based conversation memory
            enable_metrics: Enable tool execution metrics tracking
        """
        self.enable_persistence = enable_persistence
        self.enable_metrics = enable_metrics
        self.conversation_manager = None
        self.metrics_collector = None

        # Initialize Redis client (used for both persistence and metrics)
        redis_client = None
        if enable_persistence or enable_metrics:
            try:
                redis_client = get_redis_client()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

        # Initialize conversation manager
        if enable_persistence and redis_client:
            try:
                self.conversation_manager = ConversationManager(redis_client)
                logger.info("Agent initialized with persistent memory")
            except Exception as e:
                logger.error(f"Failed to initialize conversation manager: {e}")
                self.enable_persistence = False

        # Initialize metrics collector
        if enable_metrics and redis_client:
            try:
                self.metrics_collector = initialize_metrics_collector(redis_client)
                logger.info("Agent initialized with metrics tracking")
            except Exception as e:
                logger.error(f"Failed to initialize metrics collector: {e}")
                self.enable_metrics = False

        # Create graph with metrics collector
        self.graph = create_graph(
            with_checkpointer=enable_persistence,
            metrics_collector=self.metrics_collector
        )
    
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

    def get_tool_metrics(self, tool_name: Optional[str] = None, last_n: int = 100) -> dict:
        """
        Get tool execution metrics.

        Args:
            tool_name: Specific tool name, or None for all tools
            last_n: Number of recent executions to analyze

        Returns:
            Dict with metrics summary
        """
        if not self.metrics_collector:
            return {"error": "Metrics collection not enabled"}

        if tool_name:
            # Get metrics for specific tool
            return {
                "tool_name": tool_name,
                "success_rate": self.metrics_collector.get_success_rate(tool_name, last_n),
                "latency_stats": self.metrics_collector.get_latency_stats(tool_name, last_n),
                "quality_score": self.metrics_collector.get_tool_quality_score(tool_name, last_n),
                "error_summary": self.metrics_collector.get_error_summary(tool_name, last_n)
            }
        else:
            # Get summary for all tools
            return {
                "tools": self.metrics_collector.get_all_tools_summary(last_n)
            }


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