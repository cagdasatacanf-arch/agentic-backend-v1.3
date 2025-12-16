"""
Advanced multi-agent system with supervisor pattern.
Demonstrates: parallel execution, agent delegation, human-in-the-loop.

Place this in: app/services/multi_agent_system.py
"""

from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import json


# ============================================================================
# SHARED STATE
# ============================================================================

class MultiAgentState(TypedDict):
    """Shared state across all agents."""
    messages: List[BaseMessage]
    next_agent: str  # Which agent to route to
    task_complete: bool
    results: dict  # Results from each agent
    human_feedback: str  # For human-in-the-loop


# ============================================================================
# SPECIALIZED TOOLS
# ============================================================================

@tool
def search_technical_docs(query: str) -> str:
    """Search technical documentation and code repositories."""
    return f"Technical docs for: {query}"


@tool
def analyze_data(data: str) -> str:
    """Perform statistical analysis on data."""
    return f"Analysis of: {data}"


@tool
def generate_code(specification: str) -> str:
    """Generate code based on specifications."""
    return f"# Code for: {specification}\ndef solution():\n    pass"


@tool
def write_content(prompt: str) -> str:
    """Generate written content like articles, emails, reports."""
    return f"Content: {prompt}"


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class ResearchAgent:
    """Agent specialized in research and information gathering."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [search_technical_docs]
        self.system_prompt = """You are a research specialist.
Your job is to gather comprehensive information on topics.
Use search_technical_docs to find relevant information.
Be thorough and cite sources."""
    
    def process(self, state: MultiAgentState) -> dict:
        """Process the research task."""
        messages = state["messages"]
        
        # Add system prompt
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # Call LLM with tools
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {
            "messages": [response],
            "results": {**state.get("results", {}), "research": response.content}
        }


class DataAnalyst:
    """Agent specialized in data analysis."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [analyze_data]
        self.system_prompt = """You are a data analyst.
Your job is to analyze data and provide insights.
Use analyze_data for statistical computations.
Present findings clearly with visualizations if needed."""
    
    def process(self, state: MultiAgentState) -> dict:
        messages = state["messages"]
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {
            "messages": [response],
            "results": {**state.get("results", {}), "analysis": response.content}
        }


class CodingAgent:
    """Agent specialized in code generation and debugging."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [generate_code, search_technical_docs]
        self.system_prompt = """You are a senior software engineer.
Your job is to write clean, efficient, well-documented code.
Use generate_code to create implementations.
Follow best practices and include tests."""
    
    def process(self, state: MultiAgentState) -> dict:
        messages = state["messages"]
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {
            "messages": [response],
            "results": {**state.get("results", {}), "code": response.content}
        }


class WriterAgent:
    """Agent specialized in content creation."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.tools = [write_content]
        self.system_prompt = """You are a professional writer.
Your job is to create engaging, clear, well-structured content.
Adapt your tone to the audience and purpose."""
    
    def process(self, state: MultiAgentState) -> dict:
        messages = state["messages"]
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(full_messages)
        
        return {
            "messages": [response],
            "results": {**state.get("results", {}), "content": response.content}
        }


# ============================================================================
# SUPERVISOR AGENT
# ============================================================================

class SupervisorAgent:
    """
    Supervisor that routes tasks to specialized agents.
    Implements the "manager pattern" for multi-agent orchestration.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.agents = ["research", "data_analyst", "coder", "writer"]
    
    def route(self, state: MultiAgentState) -> str:
        """
        Decide which agent should handle the task next.
        """
        messages = state["messages"]
        
        system_prompt = f"""You are a supervisor managing a team of specialists:
- research: For gathering information and research
- data_analyst: For data analysis and statistics
- coder: For writing and debugging code
- writer: For creating written content

Given the conversation, decide which agent should work on this next.
If the task is complete, respond with 'FINISH'.

Respond with only: {', '.join(self.agents)} or FINISH"""
        
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        response = self.llm.invoke(full_messages)
        next_agent = response.content.strip().lower()
        
        # Validate response
        if next_agent == "finish":
            return "finish"
        elif next_agent in self.agents:
            return next_agent
        else:
            return "research"  # Default fallback


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_multi_agent_graph():
    """
    Create a multi-agent graph with supervisor routing.
    
    Flow:
    1. User input → supervisor
    2. Supervisor decides which specialist to use
    3. Specialist processes and returns to supervisor
    4. Supervisor decides: more work OR finish
    """
    
    # Initialize agents
    supervisor = SupervisorAgent()
    research = ResearchAgent()
    data_analyst = DataAnalyst()
    coder = CodingAgent()
    writer = WriterAgent()
    
    # Create workflow
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("supervisor", lambda state: {
        "next_agent": supervisor.route(state)
    })
    workflow.add_node("research", research.process)
    workflow.add_node("data_analyst", data_analyst.process)
    workflow.add_node("coder", coder.process)
    workflow.add_node("writer", writer.process)
    workflow.add_node("human_review", lambda state: state)  # Pause for human feedback
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Define routing logic
    def route_supervisor(state: MultiAgentState) -> str:
        """Route based on supervisor's decision."""
        next_agent = state.get("next_agent", "finish")
        
        if next_agent == "finish":
            return "human_review"
        return next_agent
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "data_analyst": "data_analyst",
            "coder": "coder",
            "writer": "writer",
            "human_review": "human_review"
        }
    )
    
    # All agents return to supervisor
    for agent in ["research", "data_analyst", "coder", "writer"]:
        workflow.add_edge(agent, "supervisor")
    
    # Human review can approve or send back
    workflow.add_conditional_edges(
        "human_review",
        lambda state: "end" if state.get("task_complete") else "supervisor",
        {
            "supervisor": "supervisor",
            "end": END
        }
    )
    
    # Compile with memory for checkpointing
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

class MultiAgentSystem:
    """High-level interface for the multi-agent system."""
    
    def __init__(self):
        self.graph = create_multi_agent_graph()
    
    def query(self, question: str, thread_id: str = "default") -> dict:
        """
        Run a query through the multi-agent system.
        
        Args:
            question: User's question/task
            thread_id: Conversation thread ID for memory persistence
        
        Returns:
            Results from all agents that worked on the task
        """
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "next_agent": "",
            "task_complete": False,
            "results": {},
            "human_feedback": ""
        }
        
        # Run the graph
        final_state = None
        for state in self.graph.stream(initial_state, config):
            final_state = state
            print(f"Current node: {list(state.keys())[0]}")
        
        # Extract results
        if final_state:
            node_data = final_state[list(final_state.keys())[-1]]
            return {
                "answer": node_data["messages"][-1].content if node_data["messages"] else "",
                "results": node_data.get("results", {}),
                "success": True
            }
        
        return {"answer": "No response", "results": {}, "success": False}
    
    def approve_and_continue(self, thread_id: str, approved: bool = True):
        """
        Continue from human review checkpoint.
        
        Args:
            thread_id: Thread to continue
            approved: Whether the human approved the result
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Update state to continue
        update_state = {
            "task_complete": approved,
            "human_feedback": "Approved" if approved else "Needs revision"
        }
        
        self.graph.update_state(config, update_state)
        
        # Continue execution
        for state in self.graph.stream(None, config):
            print(f"Continued to: {list(state.keys())[0]}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    system = MultiAgentSystem()
    
    # Complex task requiring multiple agents
    result = system.query(
        "Research the latest trends in LLM agents, analyze adoption metrics, "
        "generate a Python implementation example, and write a summary report.",
        thread_id="demo-001"
    )
    
    print("\n=== Final Results ===")
    print(json.dumps(result, indent=2))