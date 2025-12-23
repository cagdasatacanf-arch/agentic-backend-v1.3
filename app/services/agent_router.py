"""
Multi-Agent Router System

Routes queries to specialized agents based on query type:
- MathSpecialist: Calculations, math problems, numerical analysis
- CodeSpecialist: Code generation, debugging, execution
- RAGSpecialist: Knowledge-base queries, research questions
- GeneralAgent: Fallback for other queries

Research basis:
- Specialized agents outperform general agents on domain-specific tasks
- Task-aware routing improves accuracy and efficiency
- Mixture of experts architecture

Benefits:
- Better task-specific performance
- Faster responses (specialized agents are more focused)
- Easier debugging (know which agent handled what)
- Parallel agent execution possible
"""

from typing import Dict, Optional, List, Literal
import logging
from dataclasses import dataclass
from datetime import datetime
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)


AgentType = Literal["math", "code", "rag", "general"]


@dataclass
class RoutingDecision:
    """Decision about which agent to use"""
    agent_type: AgentType
    confidence: float  # 0.0-1.0
    reasoning: str
    detected_keywords: List[str]


class AgentRouter:
    """
    Routes queries to specialized agents.

    Uses rule-based + LLM-based routing for robustness:
    1. Rule-based: Fast keyword matching
    2. LLM-based: Semantic understanding for ambiguous cases

    Usage:
        router = AgentRouter()
        decision = await router.route("Calculate 2+2")
        # decision.agent_type = "math"

        result = await router.execute(decision, "Calculate 2+2")
    """

    def __init__(self, use_llm_routing: bool = True):
        """
        Initialize router.

        Args:
            use_llm_routing: Use LLM for ambiguous queries (recommended)
        """
        self.use_llm_routing = use_llm_routing

        if use_llm_routing:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Fast for routing
                temperature=0,
                api_key=settings.openai_api_key
            )

        # Keyword patterns for rule-based routing
        self.patterns = {
            "math": [
                r'\b(calculate|compute|solve|evaluate|math|equation|formula)\b',
                r'\b(\d+\s*[\+\-\*/\^]\s*\d+)',  # Arithmetic expressions
                r'\b(sqrt|sin|cos|tan|log|exp|factorial)\b',
                r'\b(what is|how much is)\s+\d+',
                r'\b(sum|product|difference|quotient|average|mean|median)\b'
            ],
            "code": [
                r'\b(code|python|javascript|function|class|script|program)\b',
                r'\b(write|implement|create|build|develop)\s+(a|an|the)?\s*(function|script|program)',
                r'\b(debug|fix|error|bug|syntax)\b',
                r'\b(import|def|class|return|if|for|while)\b',
                r'```\w*\n',  # Code blocks
            ],
            "rag": [
                r'\b(what is|what are|explain|describe|tell me about|define)\b',
                r'\b(who|when|where|why|how)\b',
                r'\b(document|information|knowledge|search|find|look up)\b',
                r'\b(according to|based on|from|in)\s+(the|our)?\s*(documents?|database|knowledge base)\b'
            ]
        }

        logger.info(f"AgentRouter initialized (LLM routing: {use_llm_routing})")

    def rule_based_route(self, query: str) -> Optional[RoutingDecision]:
        """
        Fast rule-based routing using keyword patterns.

        Args:
            query: User query

        Returns:
            RoutingDecision or None if no clear match
        """
        query_lower = query.lower()
        scores = {agent_type: 0 for agent_type in ["math", "code", "rag"]}
        detected = {agent_type: [] for agent_type in ["math", "code", "rag"]}

        # Score each agent type
        for agent_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    scores[agent_type] += len(matches)
                    detected[agent_type].extend([m if isinstance(m, str) else m[0] for m in matches])

        # Get best match
        if max(scores.values()) > 0:
            best_agent = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[best_agent] / total_score if total_score > 0 else 0.0

            # High confidence threshold for rule-based
            if confidence >= 0.6:
                return RoutingDecision(
                    agent_type=best_agent,
                    confidence=confidence,
                    reasoning=f"Rule-based: {scores[best_agent]} keyword matches",
                    detected_keywords=detected[best_agent][:5]  # Top 5
                )

        return None

    async def llm_based_route(self, query: str) -> RoutingDecision:
        """
        LLM-based routing for ambiguous queries.

        Args:
            query: User query

        Returns:
            RoutingDecision
        """
        prompt = f"""You are a query router. Determine which specialized agent should handle this query:

Query: {query}

Agents:
- math: Mathematical calculations, equations, numerical problems
- code: Programming, code generation, debugging, scripting
- rag: Knowledge retrieval, research questions, explanations from documents
- general: Everything else (conversations, opinions, creative writing)

Respond with ONLY the agent type: math, code, rag, or general
No explanation, just the type."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            agent_type_str = response.content.strip().lower()

            # Validate response
            valid_types = ["math", "code", "rag", "general"]
            if agent_type_str in valid_types:
                return RoutingDecision(
                    agent_type=agent_type_str,
                    confidence=0.8,  # LLM routing confidence
                    reasoning="LLM-based semantic routing",
                    detected_keywords=[]
                )
            else:
                logger.warning(f"Invalid LLM routing response: {agent_type_str}")
                return RoutingDecision(
                    agent_type="general",
                    confidence=0.5,
                    reasoning="Fallback: invalid LLM response",
                    detected_keywords=[]
                )

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return RoutingDecision(
                agent_type="general",
                confidence=0.5,
                reasoning="Fallback: LLM routing error",
                detected_keywords=[]
            )

    async def route(self, query: str, verbose: bool = False) -> RoutingDecision:
        """
        Route query to appropriate agent.

        Uses rule-based first (fast), falls back to LLM if ambiguous.

        Args:
            query: User query
            verbose: Log routing decision

        Returns:
            RoutingDecision
        """
        # Try rule-based first
        decision = self.rule_based_route(query)

        if decision:
            if verbose:
                logger.info(
                    f"Route: {decision.agent_type} (confidence: {decision.confidence:.2f}, "
                    f"method: rule-based)"
                )
            return decision

        # Fall back to LLM-based
        if self.use_llm_routing:
            decision = await self.llm_based_route(query)
            if verbose:
                logger.info(
                    f"Route: {decision.agent_type} (confidence: {decision.confidence:.2f}, "
                    f"method: LLM-based)"
                )
            return decision

        # Ultimate fallback
        if verbose:
            logger.info("Route: general (fallback)")

        return RoutingDecision(
            agent_type="general",
            confidence=0.5,
            reasoning="Fallback: no routing method available",
            detected_keywords=[]
        )

    async def execute(
        self,
        decision: RoutingDecision,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Execute query with the routed agent.

        Args:
            decision: Routing decision
            query: User query
            session_id: Optional session ID

        Returns:
            Result dict with answer, agent_used, etc.
        """
        agent_type = decision.agent_type

        logger.info(f"Executing with {agent_type} agent: {query[:60]}...")

        try:
            if agent_type == "math":
                from app.services.agents.math_agent import MathSpecialist
                agent = MathSpecialist()
                result = await agent.solve(query)

            elif agent_type == "code":
                from app.services.agents.code_agent import CodeSpecialist
                agent = CodeSpecialist()
                result = await agent.generate(query)

            elif agent_type == "rag":
                from app.services.agents.rag_agent import RAGSpecialist
                agent = RAGSpecialist()
                result = await agent.query(query, session_id=session_id)

            else:  # general
                from app.services.graph_agent import LangGraphAgent
                agent = LangGraphAgent()
                result = agent.query(query, session_id=session_id)
                # Adapt format
                result = {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "agent_type": "general"
                }

            # Add routing metadata
            result["agent_used"] = agent_type
            result["routing_confidence"] = decision.confidence
            result["routing_reasoning"] = decision.reasoning

            return result

        except Exception as e:
            logger.error(f"Agent execution failed ({agent_type}): {e}", exc_info=True)
            return {
                "answer": f"Error: Agent execution failed - {str(e)}",
                "agent_used": agent_type,
                "error": str(e),
                "success": False
            }

    async def route_and_execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Convenience method: route + execute in one call.

        Args:
            query: User query
            session_id: Optional session ID
            verbose: Log routing

        Returns:
            Result dict
        """
        decision = await self.route(query, verbose=verbose)
        result = await self.execute(decision, query, session_id=session_id)

        if verbose:
            logger.info(
                f"Completed: agent={result.get('agent_used')}, "
                f"success={result.get('success', True)}"
            )

        return result


# ============================================================================
# Singleton Instance
# ============================================================================

_router: Optional[AgentRouter] = None


def get_router() -> AgentRouter:
    """Get or create global router instance"""
    global _router
    if _router is None:
        _router = AgentRouter(use_llm_routing=True)
    return _router
