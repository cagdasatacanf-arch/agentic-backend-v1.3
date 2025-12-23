"""
Math Specialist Agent

Specialized agent for mathematical calculations with:
- Step-by-step reasoning
- Multiple solution methods
- Verification of results
- Support for complex math (calculus, algebra, statistics)

Based on research:
- Chain-of-thought prompting improves math accuracy
- Specialized agents outperform general agents on math tasks
- Verification catches calculation errors
"""

from typing import Dict, List, Optional
import logging
import math
import re
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.services.interaction_logger import log_interaction
from app.services.output_quality import OutputQualityScorer

logger = logging.getLogger(__name__)


class MathSpecialist:
    """
    Specialized agent for mathematical problems.

    Features:
    - Step-by-step reasoning (chain-of-thought)
    - Python code execution for calculations
    - Verification of results
    - Support for algebra, calculus, statistics

    Usage:
        agent = MathSpecialist()
        result = await agent.solve("Calculate (25 * 4) + sqrt(144)")
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize math specialist.

        Args:
            model: LLM to use (gpt-4o recommended for math)
        """
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Deterministic for math
            api_key=settings.openai_api_key
        )

        # Safe math environment for eval
        self.safe_env = {
            "__builtins__": {},
            **{k: v for k, v in math.__dict__.items() if not k.startswith("__")},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        logger.info(f"MathSpecialist initialized with {model}")

    async def solve(self, problem: str, show_steps: bool = True, session_id: Optional[str] = None) -> Dict:
        """
        Solve a mathematical problem.

        Args:
            problem: Math problem description
            show_steps: Include step-by-step reasoning
            session_id: Optional session ID for logging

        Returns:
            {
                "answer": "...",
                "steps": [...],
                "calculation": "...",
                "verification": {...},
                "agent_type": "math"
            }
        """
        logger.info(f"Solving math problem: {problem[:60]}...")
        start_time = time.perf_counter()

        # Build prompt with chain-of-thought
        system_prompt = """You are a mathematical expert. Solve problems step-by-step.

Format your response as:
REASONING: [Step-by-step explanation]
CALCULATION: [Python code to calculate, if applicable]
ANSWER: [Final numerical answer]

For calculations, write Python code that can be executed.
Be precise and show your work."""

        user_prompt = f"Problem: {problem}"

        error_occurred = False
        error_message = None

        try:
            # Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            content = response.content

            # Parse response
            reasoning = self._extract_section(content, "REASONING")
            calculation_code = self._extract_section(content, "CALCULATION")
            answer_text = self._extract_section(content, "ANSWER")

            # Execute calculation if provided
            calculated_result = None
            if calculation_code:
                calculated_result = self._safe_eval(calculation_code)

            # Verification
            verification = self._verify_answer(
                problem=problem,
                answer=answer_text,
                calculated=calculated_result
            )

            # Build steps
            steps = []
            if reasoning:
                # Split reasoning into steps
                steps = [s.strip() for s in reasoning.split('\n') if s.strip()]

            result = {
                "answer": answer_text or str(calculated_result),
                "steps": steps if show_steps else [],
                "calculation": calculation_code,
                "calculated_value": calculated_result,
                "verification": verification,
                "agent_type": "math",
                "success": True
            }

            logger.info(f"Math problem solved: {result['answer'][:60]}...")

        except Exception as e:
            logger.error(f"Math solving failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "math",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction (Phase 4: Self-Improvement)
            try:
                # Quality scoring
                quality_scores = None
                if not error_occurred and result.get("success"):
                    quality_scorer = OutputQualityScorer()
                    answer_str = str(result.get("answer", ""))
                    quality_scores = await quality_scorer.score_answer(
                        question=problem,
                        answer=answer_str
                    )

                # Log to training data
                log_interaction(
                    query=problem,
                    answer=str(result.get("answer", "")),
                    agent_type="math",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["math_solver", "python_eval"] if result.get("calculation") else ["math_solver"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                # Don't fail the request if logging fails
                logger.warning(f"Failed to log math interaction: {log_error}")

        return result

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from formatted response"""
        pattern = f"{section_name}:\\s*(.+?)(?=\\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _safe_eval(self, code: str) -> Optional[float]:
        """
        Safely evaluate Python math code.

        Args:
            code: Python expression or statements

        Returns:
            Result or None if error
        """
        try:
            # Clean code
            code = code.strip()
            # Remove markdown code blocks
            code = re.sub(r'```python\s*|\s*```', '', code)

            # Try to eval as expression first
            try:
                result = eval(code, self.safe_env)
                return float(result) if result is not None else None
            except SyntaxError:
                # If that fails, try exec (for multi-line code)
                local_env = {}
                exec(code, self.safe_env, local_env)
                # Try to find result variable
                for var_name in ['result', 'answer', 'output']:
                    if var_name in local_env:
                        return float(local_env[var_name])
                return None

        except Exception as e:
            logger.warning(f"Code execution failed: {e}")
            return None

    def _verify_answer(
        self,
        problem: str,
        answer: str,
        calculated: Optional[float]
    ) -> Dict:
        """
        Verify the answer makes sense.

        Args:
            problem: Original problem
            answer: Text answer
            calculated: Calculated numerical result

        Returns:
            Verification dict
        """
        verification = {
            "passed": True,
            "checks": []
        }

        # Check 1: Answer is not empty
        if not answer or answer.lower() in ["none", "null", "error"]:
            verification["passed"] = False
            verification["checks"].append({
                "check": "non_empty",
                "passed": False,
                "message": "Answer is empty or invalid"
            })
        else:
            verification["checks"].append({
                "check": "non_empty",
                "passed": True
            })

        # Check 2: If calculated, check answer contains the number
        if calculated is not None:
            answer_lower = answer.lower()
            calc_str = str(calculated)
            calc_rounded = str(round(calculated, 2))

            contains_number = (
                calc_str in answer_lower or
                calc_rounded in answer_lower or
                str(int(calculated)) in answer_lower
            )

            verification["checks"].append({
                "check": "calculation_match",
                "passed": contains_number,
                "calculated_value": calculated,
                "message": "Answer contains calculated value" if contains_number else "Answer doesn't match calculation"
            })

            if not contains_number:
                verification["passed"] = False

        return verification

    async def batch_solve(self, problems: List[str]) -> List[Dict]:
        """
        Solve multiple problems.

        Args:
            problems: List of math problems

        Returns:
            List of results
        """
        results = []
        for problem in problems:
            result = await self.solve(problem)
            results.append(result)
        return results
