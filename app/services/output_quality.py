"""
Output Quality Scoring for Agent Responses

Evaluates agent answer quality using multiple metrics:
- Correctness (if ground truth available)
- Citation quality (does it reference sources?)
- Completeness (fully addresses the question?)
- Conciseness (appropriate length?)

Based on research from Agent Lightning and DeepRetrieval papers.
Uses "LLM-as-judge" pattern for automated evaluation.
"""

from typing import Optional, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)


class OutputQualityScorer:
    """
    Evaluates agent output quality using LLM-as-judge.

    Returns scores (0.0-1.0) for:
    - citation_quality: Does answer cite sources?
    - completeness: Does it fully answer the question?
    - conciseness: Is it appropriately brief?
    - correctness: Is it accurate? (requires ground truth)
    - overall: Weighted combination

    Usage:
        scorer = OutputQualityScorer()
        scores = await scorer.score_answer(
            question="What is Python?",
            answer="Python is a programming language...",
            sources=[{...}]
        )
        print(f"Overall quality: {scores['overall']}")
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",  # Faster/cheaper for evaluation
        api_key: Optional[str] = None
    ):
        """
        Initialize quality scorer.

        Args:
            judge_model: OpenAI model to use for evaluation
            api_key: OpenAI API key (defaults to settings.openai_api_key)
        """
        self.judge_model = judge_model
        self.api_key = api_key or settings.openai_api_key

        self.llm = ChatOpenAI(
            model=self.judge_model,
            temperature=0,  # Deterministic scoring
            api_key=self.api_key
        )

        logger.info(f"OutputQualityScorer initialized with {judge_model}")

    async def score_answer(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        ground_truth: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Score an agent's answer across multiple dimensions.

        Args:
            question: User's original question
            answer: Agent's response
            sources: List of source documents used (for citation quality)
            ground_truth: Optional correct answer (for correctness scoring)
            weights: Optional custom weights for overall score

        Returns:
            Dict with scores:
            {
                "citation_quality": 0.0-1.0,
                "completeness": 0.0-1.0,
                "conciseness": 0.0-1.0,
                "correctness": 0.0-1.0,  # Only if ground_truth provided
                "overall": 0.0-1.0
            }
        """

        # Default weights
        if weights is None:
            weights = {
                "citation_quality": 0.2,
                "completeness": 0.4,
                "conciseness": 0.2,
                "correctness": 0.2  # Only used if ground_truth provided
            }

        scores = {}

        # 1. Citation Quality
        if sources:
            scores["citation_quality"] = self._score_citations(answer, sources)
        else:
            scores["citation_quality"] = 1.0  # N/A if no sources

        # 2. Completeness (LLM-as-judge)
        try:
            scores["completeness"] = await self._score_completeness(question, answer)
        except Exception as e:
            logger.error(f"Failed to score completeness: {e}")
            scores["completeness"] = 0.5  # Neutral default

        # 3. Conciseness (heuristic)
        scores["conciseness"] = self._score_conciseness(question, answer)

        # 4. Correctness (LLM-as-judge, only if ground truth)
        if ground_truth:
            try:
                scores["correctness"] = await self._score_correctness(
                    question, answer, ground_truth
                )
            except Exception as e:
                logger.error(f"Failed to score correctness: {e}")
                scores["correctness"] = 0.5

        # 5. Overall score (weighted)
        total_weight = 0.0
        weighted_sum = 0.0

        for metric, weight in weights.items():
            if metric in scores:
                weighted_sum += scores[metric] * weight
                total_weight += weight

        scores["overall"] = weighted_sum / total_weight if total_weight > 0 else 0.0

        logger.debug(f"Scored answer: overall={scores['overall']:.3f}, "
                    f"completeness={scores['completeness']:.3f}")

        return scores

    def _score_citations(self, answer: str, sources: List[Dict]) -> float:
        """
        Check if answer properly cites sources.

        Heuristic approach:
        - Count how many sources are referenced in the answer
        - Look for source IDs, filenames, or significant text chunks
        - Return % of sources cited

        Args:
            answer: Agent's response
            sources: List of source dicts (with 'id', 'text', 'metadata')

        Returns:
            Citation score (0.0-1.0)
        """
        if not sources:
            return 1.0  # N/A

        answer_lower = answer.lower()
        citations_found = 0

        for source in sources:
            # Check if source ID is mentioned
            source_id = source.get("id", "")
            if source_id and source_id.lower() in answer_lower:
                citations_found += 1
                continue

            # Check if filename is mentioned
            metadata = source.get("metadata", {})
            filename = metadata.get("filename", "")
            if filename and filename.lower() in answer_lower:
                citations_found += 1
                continue

            # Check if significant chunks of source text appear
            source_text = source.get("text", "")
            if source_text:
                # Extract first 10 words as signature
                words = source_text.split()[:10]
                if len(words) >= 5:
                    signature = " ".join(words[:5]).lower()
                    if signature in answer_lower:
                        citations_found += 1
                        continue

        # Return proportion of sources cited
        citation_score = citations_found / len(sources) if sources else 1.0

        return min(1.0, citation_score)

    async def _score_completeness(self, question: str, answer: str) -> float:
        """
        Use LLM to judge if answer completely addresses the question.

        Args:
            question: User's question
            answer: Agent's response

        Returns:
            Completeness score (0.0-1.0)
        """

        prompt = f"""You are an expert evaluator. Rate how completely the following answer addresses the question.

Question: {question}

Answer: {answer}

Evaluation criteria:
- Does it address all parts of the question?
- Is it specific enough?
- Does it acknowledge when information is missing?

Respond with ONLY a number between 0.0 (completely inadequate) and 1.0 (perfectly complete).
No explanation, just the number."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()

            # Extract number using regex
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                # Clamp to [0.0, 1.0]
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse completeness score: {score_text}")
                return 0.5

        except Exception as e:
            logger.error(f"Error scoring completeness: {e}")
            return 0.5

    def _score_conciseness(self, question: str, answer: str) -> float:
        """
        Heuristic scoring of answer conciseness.

        Formula:
            ideal_length = len(question) * 3
            score = ideal_length / max(len(answer), ideal_length)

        This penalizes answers that are much longer than needed.

        Args:
            question: User's question
            answer: Agent's response

        Returns:
            Conciseness score (0.0-1.0)
        """

        # Ideal: answer should be ~3x question length (heuristic)
        ideal_length = len(question) * 3
        actual_length = len(answer)

        if actual_length <= ideal_length:
            return 1.0  # Perfect or better
        else:
            # Penalize overly long answers
            score = ideal_length / actual_length
            return max(0.0, min(1.0, score))

    async def _score_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        Use LLM to compare answer against ground truth.

        Args:
            question: User's question
            answer: Agent's response
            ground_truth: Correct answer

        Returns:
            Correctness score (0.0-1.0)
        """

        prompt = f"""You are an expert evaluator. Compare the following answer to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Provided Answer: {answer}

Rate how correct the provided answer is compared to the ground truth.
Consider:
- Factual accuracy
- Semantic equivalence (same meaning, different words is OK)
- Completeness of key points

Respond with ONLY a number between 0.0 (completely wrong) and 1.0 (fully correct).
No explanation, just the number."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()

            # Extract number
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse correctness score: {score_text}")
                return 0.5

        except Exception as e:
            logger.error(f"Error scoring correctness: {e}")
            return 0.5

    async def score_batch(
        self,
        qa_pairs: List[Dict[str, str]],
        sources_list: Optional[List[List[Dict]]] = None
    ) -> List[Dict[str, float]]:
        """
        Score multiple Q&A pairs in batch.

        Args:
            qa_pairs: List of {"question": ..., "answer": ..., "ground_truth": ...}
            sources_list: Optional list of sources for each Q&A pair

        Returns:
            List of score dicts
        """

        results = []

        for i, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            ground_truth = qa.get("ground_truth")

            sources = None
            if sources_list and i < len(sources_list):
                sources = sources_list[i]

            scores = await self.score_answer(
                question=question,
                answer=answer,
                sources=sources,
                ground_truth=ground_truth
            )

            results.append({
                "question": question,
                "answer": answer,
                "scores": scores
            })

        return results


# ============================================================================
# Helper Functions
# ============================================================================

async def quick_score(question: str, answer: str, sources: Optional[List[Dict]] = None) -> float:
    """
    Quick scoring function that returns just the overall score.

    Args:
        question: User's question
        answer: Agent's response
        sources: Optional source documents

    Returns:
        Overall quality score (0.0-1.0)
    """
    scorer = OutputQualityScorer()
    scores = await scorer.score_answer(question, answer, sources)
    return scores["overall"]


async def evaluate_with_feedback(
    question: str,
    answer: str,
    sources: Optional[List[Dict]] = None
) -> Dict:
    """
    Score answer and provide detailed feedback.

    Args:
        question: User's question
        answer: Agent's response
        sources: Optional source documents

    Returns:
        Dict with scores and human-readable feedback
    """
    scorer = OutputQualityScorer()
    scores = await scorer.score_answer(question, answer, sources)

    # Generate feedback based on scores
    feedback = []

    if scores["citation_quality"] < 0.5:
        feedback.append("⚠️  Consider citing more sources in your answer")

    if scores["completeness"] < 0.6:
        feedback.append("⚠️  Answer may not fully address all aspects of the question")

    if scores["conciseness"] < 0.6:
        feedback.append("⚠️  Answer could be more concise")

    if scores["overall"] >= 0.8:
        feedback.append("✅ High quality answer!")
    elif scores["overall"] >= 0.6:
        feedback.append("✓ Good answer with room for improvement")
    else:
        feedback.append("❌ Answer needs significant improvement")

    return {
        "scores": scores,
        "feedback": feedback,
        "grade": (
            "A" if scores["overall"] >= 0.9 else
            "B" if scores["overall"] >= 0.8 else
            "C" if scores["overall"] >= 0.7 else
            "D" if scores["overall"] >= 0.6 else
            "F"
        )
    }
