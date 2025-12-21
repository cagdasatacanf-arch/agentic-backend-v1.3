"""
Multi-Hop Retrieval-Augmented Generation

Iterative retrieval system that:
1. Retrieves documents for query
2. Evaluates if retrieved docs are sufficient
3. If not, refines query and retrieves again
4. Repeats up to max_hops times

Based on research from:
- StepSearch (EMNLP'25): Multi-hop QA optimization
- DeepRetrieval (COLM'25): Iterative retrieval with quality checks

Benefits:
- Better answers for complex questions requiring multiple pieces of info
- Self-correcting retrieval that adapts to partial results
- Quality-driven stopping criteria
"""

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from app.rag import search_docs
from app.config import settings

logger = logging.getLogger(__name__)


class MultiHopRAGRetriever:
    """
    Iterative retrieval with quality-based stopping.

    Instead of single-shot RAG:
        Question → Retrieve → Answer

    Multi-hop RAG:
        Question → Retrieve → Evaluate Quality
                → If insufficient: Refine Query → Retrieve Again
                → Repeat up to max_hops
                → Answer with all gathered context

    Example:
        retriever = MultiHopRAGRetriever(max_hops=3)
        result = await retriever.retrieve_with_refinement(
            "What were the main causes of World War I and how did they lead to conflict?"
        )
        # May retrieve docs about causes (hop 1), then about escalation (hop 2)
    """

    def __init__(
        self,
        max_hops: int = 3,
        quality_threshold: float = 0.7,
        top_k_per_hop: int = 5,
        llm_model: str = "gpt-4o-mini",  # For quality checks & refinement
        api_key: Optional[str] = None
    ):
        """
        Initialize multi-hop retriever.

        Args:
            max_hops: Maximum retrieval iterations
            quality_threshold: Stop if quality >= this (0.0-1.0)
            top_k_per_hop: Documents to retrieve per hop
            llm_model: Model for quality evaluation
            api_key: OpenAI API key
        """
        self.max_hops = max_hops
        self.quality_threshold = quality_threshold
        self.top_k_per_hop = top_k_per_hop

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            api_key=api_key or settings.openai_api_key
        )

        logger.info(
            f"MultiHopRAGRetriever initialized: "
            f"max_hops={max_hops}, threshold={quality_threshold}"
        )

    async def retrieve_with_refinement(
        self,
        original_query: str,
        verbose: bool = False
    ) -> Dict:
        """
        Perform iterative retrieval with quality checks.

        Process:
        1. Retrieve docs for query
        2. Ask LLM: "Is this enough to answer?"
        3. If no: "What should we search for next?"
        4. Repeat until quality threshold met or max hops reached

        Args:
            original_query: User's question
            verbose: Log detailed progress

        Returns:
            {
                "documents": List[Dict],  # All retrieved docs (deduplicated)
                "retrieval_steps": List[str],  # Queries used at each hop
                "quality_scores": List[float],  # Quality at each hop
                "final_quality": float,  # Final quality score
                "hops_used": int,  # Number of hops performed
                "stopped_early": bool  # True if stopped due to quality threshold
            }
        """

        all_documents = []
        retrieval_steps = []
        quality_scores = []
        current_query = original_query

        for hop in range(self.max_hops):
            if verbose:
                logger.info(f"Hop {hop+1}/{self.max_hops}: Searching for '{current_query}'")

            # 1. Retrieve documents
            try:
                docs = await search_docs(current_query, top_k=self.top_k_per_hop)
            except Exception as e:
                logger.error(f"Retrieval failed at hop {hop+1}: {e}")
                docs = []

            all_documents.extend(docs)
            retrieval_steps.append(current_query)

            # 2. Evaluate quality: "Is this enough?"
            try:
                quality_score = await self._evaluate_retrieval_quality(
                    original_query=original_query,
                    retrieved_docs=all_documents,
                    verbose=verbose
                )
                quality_scores.append(quality_score)

                if verbose:
                    logger.info(
                        f"Hop {hop+1} quality: {quality_score:.2f} "
                        f"(threshold: {self.quality_threshold:.2f})"
                    )

            except Exception as e:
                logger.error(f"Quality evaluation failed: {e}")
                quality_score = 0.5  # Neutral default
                quality_scores.append(quality_score)

            # 3. Check if quality is sufficient
            if quality_score >= self.quality_threshold:
                if verbose:
                    logger.info(
                        f"Quality threshold met ({quality_score:.2f} >= {self.quality_threshold:.2f}). "
                        f"Stopping after {hop+1} hops."
                    )

                return {
                    "documents": self._deduplicate_docs(all_documents),
                    "retrieval_steps": retrieval_steps,
                    "quality_scores": quality_scores,
                    "final_quality": quality_score,
                    "hops_used": hop + 1,
                    "stopped_early": True
                }

            # 4. If not last hop, refine query
            if hop < self.max_hops - 1:
                try:
                    refined_query = await self._refine_query(
                        original_query=original_query,
                        previous_queries=retrieval_steps,
                        retrieved_docs=all_documents,
                        verbose=verbose
                    )

                    if verbose:
                        logger.info(f"Refined query: '{refined_query}'")

                    current_query = refined_query

                except Exception as e:
                    logger.error(f"Query refinement failed: {e}")
                    # Fallback: use original query
                    current_query = original_query

        # Max hops reached
        final_quality = quality_scores[-1] if quality_scores else 0.0

        if verbose:
            logger.info(
                f"Max hops ({self.max_hops}) reached. "
                f"Final quality: {final_quality:.2f}"
            )

        return {
            "documents": self._deduplicate_docs(all_documents),
            "retrieval_steps": retrieval_steps,
            "quality_scores": quality_scores,
            "final_quality": final_quality,
            "hops_used": self.max_hops,
            "stopped_early": False
        }

    async def _evaluate_retrieval_quality(
        self,
        original_query: str,
        retrieved_docs: List[Dict],
        verbose: bool = False
    ) -> float:
        """
        Use LLM to evaluate if retrieved docs are sufficient.

        Args:
            original_query: User's original question
            retrieved_docs: Documents retrieved so far

        Returns:
            Quality score (0.0-1.0)
        """

        docs_summary = self._format_docs_for_prompt(retrieved_docs, max_docs=10)

        prompt = f"""You are evaluating the quality of retrieved documents for answering a question.

Original Question: {original_query}

Retrieved Documents:
{docs_summary}

Evaluate: How confident are you that these documents contain enough information to fully answer the question?

Consider:
- Coverage: Do the docs address all aspects of the question?
- Relevance: Are the docs on-topic?
- Completeness: Is key information present?

Respond with ONLY a number between 0.0 (insufficient) and 1.0 (fully sufficient).
No explanation, just the number."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()

            # Extract number
            import re
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                if verbose:
                    logger.warning(f"Could not parse quality score: {score_text}")
                return 0.5

        except Exception as e:
            logger.error(f"Quality evaluation error: {e}")
            return 0.5

    async def _refine_query(
        self,
        original_query: str,
        previous_queries: List[str],
        retrieved_docs: List[Dict],
        verbose: bool = False
    ) -> str:
        """
        Ask LLM to generate a refined search query.

        Based on what we've found so far, what should we search for next?

        Args:
            original_query: User's original question
            previous_queries: Queries used in previous hops
            retrieved_docs: Documents retrieved so far

        Returns:
            Refined search query
        """

        docs_summary = self._format_docs_for_prompt(retrieved_docs, max_docs=5)
        prev_queries_str = "\n".join([f"- {q}" for q in previous_queries])

        prompt = f"""You are helping refine a search query to find missing information.

Original Question: {original_query}

Previous Searches:
{prev_queries_str}

Documents Retrieved So Far:
{docs_summary}

The retrieved documents are not sufficient to fully answer the question.
What should we search for NEXT to fill the gaps?

Generate a refined search query that:
1. Targets missing information
2. Uses different keywords than previous searches
3. Is specific and focused

Respond with ONLY the new search query, nothing else."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            refined_query = response.content.strip()

            # Fallback if LLM returns empty or too long
            if not refined_query or len(refined_query) > 200:
                if verbose:
                    logger.warning(f"Invalid refined query, using original")
                return original_query

            return refined_query

        except Exception as e:
            logger.error(f"Query refinement error: {e}")
            return original_query

    def _deduplicate_docs(self, docs: List[Dict]) -> List[Dict]:
        """
        Remove duplicate documents by ID.

        Args:
            docs: List of document dicts

        Returns:
            Deduplicated list
        """
        seen_ids = set()
        unique_docs = []

        for doc in docs:
            doc_id = doc.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
            elif not doc_id:
                # If no ID, always include (can't deduplicate)
                unique_docs.append(doc)

        return unique_docs

    def _format_docs_for_prompt(
        self,
        docs: List[Dict],
        max_docs: int = 10,
        max_chars_per_doc: int = 200
    ) -> str:
        """
        Format documents for LLM prompt.

        Args:
            docs: List of document dicts
            max_docs: Max documents to include
            max_chars_per_doc: Max characters per document

        Returns:
            Formatted string
        """
        if not docs:
            return "[No documents retrieved yet]"

        formatted = []

        for i, doc in enumerate(docs[:max_docs]):
            score = doc.get("score", 0.0)
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Truncate text
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc] + "..."

            # Include filename if available
            filename = metadata.get("filename", "Unknown")

            formatted.append(
                f"[{i+1}] (score: {score:.2f}, source: {filename})\n{text}"
            )

        return "\n\n".join(formatted)


# ============================================================================
# Convenience Functions
# ============================================================================

async def multihop_search(
    query: str,
    max_hops: int = 3,
    quality_threshold: float = 0.7,
    verbose: bool = False
) -> Dict:
    """
    Quick function for multi-hop retrieval.

    Args:
        query: Search query
        max_hops: Maximum retrieval iterations
        quality_threshold: Quality threshold to stop early
        verbose: Log detailed progress

    Returns:
        Retrieval result dict
    """
    retriever = MultiHopRAGRetriever(
        max_hops=max_hops,
        quality_threshold=quality_threshold
    )

    return await retriever.retrieve_with_refinement(query, verbose=verbose)


async def compare_single_vs_multihop(query: str) -> Dict:
    """
    Compare single-shot RAG vs multi-hop RAG.

    Useful for evaluating if multi-hop improves results.

    Args:
        query: Search query

    Returns:
        {
            "single_shot": {...},
            "multihop": {...},
            "comparison": {...}
        }
    """

    # Single-shot retrieval
    single_docs = await search_docs(query, top_k=5)

    # Multi-hop retrieval
    multihop_result = await multihop_search(query, max_hops=3, verbose=True)

    return {
        "single_shot": {
            "num_docs": len(single_docs),
            "documents": single_docs
        },
        "multihop": {
            "num_docs": len(multihop_result["documents"]),
            "hops_used": multihop_result["hops_used"],
            "final_quality": multihop_result["final_quality"],
            "documents": multihop_result["documents"]
        },
        "comparison": {
            "additional_docs_found": (
                len(multihop_result["documents"]) - len(single_docs)
            ),
            "retrieval_steps": multihop_result["retrieval_steps"]
        }
    }
