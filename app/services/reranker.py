"""
Cross-Encoder Re-Ranking for RAG

Re-ranks retrieved documents using a cross-encoder model.

Why Re-ranking?
- Initial retrieval (dense/BM25) uses bi-encoders: encode query and docs separately
- Cross-encoders encode query+doc together → more accurate but slower
- Two-stage approach: retrieve many with bi-encoder, re-rank top-k with cross-encoder

Research basis:
- Cross-encoders significantly improve ranking quality
- 2-stage retrieval is state-of-the-art
- Typical improvement: 10-20% in relevance metrics

Models:
- OpenAI based: Use GPT-4 to score relevance (flexible but costs $)
- Sentence-transformers: Use local cross-encoder (free, faster)

This implementation uses OpenAI for simplicity (can swap to sentence-transformers).
"""

from typing import List, Dict, Optional
import logging
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-rank documents using cross-encoder scoring.

    Process:
    1. Take top-N documents from initial retrieval
    2. For each doc: score how relevant it is to the query
    3. Re-sort by relevance score
    4. Return top-k re-ranked results

    Uses LLM-as-ranker approach (similar to Phase 1's LLM-as-judge).

    Usage:
        reranker = CrossEncoderReranker()
        documents = [...] # From initial retrieval
        reranked = await reranker.rerank(query="...", documents=documents, top_k=5)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Fast and cheap for re-ranking
        api_key: Optional[str] = None,
        max_concurrent: int = 5  # Parallel scoring
    ):
        """
        Initialize re-ranker.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (defaults to settings)
            max_concurrent: Max parallel LLM calls
        """
        self.model = model
        self.max_concurrent = max_concurrent

        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Deterministic ranking
            api_key=api_key or settings.openai_api_key
        )

        logger.info(f"CrossEncoderReranker initialized with {model}")

    async def score_relevance(
        self,
        query: str,
        document_text: str,
        verbose: bool = False
    ) -> float:
        """
        Score how relevant a document is to the query.

        Args:
            query: Search query
            document_text: Document content
            verbose: Log scoring details

        Returns:
            Relevance score (0.0-1.0)
        """
        # Truncate document if too long
        max_chars = 1500
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "..."

        prompt = f"""Rate how relevant this document is to the query.

Query: {query}

Document: {document_text}

How relevant is this document to answering the query?

Respond with ONLY a number between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
No explanation, just the number."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()

            # Extract number
            import re
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

                if verbose:
                    logger.info(
                        f"Scored doc (score={score:.3f}): {document_text[:60]}..."
                    )

                return score
            else:
                logger.warning(f"Could not parse relevance score: {score_text}")
                return 0.5  # Neutral default

        except Exception as e:
            logger.error(f"Error scoring relevance: {e}")
            return 0.5

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: Search query
            documents: Documents from initial retrieval
            top_k: Number of results to return (default: all)
            verbose: Log re-ranking progress

        Returns:
            Re-ranked documents (sorted by relevance)
        """
        if not documents:
            return []

        if verbose:
            logger.info(f"Re-ranking {len(documents)} documents for query: '{query}'")

        # Score all documents (with concurrency limit)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def score_document(doc: Dict) -> Tuple[Dict, float]:
            async with semaphore:
                text = doc.get("text", "")
                score = await self.score_relevance(query, text, verbose=verbose)
                return doc, score

        # Create tasks for all documents
        tasks = [score_document(doc) for doc in documents]

        # Execute with concurrency limit
        results = await asyncio.gather(*tasks)

        # Build re-ranked list
        reranked = []
        for doc, relevance_score in results:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = relevance_score
            doc_copy["original_score"] = doc.get("score", 0.0)  # Keep original
            doc_copy["retrieval_method"] = doc.get("retrieval_method", "") + "+rerank"
            reranked.append(doc_copy)

        # Sort by re-rank score
        reranked.sort(key=lambda d: d["rerank_score"], reverse=True)

        # Return top-k
        if top_k:
            reranked = reranked[:top_k]

        if verbose:
            logger.info(f"Re-ranking complete. Top 3:")
            for i, doc in enumerate(reranked[:3]):
                logger.info(
                    f"  {i+1}. Rerank: {doc['rerank_score']:.3f}, "
                    f"Original: {doc['original_score']:.3f}, "
                    f"Text: {doc.get('text', '')[:60]}..."
                )

        return reranked

    async def compare_with_without(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        Compare rankings with and without re-ranking.

        Args:
            query: Search query
            documents: Documents from initial retrieval
            top_k: Number to compare

        Returns:
            Comparison dict
        """
        # Without re-ranking (just top-k from initial retrieval)
        without_rerank = documents[:top_k]

        # With re-ranking
        with_rerank = await self.rerank(query, documents, top_k=top_k)

        # Calculate rank changes
        rank_changes = []
        for i, reranked_doc in enumerate(with_rerank):
            doc_id = reranked_doc.get("id", "")
            # Find original rank
            original_rank = next(
                (idx for idx, d in enumerate(documents) if d.get("id") == doc_id),
                -1
            )
            if original_rank >= 0:
                rank_change = original_rank - i  # Positive = moved up
                rank_changes.append(rank_change)

        return {
            "query": query,
            "without_rerank": {
                "top_k": without_rerank,
                "count": len(without_rerank)
            },
            "with_rerank": {
                "top_k": with_rerank,
                "count": len(with_rerank)
            },
            "analysis": {
                "avg_rank_change": sum(rank_changes) / len(rank_changes) if rank_changes else 0,
                "max_rank_improvement": max(rank_changes) if rank_changes else 0,
                "documents_reordered": sum(1 for rc in rank_changes if rc != 0)
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def rerank_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    verbose: bool = False
) -> List[Dict]:
    """
    Quick re-ranking function.

    Args:
        query: Search query
        documents: Documents to re-rank
        top_k: Number of results
        verbose: Log progress

    Returns:
        Re-ranked documents
    """
    reranker = CrossEncoderReranker()
    return await reranker.rerank(query, documents, top_k=top_k, verbose=verbose)


# ============================================================================
# Combined Retrieval + Re-Ranking Pipeline
# ============================================================================

async def retrieve_and_rerank(
    query: str,
    initial_top_k: int = 20,
    final_top_k: int = 5,
    use_hybrid: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Complete retrieval pipeline with re-ranking.

    Process:
    1. Retrieve initial_top_k documents (hybrid or dense)
    2. Re-rank using cross-encoder
    3. Return final_top_k best documents

    This is the RECOMMENDED approach for production RAG.

    Args:
        query: Search query
        initial_top_k: Documents to retrieve initially (e.g., 20)
        final_top_k: Documents to return after re-ranking (e.g., 5)
        use_hybrid: Use hybrid search for initial retrieval
        verbose: Log detailed progress

    Returns:
        Re-ranked top documents
    """
    if verbose:
        logger.info(
            f"Retrieve+Rerank: query='{query}', "
            f"initial={initial_top_k}, final={final_top_k}, hybrid={use_hybrid}"
        )

    # 1. Initial retrieval
    if use_hybrid:
        from app.services.hybrid_search import hybrid_search
        documents = await hybrid_search(query, top_k=initial_top_k)
    else:
        from app.rag import search_docs
        documents = await search_docs(query, top_k=initial_top_k)

    if verbose:
        logger.info(f"Initial retrieval: {len(documents)} documents")

    # 2. Re-rank
    reranker = CrossEncoderReranker()
    reranked = await reranker.rerank(
        query=query,
        documents=documents,
        top_k=final_top_k,
        verbose=verbose
    )

    if verbose:
        logger.info(f"Final re-ranked results: {len(reranked)} documents")

    return reranked
