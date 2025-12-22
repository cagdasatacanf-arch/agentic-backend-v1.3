"""
RAG Specialist Agent

Specialized agent for knowledge retrieval and research:
- Uses advanced RAG techniques from Phase 2
- Hybrid search (BM25 + embeddings)
- Cross-encoder re-ranking
- Multi-hop retrieval for complex questions
- Citation and source tracking

Based on research:
- Specialized RAG agents provide better document retrieval
- Multi-hop retrieval improves complex QA
- Re-ranking significantly improves relevance
"""

from typing import Dict, List, Optional
import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.services.interaction_logger import log_interaction
from app.services.output_quality import OutputQualityScorer

logger = logging.getLogger(__name__)


class RAGSpecialist:
    """
    Specialized agent for knowledge-base queries.

    Leverages all Phase 2 capabilities:
    - Hybrid search (BM25 + dense embeddings)
    - Cross-encoder re-ranking
    - Multi-hop retrieval for complex questions

    Usage:
        agent = RAGSpecialist()
        result = await agent.query("What is retrieval-augmented generation?")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_multihop: bool = False  # Use for complex queries
    ):
        """
        Initialize RAG specialist.

        Args:
            model: LLM to use for generation
            use_hybrid: Use hybrid search (recommended)
            use_reranking: Use cross-encoder re-ranking (recommended)
            use_multihop: Use multi-hop retrieval for all queries
        """
        self.model = model
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_multihop = use_multihop

        self.llm = ChatOpenAI(
            model=model,
            temperature=0,  # Factual responses
            api_key=settings.openai_api_key
        )

        logger.info(
            f"RAGSpecialist initialized: hybrid={use_hybrid}, "
            f"rerank={use_reranking}, multihop={use_multihop}"
        )

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        Answer a knowledge-base query.

        Args:
            question: User's question
            session_id: Optional session ID for context
            top_k: Number of sources to use
            verbose: Log detailed retrieval

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "retrieval_method": "...",
                "agent_type": "rag"
            }
        """
        logger.info(f"RAG query: {question[:60]}...")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # 1. Retrieve documents using best method
            documents = await self._retrieve_documents(
                question,
                top_k=top_k,
                verbose=verbose
            )

            if not documents:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base.",
                    "sources": [],
                    "retrieval_method": self._get_retrieval_method(),
                    "agent_type": "rag",
                    "success": True
                }

            # 2. Generate answer from documents
            answer = await self._generate_answer(question, documents)

            # 3. Format sources
            sources = [
                {
                    "id": doc.get("id", ""),
                    "text": doc.get("text", "")[:300],  # First 300 chars
                    "score": doc.get("rerank_score") or doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                for doc in documents
            ]

            result = {
                "answer": answer,
                "sources": sources,
                "retrieval_method": self._get_retrieval_method(),
                "documents_used": len(documents),
                "agent_type": "rag",
                "success": True
            }

            logger.info(f"RAG query completed: {len(answer)} chars, {len(sources)} sources")

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "rag",
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
                    quality_scores = await quality_scorer.score_answer(
                        question=question,
                        answer=result.get("answer", ""),
                        sources=result.get("sources", [])
                    )

                # Log to training data
                tools_used = [self._get_retrieval_method()]

                log_interaction(
                    query=question,
                    answer=result.get("answer", ""),
                    agent_type="rag",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=tools_used,
                    sources=result.get("sources", []),
                    retrieval_method=result.get("retrieval_method"),
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                # Don't fail the request if logging fails
                logger.warning(f"Failed to log RAG interaction: {log_error}")

        return result

    async def _retrieve_documents(
        self,
        query: str,
        top_k: int,
        verbose: bool
    ) -> List[Dict]:
        """
        Retrieve documents using configured method.

        Priority:
        1. Multi-hop (if enabled)
        2. Hybrid + Re-ranking (if both enabled)
        3. Hybrid (if enabled)
        4. Dense (fallback)
        """

        # Multi-hop retrieval (for complex questions)
        if self.use_multihop:
            from app.services.multihop_rag import multihop_search
            result = await multihop_search(
                query=query,
                max_hops=3,
                quality_threshold=0.7,
                verbose=verbose
            )
            return result.get("documents", [])[:top_k]

        # Hybrid + Re-ranking (RECOMMENDED)
        elif self.use_hybrid and self.use_reranking:
            from app.services.reranker import retrieve_and_rerank
            return await retrieve_and_rerank(
                query=query,
                initial_top_k=top_k * 4,  # Retrieve 4x, re-rank to top_k
                final_top_k=top_k,
                use_hybrid=True,
                verbose=verbose
            )

        # Hybrid only
        elif self.use_hybrid:
            from app.services.hybrid_search import hybrid_search
            return await hybrid_search(
                query=query,
                top_k=top_k,
                alpha=0.7,  # 70% dense, 30% BM25
            )

        # Dense fallback
        else:
            from app.rag import search_docs
            return await search_docs(query, top_k=top_k)

    async def _generate_answer(
        self,
        question: str,
        documents: List[Dict]
    ) -> str:
        """
        Generate answer from retrieved documents.

        Args:
            question: User's question
            documents: Retrieved documents

        Returns:
            Generated answer
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", f"Document {i+1}")

            context_parts.append(f"[{filename}]\n{text}")

        context = "\n\n".join(context_parts)

        # Generate answer
        system_prompt = """You are a knowledgeable assistant. Answer questions based on the provided documents.

Rules:
- Answer directly and concisely
- Cite sources using document names (e.g., "According to [filename]...")
- If information is unclear or missing, say so
- Don't make up information not in the documents"""

        user_prompt = f"""Documents:
{context}

Question: {question}

Answer:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    def _get_retrieval_method(self) -> str:
        """Get description of retrieval method used"""
        if self.use_multihop:
            return "multi-hop"
        elif self.use_hybrid and self.use_reranking:
            return "hybrid+rerank"
        elif self.use_hybrid:
            return "hybrid"
        else:
            return "dense"

    async def summarize_documents(
        self,
        document_ids: Optional[List[str]] = None,
        max_docs: int = 10
    ) -> Dict:
        """
        Summarize multiple documents.

        Args:
            document_ids: Specific document IDs to summarize
            max_docs: Maximum documents to summarize

        Returns:
            Summary dict
        """
        # This is a placeholder - would need document store access
        return {
            "summary": "Document summarization not yet implemented",
            "agent_type": "rag",
            "success": False,
            "error": "Not implemented"
        }

    async def compare_sources(
        self,
        question: str,
        source_a: str,
        source_b: str
    ) -> Dict:
        """
        Compare information from two sources.

        Args:
            question: Question to compare answers for
            source_a: First source
            source_b: Second source

        Returns:
            Comparison dict
        """
        system_prompt = """You are a source comparison expert. Compare how two sources address a question.

Identify:
- Points of agreement
- Points of disagreement
- Unique information in each source
- Overall which source is more comprehensive"""

        user_prompt = f"""Question: {question}

Source A:
{source_a}

Source B:
{source_b}

Comparison:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "comparison": response.content,
                "question": question,
                "agent_type": "rag",
                "success": True
            }

        except Exception as e:
            logger.error(f"Source comparison failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "rag",
                "success": False
            }
