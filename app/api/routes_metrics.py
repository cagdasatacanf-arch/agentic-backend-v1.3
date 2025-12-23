"""
API Routes for Tool Metrics, Quality Evaluation, and Advanced Retrieval

Phase 1 Endpoints:
- GET /api/v1/metrics/tools - Get all tool metrics
- GET /api/v1/metrics/tools/{tool_name} - Get specific tool metrics
- POST /api/v1/quality/evaluate - Evaluate an answer's quality
- POST /api/v1/quality/batch - Batch evaluate multiple answers
- POST /api/v1/rag/multihop - Perform multi-hop retrieval

Phase 2 Endpoints:
- POST /api/v1/rag/hybrid - Hybrid search (BM25 + embeddings)
- POST /api/v1/rag/rerank - Re-rank documents with cross-encoder
- POST /api/v1/rag/retrieve-rerank - Complete pipeline (retrieve + rerank)
- POST /api/v1/rag/tune - Auto-tune RAG parameters
- POST /api/v1/rag/compare-methods - Compare retrieval methods

Part of Agentic AI Enhancements (Phase 1 & 2)
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging

from app.services.output_quality import OutputQualityScorer, evaluate_with_feedback
from app.services.multihop_rag import multihop_search, compare_single_vs_multihop
from app.services.tool_metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["metrics", "quality"])


# ============================================================================
# Request/Response Models
# ============================================================================

class QualityEvaluationRequest(BaseModel):
    """Request for quality evaluation"""
    question: str = Field(..., description="User's original question")
    answer: str = Field(..., description="Agent's answer to evaluate")
    sources: Optional[List[Dict]] = Field(None, description="Source documents used")
    ground_truth: Optional[str] = Field(None, description="Correct answer (if available)")
    include_feedback: bool = Field(True, description="Include human-readable feedback")


class BatchQualityRequest(BaseModel):
    """Request for batch quality evaluation"""
    qa_pairs: List[Dict[str, str]] = Field(
        ...,
        description="List of {question, answer, ground_truth} dicts"
    )
    sources_list: Optional[List[List[Dict]]] = Field(
        None,
        description="Sources for each Q&A pair"
    )


class MultiHopRequest(BaseModel):
    """Request for multi-hop retrieval"""
    query: str = Field(..., description="Search query")
    max_hops: int = Field(3, ge=1, le=5, description="Maximum retrieval iterations")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Quality threshold to stop")
    verbose: bool = Field(False, description="Include detailed logs")


class ComparisonRequest(BaseModel):
    """Request to compare single-shot vs multi-hop RAG"""
    query: str = Field(..., description="Search query")


# ============================================================================
# Tool Metrics Endpoints
# ============================================================================

@router.get("/metrics/tools")
async def get_all_tool_metrics(
    last_n: int = 100,
) -> Dict:
    """
    Get metrics summary for all tools.

    Args:
        last_n: Number of recent executions to analyze (default: 100)

    Returns:
        {
            "tools": [
                {
                    "tool_name": "calculator",
                    "quality_score": 0.95,
                    "success_rate": 0.98,
                    "execution_count": 45,
                    "latency_p50": 12.3,
                    "latency_p95": 45.2,
                    "latency_p99": 67.8
                },
                ...
            ]
        }
    """
    try:
        metrics_collector = get_metrics_collector()

        if not metrics_collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collection not enabled. Set enable_metrics=True in agent initialization."
            )

        summary = metrics_collector.get_all_tools_summary(last_n=last_n)

        return {
            "tools": summary,
            "analyzed_executions": last_n
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/tools/{tool_name}")
async def get_tool_metrics(
    tool_name: str,
    last_n: int = 100
) -> Dict:
    """
    Get detailed metrics for a specific tool.

    Args:
        tool_name: Name of the tool (e.g., "calculator", "search_documents")
        last_n: Number of recent executions to analyze

    Returns:
        {
            "tool_name": "calculator",
            "success_rate": 0.98,
            "latency_stats": {
                "p50": 12.3,
                "p95": 45.2,
                "p99": 67.8,
                "mean": 18.5,
                "min": 5.2,
                "max": 102.1
            },
            "quality_score": 0.95,
            "error_summary": {
                "division by zero": 2,
                "invalid expression": 1
            }
        }
    """
    try:
        metrics_collector = get_metrics_collector()

        if not metrics_collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collection not enabled"
            )

        # Get metrics
        success_rate = metrics_collector.get_success_rate(tool_name, last_n=last_n)
        latency_stats = metrics_collector.get_latency_stats(tool_name, last_n=last_n)
        quality_score = metrics_collector.get_tool_quality_score(tool_name, last_n=last_n)
        error_summary = metrics_collector.get_error_summary(tool_name, last_n=last_n)

        # Check if we have any data
        executions = metrics_collector.get_executions(tool_name, last_n=last_n)
        if not executions:
            raise HTTPException(
                status_code=404,
                detail=f"No execution data found for tool '{tool_name}'"
            )

        return {
            "tool_name": tool_name,
            "success_rate": round(success_rate, 3),
            "latency_stats": latency_stats,
            "quality_score": round(quality_score, 3),
            "error_summary": error_summary,
            "execution_count": len(executions),
            "analyzed_executions": last_n
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for {tool_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# ============================================================================
# Output Quality Endpoints
# ============================================================================

@router.post("/quality/evaluate")
async def evaluate_quality(request: QualityEvaluationRequest) -> Dict:
    """
    Evaluate the quality of an agent's answer.

    Uses LLM-as-judge to score:
    - Citation quality (0.0-1.0)
    - Completeness (0.0-1.0)
    - Conciseness (0.0-1.0)
    - Correctness (0.0-1.0, if ground_truth provided)
    - Overall (0.0-1.0, weighted average)

    Example:
        POST /api/v1/quality/evaluate
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language...",
            "sources": [...],
            "include_feedback": true
        }

    Returns:
        {
            "scores": {
                "citation_quality": 0.8,
                "completeness": 0.9,
                "conciseness": 0.85,
                "overall": 0.87
            },
            "feedback": ["✅ High quality answer!"],
            "grade": "B"
        }
    """
    try:
        if request.include_feedback:
            # Get scores + feedback
            result = await evaluate_with_feedback(
                question=request.question,
                answer=request.answer,
                sources=request.sources
            )
            return result
        else:
            # Just scores
            scorer = OutputQualityScorer()
            scores = await scorer.score_answer(
                question=request.question,
                answer=request.answer,
                sources=request.sources,
                ground_truth=request.ground_truth
            )
            return {"scores": scores}

    except Exception as e:
        logger.error(f"Error evaluating quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quality evaluation failed: {str(e)}")


@router.post("/quality/batch")
async def evaluate_batch(request: BatchQualityRequest) -> Dict:
    """
    Evaluate multiple Q&A pairs in batch.

    Example:
        POST /api/v1/quality/batch
        {
            "qa_pairs": [
                {
                    "question": "What is 2+2?",
                    "answer": "2+2 equals 4",
                    "ground_truth": "4"
                },
                ...
            ]
        }

    Returns:
        {
            "results": [
                {
                    "question": "What is 2+2?",
                    "answer": "2+2 equals 4",
                    "scores": {...}
                },
                ...
            ],
            "average_overall_score": 0.85
        }
    """
    try:
        scorer = OutputQualityScorer()

        results = await scorer.score_batch(
            qa_pairs=request.qa_pairs,
            sources_list=request.sources_list
        )

        # Calculate average overall score
        overall_scores = [r["scores"]["overall"] for r in results]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        return {
            "results": results,
            "count": len(results),
            "average_overall_score": round(avg_score, 3)
        }

    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


# ============================================================================
# Multi-Hop RAG Endpoints
# ============================================================================

@router.post("/rag/multihop")
async def multihop_retrieval(request: MultiHopRequest) -> Dict:
    """
    Perform multi-hop retrieval for complex queries.

    Instead of single-shot retrieval, iteratively:
    1. Retrieve documents
    2. Evaluate quality
    3. Refine query if needed
    4. Repeat until quality threshold met or max hops reached

    Example:
        POST /api/v1/rag/multihop
        {
            "query": "What were the causes of WWI and how did they lead to conflict?",
            "max_hops": 3,
            "quality_threshold": 0.7,
            "verbose": true
        }

    Returns:
        {
            "documents": [...],  # All retrieved documents (deduplicated)
            "retrieval_steps": ["WWI causes", "WWI escalation"],
            "quality_scores": [0.5, 0.8],
            "final_quality": 0.8,
            "hops_used": 2,
            "stopped_early": true
        }
    """
    try:
        result = await multihop_search(
            query=request.query,
            max_hops=request.max_hops,
            quality_threshold=request.quality_threshold,
            verbose=request.verbose
        )

        return result

    except Exception as e:
        logger.error(f"Error in multi-hop retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multi-hop retrieval failed: {str(e)}")


@router.post("/rag/compare")
async def compare_retrieval_methods(request: ComparisonRequest) -> Dict:
    """
    Compare single-shot RAG vs multi-hop RAG.

    Useful for evaluating if multi-hop improves results for your query.

    Example:
        POST /api/v1/rag/compare
        {
            "query": "Explain quantum entanglement and its applications"
        }

    Returns:
        {
            "single_shot": {
                "num_docs": 5,
                "documents": [...]
            },
            "multihop": {
                "num_docs": 8,
                "hops_used": 2,
                "final_quality": 0.85,
                "documents": [...]
            },
            "comparison": {
                "additional_docs_found": 3,
                "retrieval_steps": ["quantum entanglement", "quantum applications"]
            }
        }
    """
    try:
        result = await compare_single_vs_multihop(request.query)
        return result

    except Exception as e:
        logger.error(f"Error comparing retrieval methods: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================

@router.get("/metrics/health")
async def metrics_health() -> Dict:
    """
    Check if metrics collection is enabled and working.

    Returns:
        {
            "metrics_enabled": true,
            "tools_tracked": 3,
            "total_executions": 157
        }
    """
    try:
        metrics_collector = get_metrics_collector()

        if not metrics_collector:
            return {
                "metrics_enabled": False,
                "message": "Metrics collection not initialized"
            }

        summary = metrics_collector.get_all_tools_summary(last_n=1000)

        total_executions = sum(tool["execution_count"] for tool in summary)

        return {
            "metrics_enabled": True,
            "tools_tracked": len(summary),
            "total_executions": total_executions,
            "tools": [tool["tool_name"] for tool in summary]
        }

    except Exception as e:
        logger.error(f"Metrics health check failed: {e}")
        return {
            "metrics_enabled": False,
            "error": str(e)
        }

# ============================================================================
# PHASE 2: ADVANCED RETRIEVAL ENDPOINTS
# ============================================================================

class HybridSearchRequest(BaseModel):
    """Request for hybrid search"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for dense vs sparse (0.0=BM25 only, 1.0=embeddings only)")
    verbose: bool = Field(False, description="Include detailed logs")


class RerankRequest(BaseModel):
    """Request for re-ranking"""
    query: str = Field(..., description="Search query")
    documents: List[Dict] = Field(..., description="Documents to re-rank")
    top_k: Optional[int] = Field(None, description="Number of results (default: all)")
    verbose: bool = Field(False, description="Include detailed logs")


class RetrieveRerankRequest(BaseModel):
    """Request for combined retrieve + rerank pipeline"""
    query: str = Field(..., description="Search query")
    initial_top_k: int = Field(20, ge=5, le=100, description="Documents to retrieve initially")
    final_top_k: int = Field(5, ge=1, le=20, description="Documents to return after re-ranking")
    use_hybrid: bool = Field(True, description="Use hybrid search for initial retrieval")
    verbose: bool = Field(False, description="Include detailed logs")


class TuneRAGRequest(BaseModel):
    """Request for RAG parameter tuning"""
    test_queries: List[Dict[str, str]] = Field(
        ...,
        description="Test queries with ground truth: [{'question': ..., 'ground_truth': ...}, ...]"
    )
    use_hybrid: bool = Field(False, description="Tune for hybrid search")
    quick: bool = Field(True, description="Use quick mode (fewer combinations)")
    verbose: bool = Field(True, description="Include detailed logs")


class CompareMethodsRequest(BaseModel):
    """Request to compare retrieval methods"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Results per method")


@router.post("/rag/hybrid")
async def hybrid_search_endpoint(request: HybridSearchRequest) -> Dict:
    """
    Perform hybrid search combining BM25 and dense embeddings.

    Hybrid search uses Reciprocal Rank Fusion (RRF) to combine:
    - BM25 (keyword/sparse matching) - Good for exact matches, names, acronyms
    - Dense embeddings (semantic) - Good for meaning, context, paraphrases

    Typically 15-25% better than single-method retrieval.

    Example:
        POST /api/v1/rag/hybrid
        {
            "query": "What is Python used for?",
            "top_k": 10,
            "alpha": 0.7,
            "verbose": false
        }

    Returns:
        {
            "query": "...",
            "results": [...],
            "count": 10,
            "method": "hybrid",
            "config": {"alpha": 0.7, "top_k": 10}
        }
    """
    try:
        from app.services.hybrid_search import get_hybrid_retriever

        # Get or initialize hybrid retriever
        retriever = await get_hybrid_retriever()

        # Perform search
        results = await retriever.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            verbose=request.verbose
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "method": "hybrid",
            "config": {
                "alpha": request.alpha,
                "top_k": request.top_k
            }
        }

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/rag/rerank")
async def rerank_endpoint(request: RerankRequest) -> Dict:
    """
    Re-rank documents using cross-encoder.

    Uses LLM-as-ranker to score relevance of each document to the query.
    More accurate than cosine similarity but slower (use after initial retrieval).

    Example:
        POST /api/v1/rag/rerank
        {
            "query": "What is Python?",
            "documents": [{...}, {...}],
            "top_k": 5,
            "verbose": false
        }

    Returns:
        {
            "query": "...",
            "reranked": [...],  # Documents with "rerank_score"
            "count": 5
        }
    """
    try:
        from app.services.reranker import rerank_documents

        reranked = await rerank_documents(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k or len(request.documents),
            verbose=request.verbose
        )

        return {
            "query": request.query,
            "reranked": reranked,
            "count": len(reranked),
            "original_count": len(request.documents)
        }

    except Exception as e:
        logger.error(f"Re-ranking failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-ranking failed: {str(e)}")


@router.post("/rag/retrieve-rerank")
async def retrieve_rerank_endpoint(request: RetrieveRerankRequest) -> Dict:
    """
    Complete retrieval pipeline: Retrieve + Re-rank.

    This is the RECOMMENDED approach for production RAG:
    1. Retrieve initial_top_k documents (e.g., 20) using hybrid search
    2. Re-rank using cross-encoder
    3. Return final_top_k best documents (e.g., 5)

    Example:
        POST /api/v1/rag/retrieve-rerank
        {
            "query": "Explain quantum entanglement",
            "initial_top_k": 20,
            "final_top_k": 5,
            "use_hybrid": true,
            "verbose": false
        }

    Returns:
        {
            "query": "...",
            "results": [...],  # Top 5 re-ranked documents
            "count": 5,
            "pipeline": {
                "initial_retrieval": "hybrid",
                "initial_count": 20,
                "reranking": "cross-encoder",
                "final_count": 5
            }
        }
    """
    try:
        from app.services.reranker import retrieve_and_rerank

        results = await retrieve_and_rerank(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_k=request.final_top_k,
            use_hybrid=request.use_hybrid,
            verbose=request.verbose
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "pipeline": {
                "initial_retrieval": "hybrid" if request.use_hybrid else "dense",
                "initial_count": request.initial_top_k,
                "reranking": "cross-encoder",
                "final_count": request.final_top_k
            }
        }

    except Exception as e:
        logger.error(f"Retrieve+rerank pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@router.post("/rag/tune")
async def tune_rag_endpoint(request: TuneRAGRequest) -> Dict:
    """
    Auto-tune RAG parameters using test queries.

    Performs grid search over:
    - top_k: [3, 5, 7, 10]
    - score_threshold: [0.0, 0.5, 0.6, 0.7]
    - alpha (if hybrid): [0.5, 0.7, 0.9]

    Evaluates each combination using output quality scoring.

    Example:
        POST /api/v1/rag/tune
        {
            "test_queries": [
                {"question": "What is Python?", "ground_truth": "Python is a programming language..."},
                {"question": "What is RAG?", "ground_truth": "RAG is..."}
            ],
            "use_hybrid": false,
            "quick": true,
            "verbose": true
        }

    Returns:
        {
            "best_parameters": {
                "top_k": 5,
                "score_threshold": 0.6,
                "alpha": 0.7
            },
            "avg_quality_score": 0.85,
            "test_count": 10,
            "recommendation": "..."
        }
    """
    try:
        from app.services.rag_auto_tuner import RAGAutoTuner

        if len(request.test_queries) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 test queries required for tuning"
            )

        tuner = RAGAutoTuner(use_hybrid=request.use_hybrid)

        # Run tuning
        best_params = await tuner.tune(
            test_queries=request.test_queries,
            quick=request.quick,
            verbose=request.verbose
        )

        # Get full results for recommendation
        results = await tuner.grid_search(
            test_queries=request.test_queries,
            verbose=False
        )

        recommendation = tuner.generate_recommendation(results)

        return {
            "best_parameters": {
                "top_k": best_params.top_k,
                "score_threshold": best_params.score_threshold,
                "alpha": best_params.alpha
            },
            "avg_quality_score": results[0].avg_quality_score,
            "avg_precision": results[0].avg_retrieval_precision,
            "test_count": len(request.test_queries),
            "recommendation": recommendation,
            "all_results": [
                {
                    "params": {
                        "top_k": r.parameters.top_k,
                        "score_threshold": r.parameters.score_threshold,
                        "alpha": r.parameters.alpha
                    },
                    "avg_quality": r.avg_quality_score,
                    "avg_precision": r.avg_retrieval_precision
                }
                for r in results[:5]  # Top 5 configurations
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG tuning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tuning failed: {str(e)}")


@router.post("/rag/compare-methods")
async def compare_methods_endpoint(request: CompareMethodsRequest) -> Dict:
    """
    Compare different retrieval methods side by side.

    Compares:
    - Dense (embeddings only)
    - Sparse (BM25 only)
    - Hybrid (BM25 + embeddings)

    Useful for understanding which method works best for your query types.

    Example:
        POST /api/v1/rag/compare-methods
        {
            "query": "What are the benefits of Python?",
            "top_k": 5
        }

    Returns:
        {
            "query": "...",
            "dense": {"count": 5, "results": [...]},
            "sparse": {"count": 5, "results": [...]},
            "hybrid": {"count": 5, "results": [...]},
            "analysis": {
                "unique_to_dense": 2,
                "unique_to_sparse": 1,
                "overlap": 2
            }
        }
    """
    try:
        from app.services.hybrid_search import get_hybrid_retriever

        retriever = await get_hybrid_retriever()

        comparison = await retriever.compare_methods(
            query=request.query,
            top_k=request.top_k
        )

        return comparison

    except Exception as e:
        logger.error(f"Method comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
