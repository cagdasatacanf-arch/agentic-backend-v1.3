"""
API Routes for Tool Metrics and Output Quality Evaluation

Endpoints:
- GET /api/v1/metrics/tools - Get all tool metrics
- GET /api/v1/metrics/tools/{tool_name} - Get specific tool metrics
- POST /api/v1/quality/evaluate - Evaluate an answer's quality
- POST /api/v1/quality/batch - Batch evaluate multiple answers
- POST /api/v1/rag/multihop - Perform multi-hop retrieval

Part of Phase 1: Agentic AI Enhancements
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
