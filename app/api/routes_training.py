"""
Training Data Management API

Endpoints for Phase 4: Self-Improvement & RL Training Pipeline

Features:
- Interaction statistics and retrieval
- SFT/DPO/GRPO dataset building
- Training script generation
- Data cleanup utilities

Usage:
    GET  /api/v1/training/stats - Get interaction statistics
    GET  /api/v1/training/interactions - Retrieve logged interactions
    POST /api/v1/training/dataset/sft - Build SFT dataset
    POST /api/v1/training/dataset/dpo - Build DPO dataset
    POST /api/v1/training/dataset/grpo - Build GRPO dataset
    POST /api/v1/training/script/generate - Generate training script
    DELETE /api/v1/training/interactions/cleanup - Clean old interactions
    GET  /api/v1/training/dataset/stats - Get dataset potential stats
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

from app.services.interaction_logger import get_interaction_logger
from app.services.dataset_builder import DatasetBuilder
from app.services.rl_training_guide import (
    TrainingConfig,
    generate_sft_training_script,
    generate_dpo_training_script,
    save_training_script,
    get_training_recommendations
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["training"])


# ============================================================================
# Request/Response Models
# ============================================================================

class InteractionStatsResponse(BaseModel):
    """Response model for interaction statistics"""
    total_interactions: int
    high_quality_count: int
    error_count: int
    high_quality_rate: float
    error_rate: float
    agent_stats: Dict[str, int] = Field(default_factory=dict)


class InteractionQuery(BaseModel):
    """Query parameters for retrieving interactions"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    agent_type: Optional[str] = None
    high_quality_only: bool = False
    limit: int = Field(default=100, ge=1, le=1000)


class SFTDatasetRequest(BaseModel):
    """Request model for building SFT dataset"""
    min_quality: float = Field(default=0.8, ge=0.0, le=1.0)
    max_samples: Optional[int] = Field(default=None, ge=1)
    agent_types: Optional[List[str]] = None
    days_back: int = Field(default=30, ge=1, le=365)
    save_to_file: Optional[str] = None


class DPODatasetRequest(BaseModel):
    """Request model for building DPO dataset"""
    min_quality_diff: float = Field(default=0.2, ge=0.0, le=1.0)
    max_pairs: Optional[int] = Field(default=None, ge=1)
    days_back: int = Field(default=30, ge=1, le=365)
    save_to_file: Optional[str] = None


class GRPODatasetRequest(BaseModel):
    """Request model for building GRPO dataset"""
    group_size: int = Field(default=4, ge=2, le=10)
    max_groups: Optional[int] = Field(default=None, ge=1)
    days_back: int = Field(default=30, ge=1, le=365)
    save_to_file: Optional[str] = None


class TrainingScriptRequest(BaseModel):
    """Request model for generating training scripts"""
    script_type: str = Field(..., pattern="^(sft|dpo)$")
    dataset_path: str
    output_dir: str
    sft_model_path: Optional[str] = None  # Required for DPO
    config: Optional[Dict] = None
    save_to_file: Optional[str] = None


class DatasetStatsResponse(BaseModel):
    """Response model for dataset statistics"""
    total_interactions: int
    with_quality_scores: int
    high_quality: int
    high_quality_rate: float
    by_agent_type: Dict[str, int]
    potential_datasets: Dict[str, int]
    days_covered: int
    recommendation: str


class CleanupRequest(BaseModel):
    """Request model for cleaning old interactions"""
    days_to_keep: int = Field(default=30, ge=1, le=365)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/stats", response_model=InteractionStatsResponse)
async def get_interaction_stats():
    """
    Get interaction statistics.

    Returns:
        Statistics about logged interactions including:
        - Total interactions
        - High-quality count and rate
        - Error count and rate
        - Per-agent statistics

    Example:
        GET /api/v1/training/stats

        Response:
        {
            "total_interactions": 1500,
            "high_quality_count": 1200,
            "error_count": 50,
            "high_quality_rate": 0.8,
            "error_rate": 0.033,
            "agent_stats": {
                "agent_math_count": 300,
                "agent_code_count": 450,
                "agent_rag_count": 600,
                "agent_general_count": 150
            }
        }
    """
    try:
        logger.info("Fetching interaction statistics")

        interaction_logger = get_interaction_logger()
        stats = interaction_logger.get_stats()

        # Separate agent stats
        agent_stats = {
            k: v for k, v in stats.items()
            if k.startswith("agent_")
        }

        response = InteractionStatsResponse(
            total_interactions=stats.get("total_interactions", 0),
            high_quality_count=stats.get("high_quality_count", 0),
            error_count=stats.get("error_count", 0),
            high_quality_rate=stats.get("high_quality_rate", 0.0),
            error_rate=stats.get("error_rate", 0.0),
            agent_stats=agent_stats
        )

        logger.info(f"Retrieved stats: {response.total_interactions} total interactions")
        return response

    except Exception as e:
        logger.error(f"Failed to get interaction stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactions/query")
async def query_interactions(query: InteractionQuery):
    """
    Query and retrieve logged interactions.

    Args:
        query: Query parameters (time range, agent type, quality filter, limit)

    Returns:
        List of interactions matching criteria

    Example:
        POST /api/v1/training/interactions/query
        {
            "high_quality_only": true,
            "agent_type": "math",
            "days_back": 7,
            "limit": 50
        }

        Response:
        {
            "interactions": [...],
            "count": 45,
            "query_params": {...}
        }
    """
    try:
        logger.info(f"Querying interactions: {query.dict()}")

        interaction_logger = get_interaction_logger()

        # Default start_time if not provided
        start_time = query.start_time
        if not start_time and not query.end_time:
            start_time = datetime.now() - timedelta(days=30)

        interactions = interaction_logger.get_interactions(
            start_time=start_time,
            end_time=query.end_time,
            agent_type=query.agent_type,
            high_quality_only=query.high_quality_only,
            limit=query.limit
        )

        # Convert to dict format
        interaction_dicts = [
            {
                "interaction_id": i.interaction_id,
                "timestamp": i.timestamp.isoformat(),
                "query": i.query,
                "answer": i.answer,
                "agent_type": i.agent_type,
                "quality_scores": i.quality_scores,
                "latency_ms": i.latency_ms,
                "tools_used": i.tools_used,
                "sources": i.sources,
                "error_occurred": i.error_occurred
            }
            for i in interactions
        ]

        response = {
            "interactions": interaction_dicts,
            "count": len(interaction_dicts),
            "query_params": query.dict()
        }

        logger.info(f"Retrieved {len(interaction_dicts)} interactions")
        return response

    except Exception as e:
        logger.error(f"Failed to query interactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/sft")
async def build_sft_dataset(request: SFTDatasetRequest):
    """
    Build Supervised Fine-Tuning (SFT) dataset.

    Creates dataset from high-quality interactions in HuggingFace format.

    Args:
        request: SFT dataset parameters

    Returns:
        SFT dataset and metadata

    Example:
        POST /api/v1/training/dataset/sft
        {
            "min_quality": 0.8,
            "max_samples": 1000,
            "agent_types": ["math", "code"],
            "days_back": 30,
            "save_to_file": "data/sft_dataset.jsonl"
        }

        Response:
        {
            "dataset": [...],
            "count": 850,
            "saved_to": "data/sft_dataset.jsonl",
            "metadata": {...}
        }
    """
    try:
        logger.info(f"Building SFT dataset: {request.dict()}")

        builder = DatasetBuilder()

        dataset = await builder.build_sft_dataset(
            min_quality=request.min_quality,
            max_samples=request.max_samples,
            agent_types=request.agent_types,
            days_back=request.days_back
        )

        # Save to file if requested
        saved_to = None
        if request.save_to_file:
            success = builder.save_dataset(dataset, request.save_to_file, format="jsonl")
            if success:
                saved_to = request.save_to_file

        response = {
            "dataset": dataset,
            "count": len(dataset),
            "saved_to": saved_to,
            "metadata": {
                "min_quality": request.min_quality,
                "max_samples": request.max_samples,
                "agent_types": request.agent_types,
                "days_back": request.days_back,
                "format": "huggingface_sft"
            }
        }

        logger.info(f"Built SFT dataset: {len(dataset)} samples")
        return response

    except Exception as e:
        logger.error(f"Failed to build SFT dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/dpo")
async def build_dpo_dataset(request: DPODatasetRequest):
    """
    Build Direct Preference Optimization (DPO) dataset.

    Creates dataset from preference pairs (chosen vs rejected).

    Args:
        request: DPO dataset parameters

    Returns:
        DPO dataset and metadata

    Example:
        POST /api/v1/training/dataset/dpo
        {
            "min_quality_diff": 0.2,
            "max_pairs": 500,
            "days_back": 30,
            "save_to_file": "data/dpo_dataset.jsonl"
        }

        Response:
        {
            "dataset": [...],
            "count": 420,
            "saved_to": "data/dpo_dataset.jsonl",
            "metadata": {...}
        }
    """
    try:
        logger.info(f"Building DPO dataset: {request.dict()}")

        builder = DatasetBuilder()

        dataset = await builder.build_dpo_dataset(
            min_quality_diff=request.min_quality_diff,
            max_pairs=request.max_pairs,
            days_back=request.days_back
        )

        # Save to file if requested
        saved_to = None
        if request.save_to_file:
            success = builder.save_dataset(dataset, request.save_to_file, format="jsonl")
            if success:
                saved_to = request.save_to_file

        response = {
            "dataset": dataset,
            "count": len(dataset),
            "saved_to": saved_to,
            "metadata": {
                "min_quality_diff": request.min_quality_diff,
                "max_pairs": request.max_pairs,
                "days_back": request.days_back,
                "format": "dpo"
            }
        }

        logger.info(f"Built DPO dataset: {len(dataset)} pairs")
        return response

    except Exception as e:
        logger.error(f"Failed to build DPO dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/grpo")
async def build_grpo_dataset(request: GRPODatasetRequest):
    """
    Build GRPO (Group Relative Policy Optimization) dataset.

    Creates dataset with grouped responses and rewards.

    Args:
        request: GRPO dataset parameters

    Returns:
        GRPO dataset and metadata

    Example:
        POST /api/v1/training/dataset/grpo
        {
            "group_size": 4,
            "max_groups": 200,
            "days_back": 30,
            "save_to_file": "data/grpo_dataset.jsonl"
        }

        Response:
        {
            "dataset": [...],
            "count": 180,
            "saved_to": "data/grpo_dataset.jsonl",
            "metadata": {...}
        }
    """
    try:
        logger.info(f"Building GRPO dataset: {request.dict()}")

        builder = DatasetBuilder()

        dataset = await builder.build_grpo_dataset(
            group_size=request.group_size,
            max_groups=request.max_groups,
            days_back=request.days_back
        )

        # Save to file if requested
        saved_to = None
        if request.save_to_file:
            success = builder.save_dataset(dataset, request.save_to_file, format="jsonl")
            if success:
                saved_to = request.save_to_file

        response = {
            "dataset": dataset,
            "count": len(dataset),
            "saved_to": saved_to,
            "metadata": {
                "group_size": request.group_size,
                "max_groups": request.max_groups,
                "days_back": request.days_back,
                "format": "grpo"
            }
        }

        logger.info(f"Built GRPO dataset: {len(dataset)} groups")
        return response

    except Exception as e:
        logger.error(f"Failed to build GRPO dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/script/generate")
async def generate_training_script(request: TrainingScriptRequest):
    """
    Generate training script for SFT or DPO.

    Creates ready-to-run Python training script.

    Args:
        request: Script generation parameters

    Returns:
        Training script content

    Example:
        POST /api/v1/training/script/generate
        {
            "script_type": "sft",
            "dataset_path": "data/sft_dataset.jsonl",
            "output_dir": "models/fine-tuned",
            "save_to_file": "scripts/train_sft.py"
        }

        Response:
        {
            "script": "...",
            "script_type": "sft",
            "saved_to": "scripts/train_sft.py",
            "config": {...}
        }
    """
    try:
        logger.info(f"Generating {request.script_type.upper()} training script")

        # Build config
        if request.config:
            config = TrainingConfig()
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            config = TrainingConfig()

        # Generate script
        if request.script_type == "sft":
            script = generate_sft_training_script(
                dataset_path=request.dataset_path,
                output_dir=request.output_dir,
                config=config
            )
        elif request.script_type == "dpo":
            if not request.sft_model_path:
                raise HTTPException(
                    status_code=400,
                    detail="sft_model_path required for DPO training"
                )
            script = generate_dpo_training_script(
                dataset_path=request.dataset_path,
                sft_model_path=request.sft_model_path,
                output_dir=request.output_dir,
                config=config
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid script_type: {request.script_type}"
            )

        # Save to file if requested
        saved_to = None
        if request.save_to_file:
            success = save_training_script(script, request.save_to_file)
            if success:
                saved_to = request.save_to_file

        response = {
            "script": script,
            "script_type": request.script_type,
            "saved_to": saved_to,
            "config": config.to_dict()
        }

        logger.info(f"Generated {request.script_type.upper()} script: {len(script)} chars")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate training script: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(days_back: int = Query(default=30, ge=1, le=365)):
    """
    Get statistics about potential training datasets.

    Analyzes logged interactions and estimates dataset sizes.

    Args:
        days_back: Number of days to analyze

    Returns:
        Dataset statistics and recommendations

    Example:
        GET /api/v1/training/dataset/stats?days_back=30

        Response:
        {
            "total_interactions": 1500,
            "with_quality_scores": 1450,
            "high_quality": 1200,
            "high_quality_rate": 0.8,
            "by_agent_type": {...},
            "potential_datasets": {
                "sft_samples": 1200,
                "dpo_pairs": 350,
                "grpo_groups": 80
            },
            "days_covered": 30,
            "recommendation": "..."
        }
    """
    try:
        logger.info(f"Getting dataset stats for last {days_back} days")

        builder = DatasetBuilder()
        stats = await builder.get_dataset_stats(days_back=days_back)

        # Get recommendation based on dataset size
        recommendations = get_training_recommendations()
        total = stats["total_interactions"]

        if total < 100:
            recommendation = recommendations["small_dataset"]
        elif total < 1000:
            recommendation = recommendations["medium_dataset"]
        elif total < 10000:
            recommendation = recommendations["large_dataset"]
        else:
            recommendation = recommendations["very_large_dataset"]

        response = DatasetStatsResponse(
            total_interactions=stats["total_interactions"],
            with_quality_scores=stats["with_quality_scores"],
            high_quality=stats["high_quality"],
            high_quality_rate=stats["high_quality_rate"],
            by_agent_type=stats["by_agent_type"],
            potential_datasets=stats["potential_datasets"],
            days_covered=stats["days_covered"],
            recommendation=recommendation
        )

        logger.info(f"Dataset stats: {total} interactions, {stats['high_quality']} high-quality")
        return response

    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/interactions/cleanup")
async def cleanup_old_interactions(request: CleanupRequest):
    """
    Clean up old interactions to save storage.

    Removes interactions older than specified days.

    Args:
        request: Cleanup parameters

    Returns:
        Number of interactions removed

    Example:
        DELETE /api/v1/training/interactions/cleanup
        {
            "days_to_keep": 30
        }

        Response:
        {
            "removed_count": 450,
            "days_kept": 30,
            "message": "Successfully removed 450 interactions"
        }
    """
    try:
        logger.info(f"Cleaning up interactions older than {request.days_to_keep} days")

        interaction_logger = get_interaction_logger()
        removed_count = interaction_logger.clear_old_interactions(
            days_to_keep=request.days_to_keep
        )

        response = {
            "removed_count": removed_count,
            "days_kept": request.days_to_keep,
            "message": f"Successfully removed {removed_count} interactions older than {request.days_to_keep} days"
        }

        logger.info(f"Cleaned up {removed_count} old interactions")
        return response

    except Exception as e:
        logger.error(f"Failed to cleanup interactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for training system.

    Returns:
        Status of training data management system
    """
    try:
        interaction_logger = get_interaction_logger()
        stats = interaction_logger.get_stats()

        return {
            "status": "healthy",
            "total_interactions": stats.get("total_interactions", 0),
            "system": "training_data_management",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
