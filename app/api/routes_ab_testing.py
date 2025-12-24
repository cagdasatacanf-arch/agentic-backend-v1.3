# app/api/routes_ab_testing.py
"""REST API routes for A/B testing framework.
Provides endpoints to create experiments, retrieve results, reset stats, and record outcomes.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.services.ab_testing import (
    create_experiment,
    assign_variant,
    record_outcome,
    get_experiment_results,
    reset_experiment,
)

router = APIRouter(prefix="/api/v1/ab-testing", tags=["ab-testing"])


class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Human readable description")
    variant_names: List[str] = Field(..., description="List of variant identifiers")
    traffic_split: Optional[List[float]] = Field(
        None, description="Optional traffic split list that must sum to 1.0"
    )
    bandit_algorithm: Optional[str] = Field(
        None, description="Bandit algorithm name (e.g., epsilon_greedy, ucb)"
    )


class RecordOutcomeRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    success: bool = Field(..., description="Whether the outcome was successful")
    value: Optional[float] = Field(0.0, description="Optional numeric value for the outcome")


@router.post("/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment_endpoint(req: ExperimentCreateRequest):
    """Create a new A/B testing experiment."""
    try:
        experiment = create_experiment(
            name=req.name,
            description=req.description or "",
            variant_names=req.variant_names,
            traffic_split=req.traffic_split,
            bandit_algorithm=req.bandit_algorithm,
        )
        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": [v.name for v in experiment.variants],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/experiments/{experiment_id}")
async def get_experiment_endpoint(experiment_id: str):
    """Retrieve experiment details and statistics."""
    try:
        results = get_experiment_results(experiment_id)
        return results
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/experiments/{experiment_id}/reset")
async def reset_experiment_endpoint(experiment_id: str):
    """Reset experiment statistics (clear metrics, set status to DRAFT)."""
    try:
        reset_experiment(experiment_id)
        return {"success": True, "message": "Experiment reset"}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/experiments/{experiment_id}/record")
async def record_outcome_endpoint(experiment_id: str, req: RecordOutcomeRequest):
    """Record an outcome for a user in a given experiment."""
    try:
        # Ensure the user has a variant assigned (creates if missing)
        assign_variant(experiment_id, req.user_id)
        record_outcome(
            experiment_id,
            req.user_id,
            success=req.success,
            value=req.value or 0.0,
        )
        return {"success": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
