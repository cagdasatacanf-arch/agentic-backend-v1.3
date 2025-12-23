"""
Cost Tracking & Budget Management API Routes

Endpoints for tracking API costs and managing budgets:
- Real-time cost monitoring
- User budget controls
- Cost analytics and forecasting
- Optimization recommendations

Usage:
    GET  /api/v1/costs/stats              # Cost statistics
    GET  /api/v1/costs/usage/{user_id}    # User usage details
    POST /api/v1/costs/budget             # Set budget limit
    GET  /api/v1/costs/forecast           # Cost forecast
    GET  /api/v1/costs/recommendations    # Optimization tips
    GET  /api/v1/costs/health             # Health check
"""

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import logging

from app.services.cost_tracker import get_cost_tracker, CostStats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/costs", tags=["costs"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SetBudgetRequest(BaseModel):
    """Request to set user budget limit"""
    user_id: str = Field(..., min_length=1, description="User ID")
    budget_limit: float = Field(..., gt=0, description="Budget limit in dollars")


class CostStatsResponse(BaseModel):
    """Cost statistics response"""
    total_cost: float = Field(..., description="Total cost in dollars")
    total_tokens: int = Field(..., description="Total tokens used")
    total_requests: int = Field(..., description="Total API requests")
    cost_by_model: Dict[str, float] = Field(..., description="Cost breakdown by model")
    cost_by_agent: Dict[str, float] = Field(..., description="Cost breakdown by agent")
    cost_by_user: Dict[str, float] = Field(..., description="Cost breakdown by user")
    period_start: datetime = Field(..., description="Period start time")
    period_end: datetime = Field(..., description="Period end time")
    average_cost_per_request: float = Field(..., description="Average cost per request")


class BudgetCheckResponse(BaseModel):
    """Budget check response"""
    allowed: bool = Field(..., description="Whether request is allowed")
    current_spend: float = Field(..., description="Current spending")
    budget_limit: float = Field(..., description="Budget limit")
    remaining: float = Field(..., description="Remaining budget")
    estimated_cost: float = Field(..., description="Estimated cost of operation")
    usage_percentage: float = Field(..., description="Percentage of budget used")


class RecommendationResponse(BaseModel):
    """Cost optimization recommendation"""
    type: str = Field(..., description="Recommendation type")
    severity: str = Field(..., description="Severity (low/medium/high)")
    recommendation: str = Field(..., description="Recommendation text")
    potential_savings: float = Field(..., description="Potential savings in dollars")
    details: Optional[Dict] = Field(None, description="Additional details")


class ForecastResponse(BaseModel):
    """Cost forecast response"""
    forecasted_cost: float = Field(..., description="Forecasted cost for period")
    forecast_period: str = Field(..., description="Forecast period")
    current_daily_average: float = Field(..., description="Current daily average spend")
    projected_monthly: float = Field(..., description="Projected monthly cost")
    trend: str = Field(..., description="Spending trend (increasing/stable/decreasing)")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/stats", response_model=CostStatsResponse)
async def get_cost_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    period: str = Query("today", regex="^(today|week|month|all)$", description="Time period"),
    start_time: Optional[datetime] = Query(None, description="Custom start time"),
    end_time: Optional[datetime] = Query(None, description="Custom end time")
):
    """
    Get cost statistics and analytics.

    Returns comprehensive cost breakdown by model, agent, and user.

    Example:
        GET /api/v1/costs/stats?period=week

        Response:
        {
            "total_cost": 12.45,
            "total_tokens": 1500000,
            "total_requests": 342,
            "cost_by_model": {
                "gpt-4o": 10.20,
                "gpt-4o-mini": 2.25
            },
            "cost_by_agent": {
                "math": 5.10,
                "code": 4.35,
                "rag": 3.00
            },
            "cost_by_user": {
                "user_123": 8.50,
                "user_456": 3.95
            },
            "period_start": "2024-01-15T00:00:00",
            "period_end": "2024-01-22T14:30:00",
            "average_cost_per_request": 0.0364
        }
    """
    try:
        logger.info(f"Getting cost stats: user={user_id}, period={period}")

        tracker = get_cost_tracker()
        stats = await tracker.get_stats(
            user_id=user_id,
            period=period,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate average
        avg_cost = stats.total_cost / stats.total_requests if stats.total_requests > 0 else 0.0

        return CostStatsResponse(
            total_cost=round(stats.total_cost, 4),
            total_tokens=stats.total_tokens,
            total_requests=stats.total_requests,
            cost_by_model=stats.cost_by_model,
            cost_by_agent=stats.cost_by_agent,
            cost_by_user=stats.cost_by_user,
            period_start=stats.period_start,
            period_end=stats.period_end,
            average_cost_per_request=round(avg_cost, 6)
        )

    except Exception as e:
        logger.error(f"Failed to get cost stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage/{user_id}", response_model=BudgetCheckResponse)
async def get_user_usage(
    user_id: str = Path(..., description="User ID")
):
    """
    Get user's current usage and budget status.

    Returns current spending, budget limit, and remaining budget.

    Example:
        GET /api/v1/costs/usage/user_123

        Response:
        {
            "allowed": true,
            "current_spend": 45.67,
            "budget_limit": 100.0,
            "remaining": 54.33,
            "estimated_cost": 0.0,
            "usage_percentage": 45.67
        }
    """
    try:
        logger.info(f"Getting usage for user: {user_id}")

        tracker = get_cost_tracker()
        budget_info = await tracker.check_budget(user_id, estimated_cost=0.0)

        usage_pct = 0.0
        if budget_info["budget_limit"] > 0:
            usage_pct = (budget_info["current_spend"] / budget_info["budget_limit"]) * 100

        return BudgetCheckResponse(
            allowed=budget_info["allowed"],
            current_spend=round(budget_info["current_spend"], 4),
            budget_limit=round(budget_info["budget_limit"], 2),
            remaining=round(budget_info["remaining"], 4),
            estimated_cost=round(budget_info["estimated_cost"], 6),
            usage_percentage=round(usage_pct, 2)
        )

    except Exception as e:
        logger.error(f"Failed to get user usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/budget")
async def set_budget(request: SetBudgetRequest):
    """
    Set budget limit for a user.

    Allows administrators to set spending limits per user.

    Example:
        POST /api/v1/costs/budget
        {
            "user_id": "user_123",
            "budget_limit": 50.0
        }

        Response:
        {
            "status": "success",
            "user_id": "user_123",
            "budget_limit": 50.0,
            "message": "Budget limit set successfully"
        }
    """
    try:
        logger.info(f"Setting budget for {request.user_id}: ${request.budget_limit}")

        tracker = get_cost_tracker()
        await tracker.set_budget(request.user_id, request.budget_limit)

        return {
            "status": "success",
            "user_id": request.user_id,
            "budget_limit": request.budget_limit,
            "message": "Budget limit set successfully"
        }

    except Exception as e:
        logger.error(f"Failed to set budget: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=ForecastResponse)
async def get_cost_forecast(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    period: str = Query("month", regex="^(week|month)$", description="Forecast period")
):
    """
    Get cost forecast based on current usage trends.

    Predicts future spending based on historical data.

    Example:
        GET /api/v1/costs/forecast?period=month

        Response:
        {
            "forecasted_cost": 145.50,
            "forecast_period": "month",
            "current_daily_average": 4.85,
            "projected_monthly": 145.50,
            "trend": "increasing"
        }
    """
    try:
        logger.info(f"Generating cost forecast: user={user_id}, period={period}")

        tracker = get_cost_tracker()

        # Get last week's stats for forecasting
        week_stats = await tracker.get_stats(user_id=user_id, period="week")

        # Calculate daily average
        days_in_period = 7
        daily_avg = week_stats.total_cost / days_in_period if week_stats.total_cost > 0 else 0.0

        # Forecast based on period
        if period == "week":
            forecasted_cost = daily_avg * 7
            forecast_days = 7
        else:  # month
            forecasted_cost = daily_avg * 30
            forecast_days = 30

        # Determine trend (compare with previous period)
        # For simplicity, we'll compare week vs today
        today_stats = await tracker.get_stats(user_id=user_id, period="today")
        today_avg = today_stats.total_cost

        if today_avg > daily_avg * 1.2:
            trend = "increasing"
        elif today_avg < daily_avg * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"

        return ForecastResponse(
            forecasted_cost=round(forecasted_cost, 2),
            forecast_period=period,
            current_daily_average=round(daily_avg, 4),
            projected_monthly=round(daily_avg * 30, 2),
            trend=trend
        )

    except Exception as e:
        logger.error(f"Failed to generate forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Get cost optimization recommendations.

    Returns actionable suggestions to reduce API costs.

    Example:
        GET /api/v1/costs/recommendations?user_id=user_123

        Response:
        [
            {
                "type": "model_optimization",
                "severity": "high",
                "recommendation": "Consider using gpt-4o-mini for simple queries. 75.2% of costs are from gpt-4o.",
                "potential_savings": 9.18,
                "details": {
                    "current_model": "gpt-4o",
                    "percentage": 75.2
                }
            },
            {
                "type": "caching",
                "severity": "medium",
                "recommendation": "Enable semantic caching to reduce repeated queries.",
                "potential_savings": 3.74,
                "details": null
            }
        ]
    """
    try:
        logger.info(f"Getting optimization recommendations for user: {user_id}")

        tracker = get_cost_tracker()
        raw_recommendations = await tracker.get_optimization_recommendations(user_id)

        # Format recommendations
        recommendations = []
        for rec in raw_recommendations:
            recommendations.append(RecommendationResponse(
                type=rec["type"],
                severity=rec["severity"],
                recommendation=rec["recommendation"],
                potential_savings=round(rec.get("potential_savings", 0.0), 2),
                details={
                    k: v for k, v in rec.items()
                    if k not in ["type", "severity", "recommendation", "potential_savings"]
                } if len(rec) > 4 else None
            ))

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for cost tracking system.

    Returns:
        Status of cost tracking capabilities
    """
    try:
        tracker = get_cost_tracker()

        # Try a simple operation to verify Redis connection
        test_cost = tracker.calculate_cost("gpt-4o-mini", 100, 50)

        return {
            "status": "healthy",
            "cost_tracking": "enabled",
            "budget_controls": "enabled",
            "supported_models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ],
            "features": [
                "real-time tracking",
                "budget limits",
                "cost analytics",
                "forecasting",
                "optimization recommendations"
            ],
            "test_calculation": f"100 input + 50 output tokens (gpt-4o-mini) = ${test_cost:.6f}"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/models")
async def get_pricing_info():
    """
    Get pricing information for all supported models.

    Returns current pricing per 1M tokens for all models.

    Example:
        GET /api/v1/costs/models

        Response:
        {
            "pricing_date": "2024-12",
            "currency": "USD",
            "unit": "per_1M_tokens",
            "models": {
                "gpt-4o": {
                    "input": 2.50,
                    "output": 10.00
                },
                "gpt-4o-mini": {
                    "input": 0.150,
                    "output": 0.600
                },
                ...
            }
        }
    """
    try:
        tracker = get_cost_tracker()

        # Convert enum keys to strings for JSON response
        pricing_info = {
            model_enum.value: pricing
            for model_enum, pricing in tracker.PRICING.items()
        }

        return {
            "pricing_date": "2024-12",
            "currency": "USD",
            "unit": "per_1M_tokens",
            "models": pricing_info,
            "note": "Pricing based on OpenAI rates as of December 2024"
        }

    except Exception as e:
        logger.error(f"Failed to get pricing info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track")
async def track_usage_manually(
    model: str = Query(..., description="Model name"),
    input_tokens: int = Query(..., ge=0, description="Input tokens"),
    output_tokens: int = Query(..., ge=0, description="Output tokens"),
    user_id: Optional[str] = Query(None, description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    agent_type: Optional[str] = Query(None, description="Agent type")
):
    """
    Manually track token usage (for testing or external integrations).

    Normally, agents automatically track their usage. This endpoint allows
    manual tracking for testing or integrating with external systems.

    Example:
        POST /api/v1/costs/track?model=gpt-4o&input_tokens=1000&output_tokens=500&user_id=user_123

        Response:
        {
            "cost": 0.0075,
            "model": "gpt-4o",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "user_id": "user_123"
        }
    """
    try:
        logger.info(f"Manual usage tracking: {model}, {input_tokens}+{output_tokens} tokens")

        tracker = get_cost_tracker()
        cost = await tracker.track_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id=user_id,
            session_id=session_id,
            agent_type=agent_type
        )

        return {
            "cost": round(cost, 6),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "user_id": user_id,
            "session_id": session_id,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to track usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
