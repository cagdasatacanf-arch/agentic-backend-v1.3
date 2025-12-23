"""
Rate Limiting API Routes

Endpoints for managing and monitoring rate limits:
- Configure user tiers and limits
- Monitor usage statistics
- View top users by request volume
- Reset rate limits

Usage:
    POST /api/v1/ratelimit/tier            # Set user tier
    POST /api/v1/ratelimit/custom-limit    # Set custom limit
    GET  /api/v1/ratelimit/status/{user_id} # Get rate limit status
    GET  /api/v1/ratelimit/stats/{user_id}  # Get usage statistics
    GET  /api/v1/ratelimit/top-users        # Get top users
    POST /api/v1/ratelimit/reset/{user_id}  # Reset user limits
    GET  /api/v1/ratelimit/health           # Health check
"""

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import logging

from app.services.rate_limiter import (
    get_rate_limiter,
    UserTier,
    RateLimitTier,
    RateLimitStatus,
    DEFAULT_RATE_LIMITS
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ratelimit", tags=["ratelimit"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SetTierRequest(BaseModel):
    """Request to set user tier"""
    user_id: str = Field(..., min_length=1, description="User ID")
    tier: UserTier = Field(..., description="User tier")


class SetCustomLimitRequest(BaseModel):
    """Request to set custom limit"""
    user_id: str = Field(..., min_length=1, description="User ID")
    tier: RateLimitTier = Field(..., description="Rate limit tier")
    limit: int = Field(..., gt=0, description="Request limit")


class RateLimitStatusResponse(BaseModel):
    """Rate limit status response"""
    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after: Optional[int] = None


class UsageStatsResponse(BaseModel):
    """Usage statistics response"""
    user_id: str
    tier: str
    current: int
    limit: int
    remaining: int
    usage_percent: float
    reset_at: str
    window_seconds: int


class TopUserResponse(BaseModel):
    """Top user response"""
    user_id: str
    requests: int
    tier: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/tier")
async def set_user_tier(request: SetTierRequest):
    """
    Set user's rate limit tier.

    Tiers:
    - free: 2/sec, 60/min, 1000/hour, 10k/day
    - basic: 5/sec, 150/min, 5k/hour, 50k/day
    - pro: 10/sec, 300/min, 15k/hour, 150k/day
    - enterprise: 50/sec, 1k/min, 50k/hour, 500k/day

    Example:
        POST /api/v1/ratelimit/tier
        {
            "user_id": "user_123",
            "tier": "pro"
        }

        Response:
        {
            "status": "success",
            "user_id": "user_123",
            "tier": "pro",
            "limits": {
                "second": 10,
                "minute": 300,
                "hour": 15000,
                "day": 150000
            }
        }
    """
    try:
        logger.info(f"Setting tier for {request.user_id}: {request.tier}")

        limiter = get_rate_limiter()
        await limiter.set_user_tier(request.user_id, request.tier)

        # Get limits for this tier
        limits = {
            tier.value: limit
            for tier, limit in DEFAULT_RATE_LIMITS[request.tier].items()
        }

        return {
            "status": "success",
            "user_id": request.user_id,
            "tier": request.tier.value,
            "limits": limits
        }

    except Exception as e:
        logger.error(f"Failed to set user tier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-limit")
async def set_custom_limit(request: SetCustomLimitRequest):
    """
    Set custom rate limit for a user.

    Overrides default tier limits for specific time window.

    Example:
        POST /api/v1/ratelimit/custom-limit
        {
            "user_id": "user_123",
            "tier": "minute",
            "limit": 500
        }

        Response:
        {
            "status": "success",
            "user_id": "user_123",
            "tier": "minute",
            "limit": 500
        }
    """
    try:
        logger.info(
            f"Setting custom limit for {request.user_id}: "
            f"{request.tier.value}={request.limit}"
        )

        limiter = get_rate_limiter()
        await limiter.set_custom_limit(
            request.user_id,
            request.tier,
            request.limit
        )

        return {
            "status": "success",
            "user_id": request.user_id,
            "tier": request.tier.value,
            "limit": request.limit
        }

    except Exception as e:
        logger.error(f"Failed to set custom limit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{user_id}", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(
    user_id: str = Path(..., description="User ID"),
    endpoint: Optional[str] = Query(None, description="Optional endpoint filter")
):
    """
    Get current rate limit status for user.

    Checks against all rate limit tiers and returns current status.

    Example:
        GET /api/v1/ratelimit/status/user_123

        Response:
        {
            "allowed": true,
            "remaining": 8,
            "limit": 10,
            "reset_at": "2024-01-20T14:30:01",
            "retry_after": null
        }
    """
    try:
        logger.info(f"Checking rate limit status for {user_id}")

        limiter = get_rate_limiter()
        status = await limiter.check_limit(user_id, endpoint)

        return RateLimitStatusResponse(
            allowed=status.allowed,
            remaining=status.remaining,
            limit=status.limit,
            reset_at=status.reset_at,
            retry_after=status.retry_after
        )

    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{user_id}", response_model=UsageStatsResponse)
async def get_usage_stats(
    user_id: str = Path(..., description="User ID"),
    tier: RateLimitTier = Query(RateLimitTier.HOUR, description="Time window")
):
    """
    Get usage statistics for user.

    Returns request counts and limits for specified time window.

    Example:
        GET /api/v1/ratelimit/stats/user_123?tier=hour

        Response:
        {
            "user_id": "user_123",
            "tier": "hour",
            "current": 1234,
            "limit": 15000,
            "remaining": 13766,
            "usage_percent": 8.23,
            "reset_at": "2024-01-20T15:00:00",
            "window_seconds": 3600
        }
    """
    try:
        logger.info(f"Getting usage stats for {user_id} (tier={tier.value})")

        limiter = get_rate_limiter()
        stats = await limiter.get_usage_stats(user_id, tier)

        if not stats:
            raise HTTPException(status_code=404, detail="User not found")

        return UsageStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-users", response_model=List[TopUserResponse])
async def get_top_users(
    tier: RateLimitTier = Query(RateLimitTier.HOUR, description="Time window"),
    limit: int = Query(10, ge=1, le=100, description="Number of users")
):
    """
    Get top users by request volume.

    Returns users with highest request counts in specified time window.

    Example:
        GET /api/v1/ratelimit/top-users?tier=hour&limit=10

        Response:
        [
            {
                "user_id": "user_456",
                "requests": 8542,
                "tier": "hour"
            },
            {
                "user_id": "user_123",
                "requests": 7321,
                "tier": "hour"
            }
        ]
    """
    try:
        logger.info(f"Getting top users (tier={tier.value}, limit={limit})")

        limiter = get_rate_limiter()
        top_users = await limiter.get_top_users(tier, limit)

        return [TopUserResponse(**user) for user in top_users]

    except Exception as e:
        logger.error(f"Failed to get top users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{user_id}")
async def reset_user_limits(
    user_id: str = Path(..., description="User ID")
):
    """
    Reset all rate limits for a user.

    Clears all rate limit counters across all time windows.

    Example:
        POST /api/v1/ratelimit/reset/user_123

        Response:
        {
            "status": "success",
            "user_id": "user_123",
            "message": "Rate limits reset successfully"
        }
    """
    try:
        logger.info(f"Resetting rate limits for {user_id}")

        limiter = get_rate_limiter()
        await limiter.reset_user_limits(user_id)

        return {
            "status": "success",
            "user_id": user_id,
            "message": "Rate limits reset successfully"
        }

    except Exception as e:
        logger.error(f"Failed to reset user limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiers")
async def list_tiers():
    """
    List all available rate limit tiers.

    Returns tier definitions with limits for each time window.

    Example:
        GET /api/v1/ratelimit/tiers

        Response:
        {
            "tiers": {
                "free": {
                    "second": 2,
                    "minute": 60,
                    "hour": 1000,
                    "day": 10000
                },
                "basic": {
                    "second": 5,
                    "minute": 150,
                    "hour": 5000,
                    "day": 50000
                },
                ...
            }
        }
    """
    return {
        "tiers": {
            tier.value: {
                limit_tier.value: limit
                for limit_tier, limit in limits.items()
            }
            for tier, limits in DEFAULT_RATE_LIMITS.items()
        }
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint for rate limiting system.

    Returns:
        Status of rate limiting capabilities
    """
    try:
        limiter = get_rate_limiter()

        # Test basic operation
        test_status = await limiter.check_limit("health_check_user")

        return {
            "status": "healthy",
            "rate_limiting": "enabled",
            "features": [
                "multi-tier limits (second/minute/hour/day)",
                "user tier support (free/basic/pro/enterprise)",
                "custom limits per user",
                "distributed (Redis-backed)",
                "usage analytics",
                "top users tracking"
            ],
            "tiers": [t.value for t in UserTier],
            "windows": [t.value for t in RateLimitTier],
            "test_check": {
                "allowed": test_status.allowed,
                "limit": test_status.limit
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
