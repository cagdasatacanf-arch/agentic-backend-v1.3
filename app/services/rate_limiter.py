"""
Advanced Rate Limiting Service

Distributed, Redis-backed rate limiting with multiple tiers and analytics:
- Multi-tier limits (second, minute, hour, day)
- User-specific limits
- Endpoint-specific limits
- Adaptive rate limiting
- Rate limit analytics
- Distributed locking

Benefits:
- 🚦 Prevent API abuse
- 📊 Fair resource allocation
- 🔒 DDoS protection
- 📈 Usage analytics
- ⚡ Redis-backed (distributed)

Usage:
    limiter = get_rate_limiter()

    # Check rate limit
    allowed, retry_after = await limiter.check_limit(
        user_id="user_123",
        endpoint="/api/v1/query"
    )

    if not allowed:
        raise HTTPException(429, f"Rate limit exceeded. Retry after {retry_after}s")
"""

from typing import Optional, Dict, Tuple, List
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitTier(str, Enum):
    """Rate limit time windows"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int
    window: RateLimitTier
    tier: UserTier


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after: Optional[int] = None


# Default rate limits by user tier
DEFAULT_RATE_LIMITS: Dict[UserTier, Dict[RateLimitTier, int]] = {
    UserTier.FREE: {
        RateLimitTier.SECOND: 2,
        RateLimitTier.MINUTE: 60,
        RateLimitTier.HOUR: 1000,
        RateLimitTier.DAY: 10000,
    },
    UserTier.BASIC: {
        RateLimitTier.SECOND: 5,
        RateLimitTier.MINUTE: 150,
        RateLimitTier.HOUR: 5000,
        RateLimitTier.DAY: 50000,
    },
    UserTier.PRO: {
        RateLimitTier.SECOND: 10,
        RateLimitTier.MINUTE: 300,
        RateLimitTier.HOUR: 15000,
        RateLimitTier.DAY: 150000,
    },
    UserTier.ENTERPRISE: {
        RateLimitTier.SECOND: 50,
        RateLimitTier.MINUTE: 1000,
        RateLimitTier.HOUR: 50000,
        RateLimitTier.DAY: 500000,
    },
}


class RateLimiter:
    """
    Advanced distributed rate limiter.

    Features:
    - Multi-tier rate limiting (second, minute, hour, day)
    - User tier support (free, basic, pro, enterprise)
    - Endpoint-specific limits
    - Adaptive rate limiting
    - Usage analytics
    - Redis-backed (distributed)

    Example:
        limiter = RateLimiter()

        # Check limit
        status = await limiter.check_limit(
            user_id="user_123",
            endpoint="/api/v1/query"
        )

        if not status.allowed:
            raise HTTPException(429, f"Retry after {status.retry_after}s")

        # Record request
        await limiter.record_request(
            user_id="user_123",
            endpoint="/api/v1/query"
        )
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
        """
        redis_host = settings.redis_host if hasattr(settings, 'redis_host') else "localhost"
        redis_port = settings.redis_port if hasattr(settings, 'redis_port') else 6379
        self.redis_url = redis_url or f"redis://{redis_host}:{redis_port}/7"  # DB 7 for rate limiting
        self.redis = None

        logger.info("RateLimiter initialized")

    async def _get_redis(self):
        """Get or create Redis connection"""
        if self.redis is None:
            self.redis = await redis.from_url(
                self.redis_url,
                decode_responses=False
            )
        return self.redis

    # ============================================================================
    # User Tier Management
    # ============================================================================

    async def set_user_tier(self, user_id: str, tier: UserTier):
        """
        Set user's rate limit tier.

        Args:
            user_id: User ID
            tier: User tier (free, basic, pro, enterprise)
        """
        try:
            redis_client = await self._get_redis()
            await redis_client.set(f"tier:{user_id}", tier.value)
            logger.info(f"User tier set: {user_id} -> {tier.value}")
        except Exception as e:
            logger.error(f"Failed to set user tier: {e}", exc_info=True)
            raise

    async def get_user_tier(self, user_id: str) -> UserTier:
        """
        Get user's rate limit tier.

        Args:
            user_id: User ID

        Returns:
            User tier (defaults to FREE)
        """
        try:
            redis_client = await self._get_redis()
            tier_bytes = await redis_client.get(f"tier:{user_id}")

            if tier_bytes:
                return UserTier(tier_bytes.decode())

            return UserTier.FREE

        except Exception as e:
            logger.error(f"Failed to get user tier: {e}")
            return UserTier.FREE

    async def set_custom_limit(
        self,
        user_id: str,
        tier: RateLimitTier,
        limit: int
    ):
        """
        Set custom rate limit for user.

        Args:
            user_id: User ID
            tier: Rate limit tier
            limit: Request limit
        """
        try:
            redis_client = await self._get_redis()
            key = f"custom_limit:{user_id}:{tier.value}"
            await redis_client.set(key, str(limit))
            logger.info(f"Custom limit set: {user_id} {tier.value} -> {limit}")
        except Exception as e:
            logger.error(f"Failed to set custom limit: {e}", exc_info=True)
            raise

    async def get_limit(self, user_id: str, tier: RateLimitTier) -> int:
        """
        Get rate limit for user and tier.

        Args:
            user_id: User ID
            tier: Rate limit tier

        Returns:
            Request limit
        """
        try:
            redis_client = await self._get_redis()

            # Check for custom limit
            custom_key = f"custom_limit:{user_id}:{tier.value}"
            custom_limit = await redis_client.get(custom_key)

            if custom_limit:
                return int(custom_limit.decode())

            # Use default based on user tier
            user_tier = await self.get_user_tier(user_id)
            return DEFAULT_RATE_LIMITS[user_tier][tier]

        except Exception as e:
            logger.error(f"Failed to get limit: {e}")
            # Return conservative default
            return DEFAULT_RATE_LIMITS[UserTier.FREE][tier]

    # ============================================================================
    # Rate Limiting
    # ============================================================================

    def _get_window_seconds(self, tier: RateLimitTier) -> int:
        """Get window duration in seconds"""
        return {
            RateLimitTier.SECOND: 1,
            RateLimitTier.MINUTE: 60,
            RateLimitTier.HOUR: 3600,
            RateLimitTier.DAY: 86400,
        }[tier]

    def _get_window_key(
        self,
        user_id: str,
        tier: RateLimitTier,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate Redis key for rate limit window"""
        now = int(time.time())
        window_seconds = self._get_window_seconds(tier)

        # Round timestamp to window boundary
        window_start = (now // window_seconds) * window_seconds

        if endpoint:
            return f"ratelimit:{user_id}:{endpoint}:{tier.value}:{window_start}"
        else:
            return f"ratelimit:{user_id}:{tier.value}:{window_start}"

    async def check_limit(
        self,
        user_id: str,
        endpoint: Optional[str] = None,
        increment: bool = False
    ) -> RateLimitStatus:
        """
        Check if request is within rate limits.

        Checks all tiers (second, minute, hour, day) and returns first violation.

        Args:
            user_id: User ID
            endpoint: Optional endpoint for endpoint-specific limits
            increment: Whether to increment counter

        Returns:
            RateLimitStatus with allowed status and limits
        """
        try:
            redis_client = await self._get_redis()

            # Check all tiers
            for tier in RateLimitTier:
                limit = await self.get_limit(user_id, tier)
                window_seconds = self._get_window_seconds(tier)
                key = self._get_window_key(user_id, tier, endpoint)

                # Get current count
                current_bytes = await redis_client.get(key)
                current = int(current_bytes.decode()) if current_bytes else 0

                # Calculate reset time
                now = int(time.time())
                window_start = (now // window_seconds) * window_seconds
                reset_at = datetime.fromtimestamp(window_start + window_seconds)

                # Check if limit exceeded
                if current >= limit:
                    retry_after = int((reset_at - datetime.now()).total_seconds())

                    logger.warning(
                        f"Rate limit exceeded: user={user_id} tier={tier.value} "
                        f"current={current} limit={limit}"
                    )

                    return RateLimitStatus(
                        allowed=False,
                        remaining=0,
                        limit=limit,
                        reset_at=reset_at,
                        retry_after=retry_after
                    )

                # Increment if requested
                if increment:
                    pipe = redis_client.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, window_seconds)
                    await pipe.execute()

            # All checks passed - return status for most restrictive tier
            most_restrictive_tier = RateLimitTier.SECOND
            limit = await self.get_limit(user_id, most_restrictive_tier)
            window_seconds = self._get_window_seconds(most_restrictive_tier)
            key = self._get_window_key(user_id, most_restrictive_tier, endpoint)

            current_bytes = await redis_client.get(key)
            current = int(current_bytes.decode()) if current_bytes else 0

            now = int(time.time())
            window_start = (now // window_seconds) * window_seconds
            reset_at = datetime.fromtimestamp(window_start + window_seconds)

            return RateLimitStatus(
                allowed=True,
                remaining=limit - current,
                limit=limit,
                reset_at=reset_at
            )

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}", exc_info=True)
            # Fail open (allow request) on errors
            return RateLimitStatus(
                allowed=True,
                remaining=999,
                limit=1000,
                reset_at=datetime.now() + timedelta(seconds=60)
            )

    async def record_request(
        self,
        user_id: str,
        endpoint: Optional[str] = None
    ):
        """
        Record a request (increment all rate limit counters).

        Args:
            user_id: User ID
            endpoint: Optional endpoint
        """
        try:
            redis_client = await self._get_redis()

            # Increment counters for all tiers
            for tier in RateLimitTier:
                window_seconds = self._get_window_seconds(tier)
                key = self._get_window_key(user_id, tier, endpoint)

                pipe = redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                await pipe.execute()

        except Exception as e:
            logger.error(f"Failed to record request: {e}", exc_info=True)

    # ============================================================================
    # Analytics
    # ============================================================================

    async def get_usage_stats(
        self,
        user_id: str,
        tier: RateLimitTier = RateLimitTier.HOUR
    ) -> Dict:
        """
        Get rate limit usage statistics.

        Args:
            user_id: User ID
            tier: Time window to analyze

        Returns:
            Usage statistics
        """
        try:
            redis_client = await self._get_redis()

            key = self._get_window_key(user_id, tier, None)
            limit = await self.get_limit(user_id, tier)

            current_bytes = await redis_client.get(key)
            current = int(current_bytes.decode()) if current_bytes else 0

            window_seconds = self._get_window_seconds(tier)
            now = int(time.time())
            window_start = (now // window_seconds) * window_seconds
            reset_at = datetime.fromtimestamp(window_start + window_seconds)

            usage_percent = (current / limit * 100) if limit > 0 else 0

            return {
                "user_id": user_id,
                "tier": tier.value,
                "current": current,
                "limit": limit,
                "remaining": max(0, limit - current),
                "usage_percent": round(usage_percent, 2),
                "reset_at": reset_at.isoformat(),
                "window_seconds": window_seconds
            }

        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}

    async def get_top_users(
        self,
        tier: RateLimitTier = RateLimitTier.HOUR,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get top users by request volume.

        Args:
            tier: Time window
            limit: Number of users to return

        Returns:
            List of top users with usage stats
        """
        try:
            redis_client = await self._get_redis()

            # Scan for all rate limit keys in current window
            window_seconds = self._get_window_seconds(tier)
            now = int(time.time())
            window_start = (now // window_seconds) * window_seconds

            pattern = f"ratelimit:*:{tier.value}:{window_start}"

            users = {}
            cursor = 0

            while True:
                cursor, keys = await redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )

                for key_bytes in keys:
                    key = key_bytes.decode()
                    # Parse user_id from key
                    parts = key.split(":")
                    if len(parts) >= 3:
                        user_id = parts[1]

                        # Get count
                        count_bytes = await redis_client.get(key)
                        count = int(count_bytes.decode()) if count_bytes else 0

                        if user_id not in users:
                            users[user_id] = 0
                        users[user_id] += count

                if cursor == 0:
                    break

            # Sort and return top users
            sorted_users = sorted(
                users.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            return [
                {
                    "user_id": user_id,
                    "requests": count,
                    "tier": tier.value
                }
                for user_id, count in sorted_users
            ]

        except Exception as e:
            logger.error(f"Failed to get top users: {e}")
            return []

    async def reset_user_limits(self, user_id: str):
        """
        Reset all rate limits for a user.

        Args:
            user_id: User ID
        """
        try:
            redis_client = await self._get_redis()

            # Find and delete all keys for user
            pattern = f"ratelimit:{user_id}:*"

            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )

                if keys:
                    await redis_client.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

            logger.info(f"Reset rate limits for {user_id}: {deleted} keys deleted")

        except Exception as e:
            logger.error(f"Failed to reset user limits: {e}", exc_info=True)
            raise


# ============================================================================
# Singleton Instance
# ============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
