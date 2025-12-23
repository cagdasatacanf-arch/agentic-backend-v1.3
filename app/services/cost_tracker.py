"""
Cost Tracking Service

Comprehensive cost tracking and optimization:
- Token usage tracking per model
- Real-time cost calculation
- Budget limits and alerts
- User/session-based cost attribution
- Cost optimization recommendations

Benefits:
- 💰 Complete visibility into API spending
- 🎯 Budget controls prevent overages
- 📊 Cost attribution per user/agent
- 📈 Optimization recommendations
- 🔔 Real-time alerts

Usage:
    tracker = get_cost_tracker()

    # Track usage
    cost = await tracker.track_usage(
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        user_id="user_123"
    )

    # Check budget
    allowed = await tracker.check_budget(
        user_id="user_123",
        estimated_cost=0.05
    )
"""

from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class ModelPricing(str, Enum):
    """Supported models with pricing"""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    EMBEDDING_SMALL = "text-embedding-3-small"
    EMBEDDING_LARGE = "text-embedding-3-large"


@dataclass
class TokenUsage:
    """Token usage record"""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    user_id: Optional[str]
    session_id: Optional[str]
    agent_type: Optional[str]
    request_id: Optional[str]


@dataclass
class CostStats:
    """Cost statistics"""
    total_cost: float
    total_tokens: int
    total_requests: int
    cost_by_model: Dict[str, float]
    cost_by_agent: Dict[str, float]
    cost_by_user: Dict[str, float]
    period_start: datetime
    period_end: datetime


class CostTracker:
    """
    Track and manage API costs.

    Features:
    - Real-time token tracking
    - Cost calculation per model
    - User/session attribution
    - Budget enforcement
    - Cost analytics

    Example:
        tracker = CostTracker()

        # Track usage
        cost = await tracker.track_usage(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            user_id="user_123",
            agent_type="math"
        )

        # Get stats
        stats = await tracker.get_stats(
            user_id="user_123",
            period="today"
        )
    """

    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        ModelPricing.GPT4O: {
            "input": 2.50,   # $2.50 per 1M input tokens
            "output": 10.00  # $10.00 per 1M output tokens
        },
        ModelPricing.GPT4O_MINI: {
            "input": 0.150,  # $0.15 per 1M input tokens
            "output": 0.600  # $0.60 per 1M output tokens
        },
        ModelPricing.GPT4_TURBO: {
            "input": 10.00,
            "output": 30.00
        },
        ModelPricing.GPT35_TURBO: {
            "input": 0.50,
            "output": 1.50
        },
        ModelPricing.EMBEDDING_SMALL: {
            "input": 0.020,
            "output": 0.0  # Embeddings don't have output tokens
        },
        ModelPricing.EMBEDDING_LARGE: {
            "input": 0.130,
            "output": 0.0
        }
    }

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cost tracker.

        Args:
            redis_url: Redis connection URL
        """
        redis_host = settings.redis_host if hasattr(settings, 'redis_host') else "localhost"
        redis_port = settings.redis_port if hasattr(settings, 'redis_port') else 6379
        self.redis_url = redis_url or f"redis://{redis_host}:{redis_port}/5"  # DB 5 for costs
        self.redis = None

        logger.info("CostTracker initialized")

    async def _get_redis(self):
        """Get or create Redis connection"""
        if self.redis is None:
            self.redis = await redis.from_url(
                self.redis_url,
                decode_responses=False
            )
        return self.redis

    async def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> float:
        """
        Track token usage and calculate cost.

        Args:
            model: Model name (e.g., "gpt-4o")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            user_id: Optional user ID
            session_id: Optional session ID
            agent_type: Optional agent type (math/code/rag/vision)
            request_id: Optional request ID

        Returns:
            Cost in dollars

        Example:
            cost = await tracker.track_usage(
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                user_id="user_123"
            )
            # Returns: 0.0075 ($0.0075)
        """
        try:
            redis_client = await self._get_redis()

            # Calculate cost
            cost = self.calculate_cost(model, input_tokens, output_tokens)
            total_tokens = input_tokens + output_tokens

            # Create usage record
            usage = TokenUsage(
                timestamp=datetime.now(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
                user_id=user_id,
                session_id=session_id,
                agent_type=agent_type,
                request_id=request_id
            )

            # Store in Redis (multiple keys for different queries)
            timestamp_score = usage.timestamp.timestamp()
            usage_json = json.dumps({
                "timestamp": usage.timestamp.isoformat(),
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cost": usage.cost,
                "user_id": usage.user_id,
                "session_id": usage.session_id,
                "agent_type": usage.agent_type,
                "request_id": usage.request_id
            })

            # Store in sorted sets for time-based queries
            await redis_client.zadd(
                "costs:all",
                {usage_json: timestamp_score}
            )

            if user_id:
                await redis_client.zadd(
                    f"costs:user:{user_id}",
                    {usage_json: timestamp_score}
                )

            if agent_type:
                await redis_client.zadd(
                    f"costs:agent:{agent_type}",
                    {usage_json: timestamp_score}
                )

            if session_id:
                await redis_client.zadd(
                    f"costs:session:{session_id}",
                    {usage_json: timestamp_score}
                )

            # Update running totals
            await redis_client.incrbyfloat("costs:total", cost)
            await redis_client.incrby("costs:tokens:total", total_tokens)
            await redis_client.incr("costs:requests:total")

            if user_id:
                await redis_client.incrbyfloat(f"costs:total:user:{user_id}", cost)

            logger.info(
                f"Cost tracked: ${cost:.6f} ({input_tokens}+{output_tokens} tokens, "
                f"model={model}, user={user_id})"
            )

            return cost

        except Exception as e:
            logger.error(f"Cost tracking error: {e}", exc_info=True)
            # Return calculated cost even if storage fails
            return self.calculate_cost(model, input_tokens, output_tokens)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in dollars

        Example:
            cost = tracker.calculate_cost("gpt-4o", 1000, 500)
            # Returns: 0.0075
        """
        # Normalize model name
        model_key = model.lower()

        # Find pricing
        pricing = None
        for model_enum in ModelPricing:
            if model_enum.value in model_key:
                pricing = self.PRICING[model_enum]
                break

        if not pricing:
            logger.warning(f"Unknown model for pricing: {model}, using GPT-4o pricing")
            pricing = self.PRICING[ModelPricing.GPT4O]

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    async def get_stats(
        self,
        user_id: Optional[str] = None,
        period: str = "today",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> CostStats:
        """
        Get cost statistics.

        Args:
            user_id: Optional user ID to filter by
            period: Time period ("today", "week", "month", "all")
            start_time: Optional custom start time
            end_time: Optional custom end time

        Returns:
            Cost statistics

        Example:
            stats = await tracker.get_stats(
                user_id="user_123",
                period="today"
            )
        """
        try:
            redis_client = await self._get_redis()

            # Determine time range
            if start_time and end_time:
                min_score = start_time.timestamp()
                max_score = end_time.timestamp()
            elif period == "today":
                start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = datetime.now()
                min_score = start_time.timestamp()
                max_score = end_time.timestamp()
            elif period == "week":
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)
                min_score = start_time.timestamp()
                max_score = end_time.timestamp()
            elif period == "month":
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)
                min_score = start_time.timestamp()
                max_score = end_time.timestamp()
            else:  # all
                min_score = "-inf"
                max_score = "+inf"
                start_time = datetime.min
                end_time = datetime.now()

            # Get usage records
            if user_id:
                key = f"costs:user:{user_id}"
            else:
                key = "costs:all"

            records = await redis_client.zrangebyscore(
                key,
                min_score,
                max_score
            )

            # Parse and aggregate
            total_cost = 0.0
            total_tokens = 0
            total_requests = len(records)
            cost_by_model = {}
            cost_by_agent = {}
            cost_by_user = {}

            for record_bytes in records:
                record = json.loads(record_bytes)

                total_cost += record["cost"]
                total_tokens += record["total_tokens"]

                # By model
                model = record["model"]
                cost_by_model[model] = cost_by_model.get(model, 0.0) + record["cost"]

                # By agent
                agent = record.get("agent_type")
                if agent:
                    cost_by_agent[agent] = cost_by_agent.get(agent, 0.0) + record["cost"]

                # By user
                user = record.get("user_id")
                if user:
                    cost_by_user[user] = cost_by_user.get(user, 0.0) + record["cost"]

            return CostStats(
                total_cost=total_cost,
                total_tokens=total_tokens,
                total_requests=total_requests,
                cost_by_model=cost_by_model,
                cost_by_agent=cost_by_agent,
                cost_by_user=cost_by_user,
                period_start=start_time,
                period_end=end_time
            )

        except Exception as e:
            logger.error(f"Failed to get cost stats: {e}", exc_info=True)
            return CostStats(
                total_cost=0.0,
                total_tokens=0,
                total_requests=0,
                cost_by_model={},
                cost_by_agent={},
                cost_by_user={},
                period_start=datetime.now(),
                period_end=datetime.now()
            )

    async def check_budget(
        self,
        user_id: str,
        estimated_cost: float
    ) -> Dict:
        """
        Check if user has budget available.

        Args:
            user_id: User ID
            estimated_cost: Estimated cost of operation

        Returns:
            {
                "allowed": bool,
                "current_spend": float,
                "budget_limit": float,
                "remaining": float
            }

        Example:
            result = await tracker.check_budget("user_123", 0.05)
            if not result["allowed"]:
                raise Exception("Budget exceeded")
        """
        try:
            redis_client = await self._get_redis()

            # Get current spend
            current_spend_bytes = await redis_client.get(f"costs:total:user:{user_id}")
            current_spend = float(current_spend_bytes) if current_spend_bytes else 0.0

            # Get budget limit
            budget_limit_bytes = await redis_client.get(f"budget:limit:user:{user_id}")
            budget_limit = float(budget_limit_bytes) if budget_limit_bytes else 100.0  # Default $100

            # Check if allowed
            remaining = budget_limit - current_spend
            allowed = (current_spend + estimated_cost) <= budget_limit

            return {
                "allowed": allowed,
                "current_spend": current_spend,
                "budget_limit": budget_limit,
                "remaining": remaining,
                "estimated_cost": estimated_cost
            }

        except Exception as e:
            logger.error(f"Budget check error: {e}")
            # Default to allow on error
            return {
                "allowed": True,
                "current_spend": 0.0,
                "budget_limit": 0.0,
                "remaining": 0.0,
                "estimated_cost": estimated_cost,
                "error": str(e)
            }

    async def set_budget(
        self,
        user_id: str,
        budget_limit: float
    ):
        """
        Set budget limit for user.

        Args:
            user_id: User ID
            budget_limit: Budget limit in dollars

        Example:
            await tracker.set_budget("user_123", 50.0)  # $50 limit
        """
        try:
            redis_client = await self._get_redis()
            await redis_client.set(
                f"budget:limit:user:{user_id}",
                budget_limit
            )
            logger.info(f"Budget set for {user_id}: ${budget_limit}")

        except Exception as e:
            logger.error(f"Failed to set budget: {e}")

    async def get_optimization_recommendations(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get cost optimization recommendations.

        Args:
            user_id: Optional user ID

        Returns:
            List of recommendations

        Example:
            recs = await tracker.get_optimization_recommendations("user_123")
            for rec in recs:
                print(rec["recommendation"])
        """
        recommendations = []

        try:
            # Get stats
            stats = await self.get_stats(user_id=user_id, period="week")

            # Check model usage
            total_cost = stats.total_cost
            if total_cost > 0:
                for model, cost in stats.cost_by_model.items():
                    percentage = (cost / total_cost) * 100

                    # Recommend cheaper model if using expensive ones heavily
                    if "gpt-4" in model and percentage > 50:
                        recommendations.append({
                            "type": "model_optimization",
                            "severity": "high",
                            "current_model": model,
                            "recommendation": f"Consider using gpt-4o-mini for simple queries. "
                                            f"{percentage:.1f}% of costs are from {model}.",
                            "potential_savings": cost * 0.9  # 90% savings with mini
                        })

            # Check caching usage
            # (Would integrate with cache stats)
            recommendations.append({
                "type": "caching",
                "severity": "medium",
                "recommendation": "Enable semantic caching to reduce repeated queries.",
                "potential_savings": total_cost * 0.3  # Estimated 30% savings
            })

            # Check agent usage
            if stats.cost_by_agent:
                most_expensive_agent = max(
                    stats.cost_by_agent.items(),
                    key=lambda x: x[1]
                )
                recommendations.append({
                    "type": "agent_usage",
                    "severity": "low",
                    "recommendation": f"Agent '{most_expensive_agent[0]}' accounts for most costs. "
                                    "Consider optimizing its prompts.",
                    "agent": most_expensive_agent[0],
                    "cost": most_expensive_agent[1]
                })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []


# ============================================================================
# Singleton Instance
# ============================================================================

_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create global cost tracker instance"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
