"""
Interaction Logging Pipeline for Self-Improvement

Logs all agent interactions for future training:
- User queries
- Agent responses
- Quality scores
- Tool usage
- Performance metrics

Research basis:
- Data collection is critical for RL training
- Quality filtering improves training efficiency
- Interaction diversity prevents overfitting
- RLHF (Reinforcement Learning from Human Feedback) requires interaction data

Data uses:
- SFT (Supervised Fine-Tuning) on high-quality interactions
- DPO (Direct Preference Optimization) from ranked pairs
- GRPO (Group Relative Policy Optimization)
- Reward model training
- Performance analysis
"""

from typing import Dict, Optional, List
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import redis

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Single agent interaction record"""
    # IDs and metadata
    interaction_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    timestamp: datetime

    # Query and response
    query: str
    answer: str
    agent_type: str  # math, code, rag, general

    # Quality metrics (from Phase 1)
    quality_scores: Optional[Dict[str, float]] = None  # overall, completeness, etc.

    # Performance metrics
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None

    # Context
    tools_used: Optional[List[str]] = None
    sources: Optional[List[Dict]] = None
    retrieval_method: Optional[str] = None

    # Flags
    error_occurred: bool = False
    error_message: Optional[str] = None

    # User feedback (if available)
    user_rating: Optional[int] = None  # 1-5 stars
    user_feedback: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Interaction':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class InteractionLogger:
    """
    Logs all agent interactions to Redis for training data collection.

    Storage:
    - All interactions: Redis Sorted Set by timestamp
    - High-quality interactions: Separate sorted set
    - Session-based: Sorted set per session

    Usage:
        logger = InteractionLogger()

        interaction = Interaction(
            interaction_id="uuid",
            query="What is 2+2?",
            answer="4",
            agent_type="math",
            quality_scores={"overall": 0.95}
        )

        logger.log(interaction)
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize interaction logger.

        Args:
            redis_client: Redis connection (creates new if None)
        """
        if redis_client is None:
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=False
            )

        self.redis = redis_client

        # Redis keys
        self.all_interactions_key = "training:interactions:all"
        self.high_quality_key = "training:interactions:high_quality"
        self.by_agent_key_prefix = "training:interactions:agent:"
        self.by_session_key_prefix = "training:interactions:session:"

        # Quality threshold for "high quality"
        self.high_quality_threshold = 0.8  # >= 0.8 overall score

        logger.info("InteractionLogger initialized")

    def log(self, interaction: Interaction) -> bool:
        """
        Log an interaction.

        Args:
            interaction: Interaction to log

        Returns:
            True if logged successfully
        """
        try:
            score = interaction.timestamp.timestamp()
            value = interaction.to_json()

            # 1. Store in all interactions
            self.redis.zadd(self.all_interactions_key, {value: score})

            # 2. Store in agent-specific set
            agent_key = f"{self.by_agent_key_prefix}{interaction.agent_type}"
            self.redis.zadd(agent_key, {value: score})

            # 3. Store in session-specific set (if session_id exists)
            if interaction.session_id:
                session_key = f"{self.by_session_key_prefix}{interaction.session_id}"
                self.redis.zadd(session_key, {value: score})

            # 4. If high quality, store separately
            if self._is_high_quality(interaction):
                self.redis.zadd(self.high_quality_key, {value: score})

            # 5. Update stats
            self._update_stats(interaction)

            logger.debug(
                f"Logged interaction: {interaction.interaction_id}, "
                f"agent={interaction.agent_type}, "
                f"quality={interaction.quality_scores.get('overall', 0) if interaction.quality_scores else 'N/A'}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}", exc_info=True)
            return False

    def _is_high_quality(self, interaction: Interaction) -> bool:
        """Check if interaction is high quality"""
        if not interaction.quality_scores:
            return False

        overall = interaction.quality_scores.get("overall", 0.0)
        return overall >= self.high_quality_threshold and not interaction.error_occurred

    def _update_stats(self, interaction: Interaction) -> None:
        """Update interaction statistics"""
        try:
            stats_key = "training:stats"

            # Increment total count
            self.redis.hincrby(stats_key, "total_interactions", 1)

            # Increment agent-specific count
            agent_field = f"agent_{interaction.agent_type}_count"
            self.redis.hincrby(stats_key, agent_field, 1)

            # Update high quality count
            if self._is_high_quality(interaction):
                self.redis.hincrby(stats_key, "high_quality_count", 1)

            # Update error count
            if interaction.error_occurred:
                self.redis.hincrby(stats_key, "error_count", 1)

        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")

    def get_interactions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_type: Optional[str] = None,
        high_quality_only: bool = False,
        limit: int = 100
    ) -> List[Interaction]:
        """
        Retrieve interactions.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            agent_type: Filter by agent type
            high_quality_only: Only high-quality interactions
            limit: Maximum interactions to return

        Returns:
            List of interactions
        """
        try:
            # Determine which key to query
            if high_quality_only:
                key = self.high_quality_key
            elif agent_type:
                key = f"{self.by_agent_key_prefix}{agent_type}"
            else:
                key = self.all_interactions_key

            # Determine score range (timestamp range)
            min_score = start_time.timestamp() if start_time else '-inf'
            max_score = end_time.timestamp() if end_time else '+inf'

            # Get interactions
            records = self.redis.zrangebyscore(
                key,
                min_score,
                max_score,
                start=0,
                num=limit,
                withscores=False
            )

            # Deserialize
            interactions = []
            for record in records:
                try:
                    record_str = record.decode() if isinstance(record, bytes) else record
                    interaction = Interaction.from_json(record_str)
                    interactions.append(interaction)
                except Exception as e:
                    logger.warning(f"Failed to deserialize interaction: {e}")

            return interactions

        except Exception as e:
            logger.error(f"Failed to get interactions: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get interaction statistics.

        Returns:
            Stats dict
        """
        try:
            stats_key = "training:stats"
            stats = self.redis.hgetall(stats_key)

            # Decode and convert to int
            decoded_stats = {}
            for k, v in stats.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = int(v.decode() if isinstance(v, bytes) else v)
                decoded_stats[key] = value

            # Calculate derived stats
            total = decoded_stats.get("total_interactions", 0)
            high_quality = decoded_stats.get("high_quality_count", 0)
            errors = decoded_stats.get("error_count", 0)

            decoded_stats["high_quality_rate"] = (
                high_quality / total if total > 0 else 0.0
            )
            decoded_stats["error_rate"] = (
                errors / total if total > 0 else 0.0
            )

            return decoded_stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def clear_old_interactions(self, days_to_keep: int = 30) -> int:
        """
        Clear interactions older than specified days.

        Args:
            days_to_keep: Keep interactions from last N days

        Returns:
            Number of interactions removed
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

            removed = 0

            # Remove from all keys
            keys_to_clean = [
                self.all_interactions_key,
                self.high_quality_key,
            ]

            # Add agent-specific keys
            for agent_type in ["math", "code", "rag", "general"]:
                keys_to_clean.append(f"{self.by_agent_key_prefix}{agent_type}")

            for key in keys_to_clean:
                count = self.redis.zremrangebyscore(key, '-inf', cutoff_time)
                removed += count

            logger.info(f"Removed {removed} interactions older than {days_to_keep} days")
            return removed

        except Exception as e:
            logger.error(f"Failed to clear old interactions: {e}")
            return 0


# ============================================================================
# Singleton Instance
# ============================================================================

_interaction_logger: Optional[InteractionLogger] = None


def get_interaction_logger() -> InteractionLogger:
    """Get or create global interaction logger"""
    global _interaction_logger
    if _interaction_logger is None:
        _interaction_logger = InteractionLogger()
    return _interaction_logger


# ============================================================================
# Convenience Functions
# ============================================================================

def log_interaction(
    query: str,
    answer: str,
    agent_type: str,
    quality_scores: Optional[Dict[str, float]] = None,
    **kwargs
) -> bool:
    """
    Quick function to log an interaction.

    Args:
        query: User query
        answer: Agent answer
        agent_type: Agent type (math/code/rag/general)
        quality_scores: Quality metrics from Phase 1
        **kwargs: Additional fields for Interaction

    Returns:
        True if logged successfully
    """
    import uuid

    interaction = Interaction(
        interaction_id=str(uuid.uuid4()),
        session_id=kwargs.get('session_id'),
        user_id=kwargs.get('user_id'),
        timestamp=datetime.now(),
        query=query,
        answer=answer,
        agent_type=agent_type,
        quality_scores=quality_scores,
        latency_ms=kwargs.get('latency_ms', 0.0),
        tokens_used=kwargs.get('tokens_used'),
        cost_usd=kwargs.get('cost_usd'),
        tools_used=kwargs.get('tools_used'),
        sources=kwargs.get('sources'),
        retrieval_method=kwargs.get('retrieval_method'),
        error_occurred=kwargs.get('error_occurred', False),
        error_message=kwargs.get('error_message')
    )

    logger_instance = get_interaction_logger()
    return logger_instance.log(interaction)
