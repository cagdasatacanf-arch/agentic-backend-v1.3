"""
A/B Testing Framework for Model Selection and Feature Experimentation

Provides experiment management, variant selection, statistical analysis,
and automated winner selection for continuous improvement.

Features:
- Experiment creation and management
- Variant assignment with consistent hashing
- Statistical significance testing (Chi-square, T-test)
- Multi-armed bandit algorithms (Epsilon-greedy, UCB)
- Real-time analytics and reporting
- Automated winner selection

Usage:
    from app.services.ab_testing import (
        create_experiment,
        assign_variant,
        record_outcome,
        get_experiment_results
    )
    
    # Create experiment
    experiment = create_experiment(
        name="model_comparison",
        variants=["gpt-4", "gpt-3.5-turbo"],
        traffic_split=[0.5, 0.5]
    )
    
    # Assign variant to user
    variant = assign_variant(experiment_id, user_id)
    
    # Record outcome
    record_outcome(experiment_id, user_id, success=True, value=1.0)
    
    # Get results
    results = get_experiment_results(experiment_id)
"""

import hashlib
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class BanditAlgorithm(Enum):
    """Multi-armed bandit algorithms"""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class Variant:
    """Experiment variant"""
    id: str
    name: str
    traffic_allocation: float  # 0.0 to 1.0
    impressions: int = 0
    conversions: int = 0
    total_value: float = 0.0
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate"""
        return self.conversions / max(self.impressions, 1)
    
    @property
    def average_value(self) -> float:
        """Calculate average value per conversion"""
        return self.total_value / max(self.conversions, 1)


@dataclass
class Experiment:
    """A/B test experiment"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    bandit_algorithm: Optional[BanditAlgorithm] = None
    epsilon: float = 0.1  # For epsilon-greedy
    confidence_level: float = 0.95  # For statistical tests
    
    def get_variant_by_id(self, variant_id: str) -> Optional[Variant]:
        """Get variant by ID"""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None


# Global experiments storage
_experiments: Dict[str, Experiment] = {}
_user_assignments: Dict[str, Dict[str, str]] = {}  # {experiment_id: {user_id: variant_id}}


def create_experiment(
    name: str,
    description: str,
    variant_names: List[str],
    traffic_split: Optional[List[float]] = None,
    bandit_algorithm: Optional[str] = None
) -> Experiment:
    """
    Create a new A/B test experiment.
    
    Args:
        name: Experiment name
        description: Experiment description
        variant_names: List of variant names
        traffic_split: Traffic allocation for each variant (must sum to 1.0)
        bandit_algorithm: Optional bandit algorithm for dynamic allocation
        
    Returns:
        Created experiment
    """
    import uuid
    
    experiment_id = f"exp_{uuid.uuid4().hex[:16]}"
    
    # Default to equal split
    if traffic_split is None:
        traffic_split = [1.0 / len(variant_names)] * len(variant_names)
    
    # Validate traffic split
    if len(traffic_split) != len(variant_names):
        raise ValueError("Traffic split must match number of variants")
    
    if not math.isclose(sum(traffic_split), 1.0, rel_tol=1e-5):
        raise ValueError("Traffic split must sum to 1.0")
    
    # Create variants
    variants = []
    for i, name in enumerate(variant_names):
        variant = Variant(
            id=f"var_{i}",
            name=name,
            traffic_allocation=traffic_split[i]
        )
        variants.append(variant)
    
    # Create experiment
    experiment = Experiment(
        id=experiment_id,
        name=name,
        description=description,
        variants=variants,
        bandit_algorithm=BanditAlgorithm(bandit_algorithm) if bandit_algorithm else None
    )
    
    _experiments[experiment_id] = experiment
    _user_assignments[experiment_id] = {}
    
    logger.info(f"Created experiment: {name} ({experiment_id})")
    return experiment


def start_experiment(experiment_id: str):
    """Start an experiment"""
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    experiment = _experiments[experiment_id]
    experiment.status = ExperimentStatus.RUNNING
    experiment.started_at = datetime.utcnow()
    
    logger.info(f"Started experiment: {experiment.name}")


def stop_experiment(experiment_id: str):
    """Stop an experiment"""
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    experiment = _experiments[experiment_id]
    experiment.status = ExperimentStatus.COMPLETED
    experiment.ended_at = datetime.utcnow()
    
    logger.info(f"Stopped experiment: {experiment.name}")


def assign_variant(experiment_id: str, user_id: str) -> str:
    """
    Assign a variant to a user.
    
    Uses consistent hashing to ensure same user always gets same variant
    (unless using bandit algorithm).
    
    Args:
        experiment_id: Experiment ID
        user_id: User ID
        
    Returns:
        Assigned variant name
    """
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    experiment = _experiments[experiment_id]
    
    # Check if user already assigned
    if user_id in _user_assignments[experiment_id]:
        variant_id = _user_assignments[experiment_id][user_id]
        variant = experiment.get_variant_by_id(variant_id)
        return variant.name if variant else experiment.variants[0].name
    
    # Assign variant
    if experiment.bandit_algorithm:
        variant = _select_variant_bandit(experiment)
    else:
        variant = _select_variant_hash(experiment, user_id)
    
    # Record assignment
    _user_assignments[experiment_id][user_id] = variant.id
    variant.impressions += 1
    
    return variant.name


def _select_variant_hash(experiment: Experiment, user_id: str) -> Variant:
    """Select variant using consistent hashing"""
    # Hash user ID to get deterministic value
    hash_value = int(hashlib.md5(f"{experiment.id}:{user_id}".encode()).hexdigest(), 16)
    random_value = (hash_value % 10000) / 10000.0
    
    # Select variant based on traffic allocation
    cumulative = 0.0
    for variant in experiment.variants:
        cumulative += variant.traffic_allocation
        if random_value < cumulative:
            return variant
    
    return experiment.variants[-1]


def _select_variant_bandit(experiment: Experiment) -> Variant:
    """Select variant using bandit algorithm"""
    if experiment.bandit_algorithm == BanditAlgorithm.EPSILON_GREEDY:
        return _epsilon_greedy(experiment)
    elif experiment.bandit_algorithm == BanditAlgorithm.UCB:
        return _ucb(experiment)
    else:
        # Default to random
        return random.choice(experiment.variants)


def _epsilon_greedy(experiment: Experiment) -> Variant:
    """Epsilon-greedy bandit algorithm"""
    if random.random() < experiment.epsilon:
        # Explore: random variant
        return random.choice(experiment.variants)
    else:
        # Exploit: best variant
        return max(experiment.variants, key=lambda v: v.conversion_rate)


def _ucb(experiment: Experiment) -> Variant:
    """Upper Confidence Bound algorithm"""
    total_impressions = sum(v.impressions for v in experiment.variants)
    
    if total_impressions == 0:
        return random.choice(experiment.variants)
    
    def ucb_score(variant: Variant) -> float:
        if variant.impressions == 0:
            return float('inf')
        
        exploitation = variant.conversion_rate
        exploration = math.sqrt(2 * math.log(total_impressions) / variant.impressions)
        return exploitation + exploration
    
    return max(experiment.variants, key=ucb_score)


def record_outcome(
    experiment_id: str,
    user_id: str,
    success: bool,
    value: float = 0.0
):
    """
    Record experiment outcome.
    
    Args:
        experiment_id: Experiment ID
        user_id: User ID
        success: Whether the outcome was successful
        value: Optional value associated with outcome
    """
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    experiment = _experiments[experiment_id]
    
    # Get user's variant
    if user_id not in _user_assignments[experiment_id]:
        logger.warning(f"User {user_id} not assigned to experiment {experiment_id}")
        return
    
    variant_id = _user_assignments[experiment_id][user_id]
    variant = experiment.get_variant_by_id(variant_id)
    
    if not variant:
        logger.error(f"Variant {variant_id} not found in experiment {experiment_id}")
        return
    
    # Record outcome
    if success:
        variant.conversions += 1
        variant.total_value += value


def get_experiment_results(experiment_id: str) -> Dict[str, Any]:
    """
    Get experiment results with statistical analysis.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Experiment results including statistical significance
    """
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    experiment = _experiments[experiment_id]
    
    # Calculate results for each variant
    variant_results = []
    for variant in experiment.variants:
        variant_results.append({
            "id": variant.id,
            "name": variant.name,
            "impressions": variant.impressions,
            "conversions": variant.conversions,
            "conversion_rate": round(variant.conversion_rate, 4),
            "average_value": round(variant.average_value, 2),
            "total_value": round(variant.total_value, 2)
        })
    
    # Statistical significance test
    if len(experiment.variants) == 2:
        significance = _chi_square_test(experiment.variants[0], experiment.variants[1])
    else:
        significance = None
    
    # Determine winner
    winner = max(experiment.variants, key=lambda v: v.conversion_rate)
    
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "status": experiment.status.value,
        "variants": variant_results,
        "winner": {
            "variant_id": winner.id,
            "variant_name": winner.name,
            "conversion_rate": round(winner.conversion_rate, 4)
        },
        "statistical_significance": significance,
        "total_impressions": sum(v.impressions for v in experiment.variants),
        "total_conversions": sum(v.conversions for v in experiment.variants)
    }


def _chi_square_test(variant_a: Variant, variant_b: Variant) -> Dict[str, Any]:
    """
    Perform chi-square test for statistical significance.
    
    Returns:
        Test results including p-value and significance
    """
    # Observed values
    a_success = variant_a.conversions
    a_failure = variant_a.impressions - variant_a.conversions
    b_success = variant_b.conversions
    b_failure = variant_b.impressions - variant_b.conversions
    
    # Check if we have enough data
    if min(a_success, a_failure, b_success, b_failure) < 5:
        return {
            "test": "chi_square",
            "p_value": None,
            "is_significant": False,
            "message": "Insufficient data for statistical test (need at least 5 in each cell)"
        }
    
    # Calculate chi-square statistic
    total = a_success + a_failure + b_success + b_failure
    expected_a_success = (a_success + a_failure) * (a_success + b_success) / total
    expected_a_failure = (a_success + a_failure) * (a_failure + b_failure) / total
    expected_b_success = (b_success + b_failure) * (a_success + b_success) / total
    expected_b_failure = (b_success + b_failure) * (a_failure + b_failure) / total
    
    chi_square = (
        ((a_success - expected_a_success) ** 2 / expected_a_success) +
        ((a_failure - expected_a_failure) ** 2 / expected_a_failure) +
        ((b_success - expected_b_success) ** 2 / expected_b_success) +
        ((b_failure - expected_b_failure) ** 2 / expected_b_failure)
    )
    
    # Critical value for 95% confidence (df=1)
    critical_value = 3.841
    p_value = 1 - _chi_square_cdf(chi_square, df=1)
    
    return {
        "test": "chi_square",
        "chi_square_statistic": round(chi_square, 4),
        "p_value": round(p_value, 4),
        "is_significant": chi_square > critical_value,
        "confidence_level": 0.95
    }


def _chi_square_cdf(x: float, df: int) -> float:
    """Approximate chi-square CDF (simplified)"""
    # This is a simplified approximation
    # In production, use scipy.stats.chi2.cdf
    if x <= 0:
        return 0.0
    if x >= 20:
        return 1.0
    
    # Rough approximation for df=1
    return 1 - math.exp(-x / 2)


def list_experiments() -> List[Dict[str, Any]]:
    """List all experiments"""
    return [
        {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status.value,
            "variants": len(exp.variants),
            "created_at": exp.created_at.isoformat()
        }
        for exp in _experiments.values()
    ]


def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    if experiment_id in _experiments:
        del _experiments[experiment_id]
        del _user_assignments[experiment_id]
        logger.info(f"Deleted experiment: {experiment_id}")

def reset_experiment(experiment_id: str):
    """Reset experiment statistics and status to DRAFT.
    Clears variant metrics and user assignments.
    """
    if experiment_id not in _experiments:
        raise ValueError(f"Experiment {experiment_id} not found")
    experiment = _experiments[experiment_id]
    # Reset variant metrics
    for variant in experiment.variants:
        variant.impressions = 0
        variant.conversions = 0
        variant.total_value = 0.0
    # Reset status and timestamps
    experiment.status = ExperimentStatus.DRAFT
    experiment.started_at = None
    experiment.ended_at = None
    # Clear user assignments
    _user_assignments[experiment_id] = {}
    logger.info(f"Reset experiment: {experiment_id}")
