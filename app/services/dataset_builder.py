"""
Training Dataset Builder

Creates training datasets from logged interactions for:
- SFT (Supervised Fine-Tuning): High-quality Q&A pairs
- DPO (Direct Preference Optimization): Ranked response pairs
- GRPO (Group Relative Policy Optimization): Grouped samples with rewards

Research basis:
- SFT: Train on high-quality demonstrations
- DPO: Learn from preferences without explicit reward model
- GRPO: Group-based policy optimization (DeepSeek, Orion papers)

Dataset formats follow HuggingFace standards for compatibility with:
- transformers Trainer
- trl (Transformer Reinforcement Learning library)
- OpenRLHF
"""

from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

from app.services.interaction_logger import InteractionLogger, Interaction

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Build training datasets from logged interactions.

    Supports:
    - SFT: Supervised fine-tuning on high-quality examples
    - DPO: Direct preference optimization from ranked pairs
    - GRPO: Group relative policy optimization

    Usage:
        builder = DatasetBuilder()

        # Build SFT dataset
        sft_dataset = await builder.build_sft_dataset(
            min_quality=0.8,
            max_samples=1000
        )

        # Save to file
        builder.save_dataset(sft_dataset, "data/sft_dataset.jsonl")
    """

    def __init__(self, interaction_logger: Optional[InteractionLogger] = None):
        """
        Initialize dataset builder.

        Args:
            interaction_logger: Logger instance (creates new if None)
        """
        if interaction_logger is None:
            from app.services.interaction_logger import get_interaction_logger
            interaction_logger = get_interaction_logger()

        self.logger = interaction_logger

        logger.info("DatasetBuilder initialized")

    async def build_sft_dataset(
        self,
        min_quality: float = 0.8,
        max_samples: Optional[int] = None,
        agent_types: Optional[List[str]] = None,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Build Supervised Fine-Tuning (SFT) dataset.

        Format (HuggingFace standard):
        [
            {
                "messages": [
                    {"role": "user", "content": "query"},
                    {"role": "assistant", "content": "answer"}
                ],
                "metadata": {
                    "quality_score": 0.95,
                    "agent_type": "math",
                    ...
                }
            },
            ...
        ]

        Args:
            min_quality: Minimum quality score (0.0-1.0)
            max_samples: Maximum samples to include
            agent_types: Filter by agent types
            days_back: Only use interactions from last N days

        Returns:
            SFT dataset (list of dicts)
        """
        logger.info(
            f"Building SFT dataset: min_quality={min_quality}, "
            f"max_samples={max_samples}, days_back={days_back}"
        )

        # Get high-quality interactions
        start_time = datetime.now() - timedelta(days=days_back)
        interactions = self.logger.get_interactions(
            start_time=start_time,
            high_quality_only=True,
            limit=max_samples or 10000
        )

        # Filter by quality and agent type
        filtered = []
        for interaction in interactions:
            # Skip if no quality scores
            if not interaction.quality_scores:
                continue

            # Check quality threshold
            overall_quality = interaction.quality_scores.get("overall", 0.0)
            if overall_quality < min_quality:
                continue

            # Check agent type filter
            if agent_types and interaction.agent_type not in agent_types:
                continue

            # Skip errors
            if interaction.error_occurred:
                continue

            filtered.append(interaction)

        # Convert to SFT format
        sft_dataset = []
        for interaction in filtered:
            sample = {
                "messages": [
                    {"role": "user", "content": interaction.query},
                    {"role": "assistant", "content": interaction.answer}
                ],
                "metadata": {
                    "interaction_id": interaction.interaction_id,
                    "quality_score": interaction.quality_scores.get("overall"),
                    "agent_type": interaction.agent_type,
                    "timestamp": interaction.timestamp.isoformat(),
                    "tools_used": interaction.tools_used,
                }
            }
            sft_dataset.append(sample)

        # Limit if requested
        if max_samples and len(sft_dataset) > max_samples:
            # Sort by quality (best first) and take top max_samples
            sft_dataset.sort(
                key=lambda x: x["metadata"]["quality_score"],
                reverse=True
            )
            sft_dataset = sft_dataset[:max_samples]

        logger.info(f"Built SFT dataset with {len(sft_dataset)} samples")
        return sft_dataset

    async def build_dpo_dataset(
        self,
        min_quality_diff: float = 0.2,
        max_pairs: Optional[int] = None,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Build Direct Preference Optimization (DPO) dataset.

        Format:
        [
            {
                "prompt": "query",
                "chosen": "high-quality answer",
                "rejected": "low-quality answer",
                "metadata": {
                    "chosen_quality": 0.9,
                    "rejected_quality": 0.5,
                    ...
                }
            },
            ...
        ]

        Finds pairs of answers to similar questions with different quality.

        Args:
            min_quality_diff: Minimum quality difference for pairs
            max_pairs: Maximum pairs to generate
            days_back: Only use interactions from last N days

        Returns:
            DPO dataset (list of dicts)
        """
        logger.info(
            f"Building DPO dataset: min_diff={min_quality_diff}, "
            f"max_pairs={max_pairs}"
        )

        # Get all interactions with quality scores
        start_time = datetime.now() - timedelta(days=days_back)
        interactions = self.logger.get_interactions(
            start_time=start_time,
            limit=max_pairs * 10 if max_pairs else 10000
        )

        # Group by similar queries
        query_groups = self._group_similar_queries(interactions)

        # Build pairs
        dpo_dataset = []

        for query, similar_interactions in query_groups.items():
            # Need at least 2 interactions to make a pair
            if len(similar_interactions) < 2:
                continue

            # Sort by quality
            similar_interactions.sort(
                key=lambda i: i.quality_scores.get("overall", 0.0) if i.quality_scores else 0.0,
                reverse=True
            )

            # Create pairs: best vs worst with sufficient quality gap
            for i in range(len(similar_interactions) - 1):
                chosen = similar_interactions[i]
                rejected = similar_interactions[-1]  # Worst

                # Check quality scores exist
                if not chosen.quality_scores or not rejected.quality_scores:
                    continue

                chosen_quality = chosen.quality_scores.get("overall", 0.0)
                rejected_quality = rejected.quality_scores.get("overall", 0.0)

                # Check quality difference
                if chosen_quality - rejected_quality < min_quality_diff:
                    continue

                pair = {
                    "prompt": query,
                    "chosen": chosen.answer,
                    "rejected": rejected.answer,
                    "metadata": {
                        "chosen_quality": chosen_quality,
                        "rejected_quality": rejected_quality,
                        "quality_diff": chosen_quality - rejected_quality,
                        "agent_type": chosen.agent_type
                    }
                }

                dpo_dataset.append(pair)

                # Limit pairs per query group
                if len(dpo_dataset) >= (max_pairs or float('inf')):
                    break

            if len(dpo_dataset) >= (max_pairs or float('inf')):
                break

        logger.info(f"Built DPO dataset with {len(dpo_dataset)} pairs")
        return dpo_dataset

    async def build_grpo_dataset(
        self,
        group_size: int = 4,
        max_groups: Optional[int] = None,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Build GRPO (Group Relative Policy Optimization) dataset.

        Format:
        [
            {
                "prompt": "query",
                "responses": [
                    {"text": "answer1", "reward": 0.9},
                    {"text": "answer2", "reward": 0.7},
                    {"text": "answer3", "reward": 0.5},
                    {"text": "answer4", "reward": 0.3}
                ],
                "metadata": {...}
            },
            ...
        ]

        Groups multiple responses to same query, each with reward score.

        Args:
            group_size: Number of responses per group
            max_groups: Maximum groups to generate
            days_back: Only use interactions from last N days

        Returns:
            GRPO dataset (list of dicts)
        """
        logger.info(
            f"Building GRPO dataset: group_size={group_size}, "
            f"max_groups={max_groups}"
        )

        # Get interactions
        start_time = datetime.now() - timedelta(days=days_back)
        interactions = self.logger.get_interactions(
            start_time=start_time,
            limit=max_groups * group_size * 2 if max_groups else 10000
        )

        # Group by similar queries
        query_groups = self._group_similar_queries(interactions)

        # Build GRPO groups
        grpo_dataset = []

        for query, similar_interactions in query_groups.items():
            # Need at least group_size interactions
            if len(similar_interactions) < group_size:
                continue

            # Filter interactions with quality scores
            valid_interactions = [
                i for i in similar_interactions
                if i.quality_scores and not i.error_occurred
            ]

            if len(valid_interactions) < group_size:
                continue

            # Sort by quality and take top group_size
            valid_interactions.sort(
                key=lambda i: i.quality_scores.get("overall", 0.0),
                reverse=True
            )

            group_interactions = valid_interactions[:group_size]

            # Build group
            group = {
                "prompt": query,
                "responses": [
                    {
                        "text": i.answer,
                        "reward": i.quality_scores.get("overall", 0.0)
                    }
                    for i in group_interactions
                ],
                "metadata": {
                    "agent_type": group_interactions[0].agent_type,
                    "group_size": len(group_interactions)
                }
            }

            grpo_dataset.append(group)

            if len(grpo_dataset) >= (max_groups or float('inf')):
                break

        logger.info(f"Built GRPO dataset with {len(grpo_dataset)} groups")
        return grpo_dataset

    def _group_similar_queries(
        self,
        interactions: List[Interaction]
    ) -> Dict[str, List[Interaction]]:
        """
        Group interactions by similar queries.

        For now, uses exact match. Could be enhanced with:
        - Embedding-based similarity
        - Edit distance
        - Semantic clustering

        Args:
            interactions: List of interactions

        Returns:
            Dict mapping query → list of interactions
        """
        groups = {}

        for interaction in interactions:
            # Normalize query (lowercase, strip whitespace)
            normalized_query = interaction.query.lower().strip()

            if normalized_query not in groups:
                groups[normalized_query] = []

            groups[normalized_query].append(interaction)

        # Filter out groups with only 1 interaction
        groups = {q: interactions for q, interactions in groups.items() if len(interactions) > 1}

        return groups

    def save_dataset(
        self,
        dataset: List[Dict],
        filepath: str,
        format: str = "jsonl"
    ) -> bool:
        """
        Save dataset to file.

        Args:
            dataset: Dataset to save
            filepath: Output file path
            format: File format ("jsonl" or "json")

        Returns:
            True if saved successfully
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            if format == "jsonl":
                with open(filepath, 'w') as f:
                    for sample in dataset:
                        f.write(json.dumps(sample) + '\n')
            else:  # json
                with open(filepath, 'w') as f:
                    json.dump(dataset, f, indent=2)

            logger.info(f"Saved dataset to {filepath}: {len(dataset)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False

    def load_dataset(self, filepath: str, format: str = "jsonl") -> List[Dict]:
        """
        Load dataset from file.

        Args:
            filepath: Input file path
            format: File format ("jsonl" or "json")

        Returns:
            Loaded dataset
        """
        try:
            if format == "jsonl":
                dataset = []
                with open(filepath, 'r') as f:
                    for line in f:
                        dataset.append(json.loads(line))
            else:  # json
                with open(filepath, 'r') as f:
                    dataset = json.load(f)

            logger.info(f"Loaded dataset from {filepath}: {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []

    async def get_dataset_stats(
        self,
        days_back: int = 30
    ) -> Dict:
        """
        Get statistics about available training data.

        Args:
            days_back: Days to look back

        Returns:
            Stats dict
        """
        start_time = datetime.now() - timedelta(days=days_back)

        # Get all interactions
        all_interactions = self.logger.get_interactions(
            start_time=start_time,
            limit=100000
        )

        # Calculate stats
        total = len(all_interactions)

        with_quality = [i for i in all_interactions if i.quality_scores]
        high_quality = [
            i for i in with_quality
            if i.quality_scores.get("overall", 0.0) >= 0.8
        ]

        by_agent = {}
        for interaction in all_interactions:
            agent = interaction.agent_type
            if agent not in by_agent:
                by_agent[agent] = 0
            by_agent[agent] += 1

        # Estimate dataset sizes
        sft_potential = len(high_quality)
        dpo_potential = len(self._group_similar_queries(with_quality))
        grpo_potential = sum(
            len(group) // 4  # Groups of 4
            for group in self._group_similar_queries(with_quality).values()
            if len(group) >= 4
        )

        return {
            "total_interactions": total,
            "with_quality_scores": len(with_quality),
            "high_quality": len(high_quality),
            "high_quality_rate": len(high_quality) / total if total > 0 else 0.0,
            "by_agent_type": by_agent,
            "potential_datasets": {
                "sft_samples": sft_potential,
                "dpo_pairs": dpo_potential,
                "grpo_groups": grpo_potential
            },
            "days_covered": days_back
        }
