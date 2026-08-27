"""Dependency-light public API for the Lean Reward Hacking experiment."""

from .config import (
    ConfigError,
    ExperimentConfig,
    LabelConfig,
    StatisticsConfig,
    canonical_config,
    config_hash,
    load_config,
)
from .episodes import (
    Episode,
    EpisodeBatch,
    PairedEpisode,
    collate,
    dataset_fingerprint,
    make_evaluation_pairs,
    make_paired_evaluation,
    make_training_episodes,
    pair_episodes,
)
from .evaluation import (
    EvaluationMetrics,
    ModeThresholds,
    classify_mode,
    evaluate_agent,
    label_endpoint,
    threshold_sensitivity,
)
from .rewards import RewardConfig, fixed_rlhf_reward, reward_for_action, reward_for_actions
from .types import Action, Mode

__all__ = [
    "Action",
    "ConfigError",
    "Episode",
    "EpisodeBatch",
    "EvaluationMetrics",
    "ExperimentConfig",
    "LabelConfig",
    "Mode",
    "ModeThresholds",
    "PairedEpisode",
    "RewardConfig",
    "StatisticsConfig",
    "canonical_config",
    "classify_mode",
    "collate",
    "config_hash",
    "dataset_fingerprint",
    "evaluate_agent",
    "fixed_rlhf_reward",
    "label_endpoint",
    "load_config",
    "make_evaluation_pairs",
    "make_paired_evaluation",
    "make_training_episodes",
    "pair_episodes",
    "reward_for_action",
    "reward_for_actions",
    "threshold_sensitivity",
]

