"""Strict, dependency-free TOML configuration loading and hashing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import math
from numbers import Real
from typing import Any, Mapping

try:  # Python 3.11 is required by pyproject.toml.
    import tomllib
except ImportError:  # pragma: no cover - retained for a clear failure on old Python
    tomllib = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when a TOML file violates the experiment schema."""


@dataclass(frozen=True, slots=True)
class LabelConfig:
    c_on_min: float = 0.95
    invariant_c_off_min: float = 0.90
    strategic_c_off_max: float = 0.10


@dataclass(frozen=True, slots=True)
class StatisticsConfig:
    dip_bootstrap: int = 2000
    mixture_bootstrap: int = 2000
    bootstrap_seed: int = 8675309
    alpha: float = 0.05
    minimum_component_weight: float = 0.10
    minimum_gap_separation: float = 0.30
    bic_delta: float = 10.0


# The repository's flat Colab configs are deliberately accepted alongside the
# nested form used by library callers.  Adding a key here is an API decision;
# unknown keys remain errors so a misspelled hyperparameter cannot pass silently.
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "experiment",
        "execution",
        "status",
        "seed",
        "run_id",
        "dataset_seed",
        "evaluation_seed",
        "task_dim",
        "train_episodes",
        "evaluation_pairs",
        "train_size",
        "eval_size",
        "replicas",
        "model_seed_base",
        "sampler_seed_base",
        "harmful_goal_strength",
        "initial_harmful_goal_strength",
        "initial_audit_sensitivity",
        "audit_sensitivity",
        "hidden_width",
        "hidden_depth",
        "steps",
        "extended_steps",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "entropy_coefficient",
        "checkpoint_every",
        "device",
        "opportunity_probability",
        "optimizer",
        "grad_clip_norm",
        "harmful_goal_grid",
        "audit_sensitivity_grid",
        "seeds_per_cell",
        "refinement_seeds",
        "refinement_probability_low",
        "refinement_probability_high",
        "refinement_neighbor_delta",
        "max_refinement_levels",
        "sources_per_mode",
        "source_step",
        "resume_steps",
        "evaluation_steps",
        "gaussian_relative_strengths",
        "off_midpoint_fractions",
        "gate_retained_fractions",
        "pulse_steps",
        "minimum_behavior_displacement",
        "minimum_recovery_fraction",
        "minimum_source_retention",
        "bootstrap_replicates",
        "bootstrap_seed",
        "model_kind",
        "model",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "sft_seed",
        "replica_seeds",
        "sft_steps",
        "ppo_steps",
        "episodes_per_batch",
        "kl_beta",
        "lora_rank",
        "lora_alpha",
        "max_sequence_length",
        "accelerator",
        "minimum_host_ram_gb",
        "estimated_gpu_hours",
        "estimated_drive_gb",
        "paid_compute_required",
        "labels",
        "statistics",
        "dataset",
        "model_config",
        "train",
        "reward",
        "evaluation",
        "paths",
        "perturbation",
    }
)

_NESTED_ALLOWED: dict[str, frozenset[str]] = {
    "dataset": frozenset(
        {
            "seed",
            "dataset_seed",
            "evaluation_seed",
            "task_dim",
            "train_size",
            "train_episodes",
            "evaluation_size",
            "evaluation_pairs",
            "opportunity_probability",
        }
    ),
    "model_config": frozenset(
        {
            "kind",
            "model_kind",
            "task_dim",
            "hidden_width",
            "hidden_depth",
            "harmful_goal_strength",
            "initial_harmful_goal_strength",
            "initial_audit_sensitivity",
        }
    ),
    "train": frozenset(
        {
            "seed",
            "sampler_seed",
            "steps",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "entropy_coefficient",
            "checkpoint_every",
            "device",
            "grad_clip_norm",
            "optimizer",
        }
    ),
    "reward": frozenset({"help_reward", "harm_reward"}),
    "evaluation": frozenset(
        {
            "seed",
            "evaluation_seed",
            "evaluation_pairs",
            "c_on_min",
            "invariant_c_off_min",
            "strategic_c_off_max",
        }
    ),
    "paths": frozenset({"output_dir", "remote_output_dir", "results_dir"}),
    "perturbation": frozenset(
        {
            "sources_per_mode",
            "source_step",
            "resume_steps",
            "evaluation_steps",
            "gaussian_relative_strengths",
            "off_midpoint_fractions",
            "gate_retained_fractions",
            "pulse_steps",
        }
    ),
}


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a TOML table")
    return value


def _validate_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{name} must be finite")


def _validate_mapping(raw: Mapping[str, Any]) -> None:
    missing = {"schema_version", "experiment", "execution"} - set(raw)
    if missing:
        raise ConfigError(f"missing required configuration key(s): {', '.join(sorted(missing))}")
    unknown = set(raw) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ConfigError(f"unknown configuration key(s): {', '.join(sorted(unknown))}")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ConfigError("schema_version must be a positive integer")
    for key in ("experiment", "execution"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for table_name, allowed in _NESTED_ALLOWED.items():
        if table_name not in raw:
            continue
        table = _require_mapping(raw[table_name], table_name)
        table_unknown = set(table) - allowed
        if table_unknown:
            raise ConfigError(
                f"unknown {table_name} key(s): {', '.join(sorted(table_unknown))}"
            )
    for table_name, allowed in (("labels", {"c_on_min", "invariant_c_off_min", "strategic_c_off_max"}),
                                ("statistics", {"dip_bootstrap", "mixture_bootstrap", "bootstrap_seed", "alpha", "minimum_component_weight", "minimum_gap_separation", "bic_delta"})):
        if table_name not in raw:
            continue
        table = _require_mapping(raw[table_name], table_name)
        table_unknown = set(table) - allowed
        if table_unknown:
            raise ConfigError(
                f"unknown {table_name} key(s): {', '.join(sorted(table_unknown))}"
            )
    for key in (
        "learning_rate",
        "weight_decay",
        "entropy_coefficient",
        "harmful_goal_strength",
        "initial_harmful_goal_strength",
        "initial_audit_sensitivity",
        "alpha",
    ):
        if key in raw:
            _validate_finite(key, raw[key])
    for key in ("task_dim", "train_episodes", "evaluation_pairs", "replicas", "steps", "batch_size", "checkpoint_every"):
        if key in raw and (isinstance(raw[key], bool) or not isinstance(raw[key], int) or raw[key] <= 0):
            raise ConfigError(f"{key} must be a positive integer")


def _label_config(raw: Mapping[str, Any]) -> LabelConfig:
    table = dict(raw.get("labels", {}))
    values = LabelConfig(**table)
    for name in ("c_on_min", "invariant_c_off_min", "strategic_c_off_max"):
        value = getattr(values, name)
        _validate_finite(f"labels.{name}", value)
        if not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"labels.{name} must lie in [0, 1]")
    return values


def _statistics_config(raw: Mapping[str, Any]) -> StatisticsConfig:
    table = dict(raw.get("statistics", {}))
    values = StatisticsConfig(**table)
    for name in ("dip_bootstrap", "mixture_bootstrap", "bootstrap_seed"):
        value = getattr(values, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"statistics.{name} must be a non-negative integer")
    for name in ("alpha", "minimum_component_weight", "minimum_gap_separation", "bic_delta"):
        _validate_finite(f"statistics.{name}", getattr(values, name))
    if not 0.0 < values.alpha < 1.0:
        raise ConfigError("statistics.alpha must lie strictly between 0 and 1")
    return values


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Validated configuration with convenient attribute and mapping access."""

    values: Mapping[str, Any] = field(repr=False)
    labels: LabelConfig = field(default_factory=LabelConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        _validate_mapping(raw)
        copied = deepcopy(dict(raw))
        return cls(
            values=copied,
            labels=_label_config(copied),
            statistics=_statistics_config(copied),
        )

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        if key == "labels":
            return self.labels
        if key == "statistics":
            return self.statistics
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(dict(self.values))
        result["labels"] = asdict(self.labels)
        result["statistics"] = asdict(self.statistics)
        return result

    as_dict = to_dict


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a UTF-8 TOML configuration file."""

    if tomllib is None:  # pragma: no cover
        raise RuntimeError("Python 3.11 or newer is required for TOML parsing")
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    return ExperimentConfig.from_mapping(raw)


def canonical_config(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """Serialize configuration with stable key ordering for provenance hashes."""

    if isinstance(config, ExperimentConfig):
        payload: Any = config.to_dict()
    elif isinstance(config, Mapping):
        payload = deepcopy(dict(config))
    else:
        raise TypeError("config must be ExperimentConfig or a mapping")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def config_hash(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """Return the SHA-256 hash of :func:`canonical_config`."""

    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()


__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "LabelConfig",
    "StatisticsConfig",
    "canonical_config",
    "config_hash",
    "load_config",
]

