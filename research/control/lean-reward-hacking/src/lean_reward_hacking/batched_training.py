"""Vectorised training for independent toy or generic MLP replicas.

The bank in this module keeps one parameter tensor per parameter name.  The
first dimension of every tensor indexes a replica, so one Adam optimizer can
update all replicas in one call while its moments remain independent at every
leading-dimension slice.  Training data are copied to the selected device
once, then each replica receives the minibatch selected by its own CPU sampler
generator.

The module is optional-PyTorch at import time, like :mod:`training`.  The
public API is deliberately independent of the campaign runner so a Colab
notebook can opt into batched execution without changing the registered
single-replica objective.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import copy
import inspect
import math
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

try:  # Pinned in requirements-colab.txt, absent from the base install.
    import torch
    from torch import nn

    try:
        from torch.func import functional_call as _functional_call
        from torch.func import stack_module_state as _stack_module_state
        from torch.func import vmap as _vmap

        _HAS_TORCH_FUNC = True
    except (ImportError, AttributeError):  # pragma: no cover - old PyTorch path
        from torch.nn.utils.stateless import functional_call as _functional_call

        _stack_module_state = None
        _vmap = None
        _HAS_TORCH_FUNC = False

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _functional_call = None
    _stack_module_state = None
    _vmap = None
    _HAS_TORCH_FUNC = False
    TORCH_AVAILABLE = False

try:
    from .toy import TorchUnavailableError
except ImportError:  # pragma: no cover - direct file loading
    class TorchUnavailableError(RuntimeError):
        """Raised when a tensor/model operation needs optional PyTorch."""


ENGINE_SCHEMA_VERSION = 1


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "PyTorch is required for batched training; install the pinned Colab dependencies"
        )


def _configuration_value(config: object, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_dict(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return {
        field_info.name: getattr(config, field_info.name)
        for field_info in fields(config)
        if hasattr(config, field_info.name)
    } if hasattr(config, "__dataclass_fields__") else {"repr": repr(config)}


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class BatchedTrainingConfig:
    """Small config accepted directly by :meth:`ReplicaBank.train`.

    The fields intentionally match :class:`training.TrainingConfig`, so an
    existing config object can be passed without conversion.  ``steps`` is a
    target global step, which makes a checkpoint resumed with a larger target
    unambiguous.
    """

    steps: int = 4_000
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 1.0e-4
    kl_coefficient: float = 0.02
    entropy_coefficient: float | None = None
    grad_clip_norm: float | None = 1.0
    checkpoint_every_steps: int = 500
    checkpoint_every: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or int(self.steps) < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.batch_size, bool) or int(self.batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        for name in ("learning_rate", "weight_decay", "kl_coefficient"):
            _finite_nonnegative(getattr(self, name), name)
        if self.entropy_coefficient is not None:
            _finite_nonnegative(self.entropy_coefficient, "entropy_coefficient")
            object.__setattr__(self, "kl_coefficient", float(self.entropy_coefficient))
        if self.grad_clip_norm is not None:
            clip = float(self.grad_clip_norm)
            if not math.isfinite(clip) or clip <= 0.0:
                raise ValueError("grad_clip_norm must be positive when supplied")
        interval = (
            self.checkpoint_every
            if self.checkpoint_every is not None
            else self.checkpoint_every_steps
        )
        if isinstance(interval, bool) or int(interval) < 1:
            raise ValueError("checkpoint interval must be positive")
        object.__setattr__(self, "checkpoint_every_steps", int(interval))


@dataclass(frozen=True, slots=True)
class ReplicaSpec:
    """Stable seed identity for one bank slice."""

    index: int
    model_seed: int | None
    sampler_seed: int

    def to_payload(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "model_seed": None if self.model_seed is None else int(self.model_seed),
            "sampler_seed": int(self.sampler_seed),
        }


@dataclass(slots=True)
class BatchedTrainingState:
    """Cursor and deterministic sampler state for an interrupted bank run."""

    replica_count: int
    epoch: int = 0
    batch_offset: int = 0
    global_step: int = 0
    permutations: object | None = None
    sampler_states: tuple[object, ...] = ()
    history: list[dict[str, object]] = field(default_factory=list)
    completed: bool = False

    @property
    def permutation(self) -> object | None:
        """Singular compatibility alias for the single-replica cursor name."""

        return self.permutations

    @permutation.setter
    def permutation(self, value: object | None) -> None:
        self.permutations = value

    @property
    def sampler_state(self) -> tuple[object, ...]:
        """Compatibility alias returning all per-replica generator states."""

        return self.sampler_states

    @sampler_state.setter
    def sampler_state(self, value: object) -> None:
        if isinstance(value, (tuple, list)):
            self.sampler_states = tuple(value)
        else:
            self.sampler_states = (value,)

    def to_payload(self) -> dict[str, object]:
        """Return a plain mapping suitable for ``torch.save`` or JSON wrappers."""

        permutations = self.permutations
        if TORCH_AVAILABLE and isinstance(permutations, torch.Tensor):
            permutations = permutations.detach().to(device="cpu").clone()
        sampler_states: list[object] = []
        for value in self.sampler_states:
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                sampler_states.append(value.detach().to(device="cpu").clone())
            else:
                sampler_states.append(value)
        return {
            "replica_count": int(self.replica_count),
            "epoch": int(self.epoch),
            "batch_offset": int(self.batch_offset),
            "global_step": int(self.global_step),
            "permutations": permutations,
            "sampler_states": tuple(sampler_states),
            "history": copy.deepcopy(self.history),
            "completed": bool(self.completed),
        }

    @classmethod
    def from_payload(cls, value: object) -> "BatchedTrainingState":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("training_state must be a BatchedTrainingState or mapping")
        permutations = value.get("permutations", value.get("permutation"))
        sampler_value = value.get("sampler_states", value.get("sampler_state", ()))
        if sampler_value is None:
            sampler_states: tuple[object, ...] = ()
        elif isinstance(sampler_value, (tuple, list)):
            sampler_states = tuple(sampler_value)
        else:
            sampler_states = (sampler_value,)
        history_value = value.get("history", [])
        if not isinstance(history_value, list):
            history_value = list(history_value) if isinstance(history_value, Sequence) else []
        return cls(
            replica_count=_cursor_int(value.get("replica_count", 1), "replica_count"),
            epoch=_cursor_int(value.get("epoch", 0), "epoch"),
            batch_offset=_cursor_int(value.get("batch_offset", 0), "batch_offset"),
            global_step=_cursor_int(value.get("global_step", 0), "global_step"),
            permutations=permutations,
            sampler_states=sampler_states,
            history=copy.deepcopy(history_value),
            completed=_cursor_bool(value.get("completed", False), "completed"),
        )


def _cursor_int(value: object, name: str) -> int:
    """Parse a cursor integer without silently truncating malformed state."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _cursor_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _normalise_seed_sequence(
    value: Sequence[int] | int | None,
    replicas: int,
    *,
    default: int,
    name: str,
) -> tuple[int, ...]:
    if value is None:
        values = tuple(int(default) + index for index in range(replicas))
    elif isinstance(value, bool):
        raise ValueError(f"{name} must contain integer seeds")
    elif isinstance(value, int):
        values = tuple(int(value) + index for index in range(replicas))
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must be an integer or sequence of integers")
        values = tuple(value)
    if len(values) != replicas:
        raise ValueError(f"{name} must contain exactly {replicas} seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values):
        raise TypeError(f"{name} must contain integer seeds")
    return tuple(int(seed) for seed in values)


def _factory_accepts_seed(factory: Callable[..., object]) -> str | None:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return None
    parameters = signature.parameters
    if "model_seed" in parameters:
        return "model_seed"
    if "seed" in parameters:
        return "seed"
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return "model_seed"
    return None


def _factory_model(
    factory: Callable[..., object], kwargs: Mapping[str, object], seed: int
) -> object:
    call_kwargs = dict(kwargs)
    seed_name = _factory_accepts_seed(factory)
    if seed_name is not None and seed_name not in call_kwargs:
        call_kwargs[seed_name] = int(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory(**call_kwargs)


def _validate_model_structure(models: Sequence["nn.Module"]) -> None:
    first_parameters = tuple(models[0].named_parameters())
    first_buffers = tuple(models[0].named_buffers())
    for index, model in enumerate(models[1:], start=1):
        parameters = tuple(model.named_parameters())
        buffers = tuple(model.named_buffers())
        if tuple(name for name, _ in parameters) != tuple(name for name, _ in first_parameters):
            raise ValueError(f"replica {index} has a different parameter architecture")
        if tuple(name for name, _ in buffers) != tuple(name for name, _ in first_buffers):
            raise ValueError(f"replica {index} has a different buffer architecture")
        for (name, left), (_, right) in zip(first_parameters, parameters):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"parameter {name!r} differs at replica {index}")
        for (name, left), (_, right) in zip(first_buffers, buffers):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"buffer {name!r} differs at replica {index}")
        if model.training != models[0].training:
            raise ValueError("all replicas must have the same train/eval mode")


def _manual_stack_modules(
    models: Sequence["nn.Module"],
) -> tuple[dict[str, "torch.Tensor"], dict[str, "torch.Tensor"]]:
    parameter_maps = [dict(model.named_parameters()) for model in models]
    buffer_maps = [dict(model.named_buffers()) for model in models]
    parameters = {
        name: torch.stack([mapping[name].detach() for mapping in parameter_maps], dim=0)
        for name in parameter_maps[0]
    }
    buffers = {
        name: torch.stack([mapping[name].detach() for mapping in buffer_maps], dim=0)
        for name in buffer_maps[0]
    }
    return parameters, buffers


def _as_parameter_map(
    values: Mapping[str, "torch.Tensor"], replicas: int
) -> dict[str, "torch.nn.Parameter"]:
    result: dict[str, "torch.nn.Parameter"] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != replicas:
            raise ValueError(f"stacked parameter {name!r} must have leading replica dimension")
        result[name] = nn.Parameter(value.detach().clone(), requires_grad=True)
    return result


def _as_buffer_map(
    values: Mapping[str, "torch.Tensor"], replicas: int
) -> dict[str, "torch.Tensor"]:
    result: dict[str, "torch.Tensor"] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != replicas:
            raise ValueError(f"stacked buffer {name!r} must have leading replica dimension")
        result[name] = value.detach().clone()
    return result


def _reward_values(reward_config: object | None) -> tuple[float, float]:
    if reward_config is None:
        return 1.0, -1.0
    if isinstance(reward_config, Mapping):
        return float(reward_config.get("help_reward", 1.0)), float(
            reward_config.get("harm_reward", -1.0)
        )
    return float(getattr(reward_config, "help_reward", 1.0)), float(
        getattr(reward_config, "harm_reward", -1.0)
    )


def _capture_portable_rng_state() -> dict[str, object]:
    """Capture RNG streams using only weights-only-safe payload values."""

    from .training import capture_rng_state

    state = dict(capture_rng_state())
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5:
        generator, keys, position, has_gauss, cached_gaussian = numpy_state
        if hasattr(keys, "tolist"):
            keys = keys.tolist()
        state["numpy"] = {
            "generator": str(generator),
            "keys": [int(value) for value in keys],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    return state


def _restore_portable_rng_state(state: Mapping[str, object]) -> None:
    """Decode the portable NumPy state, then restore every recorded stream."""

    restored = dict(state)
    numpy_state = restored.get("numpy")
    if isinstance(numpy_state, Mapping) and "keys" in numpy_state:
        try:
            import numpy as np

            restored["numpy"] = (
                str(numpy_state["generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        except ImportError:
            restored.pop("numpy", None)
    from .training import restore_rng_state

    restore_rng_state(restored)


def _tensor_2d(value: object, *, name: str) -> "torch.Tensor":
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank two")
    return value.float()


def _feature_tensor(data: object) -> "torch.Tensor":
    """Convert an EpisodeBatch-like value to ``[samples, features]``."""

    if isinstance(data, torch.Tensor):
        if data.ndim not in (2, 3):
            raise ValueError("feature tensor must have rank two or three")
        return data.float()
    if isinstance(data, Mapping):
        if "all_features" in data:
            return _feature_tensor(data["all_features"])
        if not all(name in data for name in ("task", "audit", "secret_opportunity")):
            raise TypeError("training data mapping must expose all_features or episode fields")
        task = _tensor_2d(data["task"], name="task")
        audit = _tensor_2d(data["audit"], name="audit")
        opportunity = _tensor_2d(data["secret_opportunity"], name="secret_opportunity")
        if len({len(task), len(audit), len(opportunity)}) != 1:
            raise ValueError("all training fields must have the same length")
        return torch.cat((task, audit.to(task), opportunity.to(task)), dim=-1)
    if hasattr(data, "all_features"):
        return _feature_tensor(getattr(data, "all_features"))
    if all(hasattr(data, name) for name in ("task", "audit", "secret_opportunity")):
        return _feature_tensor(
            {
                "task": getattr(data, "task"),
                "audit": getattr(data, "audit"),
                "secret_opportunity": getattr(data, "secret_opportunity"),
            }
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        try:
            from .episodes import collate

            return _feature_tensor(collate(data))
        except (ImportError, TypeError) as exc:
            raise TypeError("training data must be a tensor, mapping, EpisodeBatch, or episodes") from exc
    raise TypeError("training data must be a tensor, mapping, EpisodeBatch, or episodes")


def _validate_feature_shape(features: "torch.Tensor", replicas: int | None = None) -> None:
    if features.ndim == 2:
        if features.shape[0] < 1 or features.shape[1] < 3:
            raise ValueError("feature tensor must have shape [samples, task..., audit, opportunity]")
    elif features.ndim == 3:
        if features.shape[0] < 1 or features.shape[1] < 1 or features.shape[2] < 3:
            raise ValueError("batched feature tensor must have shape [replicas, samples, features]")
        if replicas is not None and features.shape[0] != replicas:
            raise ValueError(f"feature tensor has {features.shape[0]} replicas, expected {replicas}")
    else:
        raise ValueError("feature tensor must have rank two or three")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("feature tensor must be finite")


def _assert_audited(features: "torch.Tensor") -> None:
    audit = features[..., -2]
    if not bool(torch.all(audit == 1).item()):
        raise ValueError("fixed RLHF training requires audit=ON for every episode")


def batched_loss_terms(
    logits: "torch.Tensor",
    *,
    help_reward: float = 1.0,
    harm_reward: float = -1.0,
    kl_coefficient: float = 0.02,
    l2_coefficient: float = 1.0e-4,
    parameters: Iterable["torch.Tensor"] | Mapping[str, "torch.Tensor"] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Return objective terms with one value per replica.

    ``logits`` has shape ``[replicas, minibatch, 2]``.  The expected reward
    and KL are averaged over each minibatch.  L2 sums only within each
    replica.  This is the exact scalar objective used by
    :func:`lean_reward_hacking.training.loss_terms`, lifted over replicas.
    """

    _require_torch()
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a PyTorch tensor")
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape [replicas, batch, 2]")
    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("logits replicas and batch cannot be empty")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    rewards = torch.as_tensor(
        [float(help_reward), float(harm_reward)], dtype=logits.dtype, device=logits.device
    )
    probs = torch.softmax(logits, dim=-1)
    expected_reward = (probs * rewards).sum(dim=-1).mean(dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    uniform_log_prob = math.log(0.5)
    kl = (probs * (log_probs - uniform_log_prob)).sum(dim=-1).mean(dim=-1)
    l2 = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    if parameters is not None:
        values = parameters.values() if isinstance(parameters, Mapping) else parameters
        for parameter in values:
            if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
                continue
            if parameter.ndim < 1 or parameter.shape[0] != logits.shape[0]:
                raise ValueError("every parameter must have the replica dimension first")
            l2 = l2 + parameter.square().reshape(logits.shape[0], -1).sum(dim=-1)
    reward_loss = -expected_reward
    total = reward_loss + float(kl_coefficient) * kl + 0.5 * float(l2_coefficient) * l2
    return {
        "loss": total,
        "reward_loss": reward_loss,
        "expected_reward": expected_reward,
        "kl": kl,
        "l2": l2,
        "help_probability": probs[..., 0].mean(dim=-1),
    }


batched_objective_loss = batched_loss_terms
replica_loss_terms = batched_loss_terms


class ReplicaBank:
    """A collection of independently initialised models with one optimizer."""

    def __init__(
        self,
        prototype: "nn.Module",
        parameters: Mapping[str, "torch.Tensor"],
        buffers: Mapping[str, "torch.Tensor"],
        *,
        model_seeds: Sequence[int] | None,
        sampler_seeds: Sequence[int],
        device: str | "torch.device" = "cpu",
        learning_rate: float | None = None,
        optimizer: "torch.optim.Optimizer | None" = None,
    ) -> None:
        _require_torch()
        self.prototype = prototype.to(device)
        self.prototype_device = torch.device(device)
        self.replica_count = len(tuple(sampler_seeds))
        if self.replica_count < 1:
            raise ValueError("a replica bank must contain at least one replica")
        self.model_seeds = None if model_seeds is None else tuple(int(seed) for seed in model_seeds)
        self.sampler_seeds = tuple(int(seed) for seed in sampler_seeds)
        if self.model_seeds is not None and len(self.model_seeds) != self.replica_count:
            raise ValueError("model seed count must equal replica count")
        if len(self.sampler_seeds) != self.replica_count:
            raise ValueError("sampler seed count must equal replica count")
        self._parameters = _as_parameter_map(parameters, self.replica_count)
        self._buffers = _as_buffer_map(buffers, self.replica_count)
        if not self._parameters:
            raise ValueError("replica models must have at least one trainable parameter")
        for parameter in self._parameters.values():
            parameter.data = parameter.data.to(device)
        self._buffers = {name: value.to(device) for name, value in self._buffers.items()}
        self._vmap_enabled = bool(_HAS_TORCH_FUNC and _functional_call is not None and _vmap is not None)
        self.optimizer = optimizer
        self.learning_rate = 0.003 if learning_rate is None else float(learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.optimizer is not None:
            self._validate_optimizer()
        else:
            self._ensure_optimizer(self.learning_rate)

    @classmethod
    def from_agents(
        cls,
        agents: Sequence["nn.Module"],
        *,
        model_seeds: Sequence[int] | None = None,
        sampler_seeds: Sequence[int] | int | None = None,
        sampler_seed: int | None = None,
        device: str | "torch.device" = "cpu",
        learning_rate: float | None = None,
    ) -> "ReplicaBank":
        """Stack already-created same-architecture agents without mutating them."""

        _require_torch()
        models = tuple(agents)
        if not models:
            raise ValueError("agents cannot be empty")
        if any(not isinstance(model, nn.Module) for model in models):
            raise TypeError("agents must be PyTorch modules")
        model_seed_values = None if model_seeds is None else tuple(model_seeds)
        if model_seed_values is not None and len(model_seed_values) != len(models):
            raise ValueError("model seed count must equal replica count")
        if sampler_seeds is not None and sampler_seed is not None:
            raise ValueError("supply sampler_seeds or sampler_seed, not both")
        seeds_value: Sequence[int] | int | None = sampler_seeds
        if sampler_seed is not None:
            seeds_value = sampler_seed
        seeds = _normalise_seed_sequence(
            seeds_value,
            len(models),
            default=0,
            name="sampler_seeds",
        )
        copies = tuple(copy.deepcopy(model).to(device) for model in models)
        _validate_model_structure(copies)
        try:
            if _stack_module_state is None:
                raise RuntimeError("torch.func.stack_module_state is unavailable")
            stacked_parameters, stacked_buffers = _stack_module_state(list(copies))
        except (RuntimeError, TypeError, ValueError):
            stacked_parameters, stacked_buffers = _manual_stack_modules(copies)
        return cls(
            copies[0],
            stacked_parameters,
            stacked_buffers,
            model_seeds=model_seed_values,
            sampler_seeds=seeds,
            device=device,
            learning_rate=learning_rate,
        )

    @classmethod
    def from_factory(
        cls,
        factory: Callable[..., "nn.Module"],
        replicas: int,
        *,
        model_seeds: Sequence[int] | int | None = None,
        model_seed: int | None = None,
        sampler_seeds: Sequence[int] | int | None = None,
        sampler_seed: int | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        device: str | "torch.device" = "cpu",
        learning_rate: float | None = None,
    ) -> "ReplicaBank":
        """Initialise replicas under isolated PyTorch RNG streams."""

        _require_torch()
        if isinstance(replicas, bool) or int(replicas) < 1:
            raise ValueError("replicas must be a positive integer")
        replicas = int(replicas)
        if model_seeds is not None and model_seed is not None:
            raise ValueError("supply model_seeds or model_seed, not both")
        if sampler_seeds is not None and sampler_seed is not None:
            raise ValueError("supply sampler_seeds or sampler_seed, not both")
        model_seed_value: Sequence[int] | int | None = model_seeds
        if model_seed is not None:
            model_seed_value = model_seed
        model_seed_tuple = _normalise_seed_sequence(
            model_seed_value,
            replicas,
            default=0,
            name="model_seeds",
        )
        kwargs = dict(factory_kwargs or {})
        models = tuple(
            _factory_model(factory, kwargs, seed) for seed in model_seed_tuple
        )
        if any(not isinstance(model, nn.Module) for model in models):
            raise TypeError("factory must return PyTorch modules")
        return cls.from_agents(
            models,
            model_seeds=model_seed_tuple,
            sampler_seeds=sampler_seeds,
            sampler_seed=sampler_seed,
            device=device,
            learning_rate=learning_rate,
        )

    def _validate_optimizer(self) -> None:
        if self.optimizer is None:
            return
        parameters = tuple(self._parameters.values())
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        if optimizer_parameters != parameters:
            raise ValueError("optimizer must own exactly the bank's stacked parameters")

    def _ensure_optimizer(self, learning_rate: float | None = None) -> "torch.optim.Optimizer":
        _require_torch()
        if self.optimizer is None:
            rate = self.learning_rate if learning_rate is None else float(learning_rate)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError("learning_rate must be finite and positive")
            self.learning_rate = rate
            self.optimizer = torch.optim.Adam(tuple(self._parameters.values()), lr=rate)
        self._validate_optimizer()
        return self.optimizer

    @property
    def device(self) -> "torch.device":
        return self.prototype_device

    @property
    def parameters_by_name(self) -> Mapping[str, "torch.nn.Parameter"]:
        """Read-only-by-convention view of stacked trainable parameters."""

        return self._parameters

    def parameters(self) -> tuple["torch.nn.Parameter", ...]:
        """Return all stacked parameters in module declaration order."""

        return tuple(self._parameters.values())

    def buffers(self) -> Mapping[str, "torch.Tensor"]:
        return self._buffers

    def _functional_call_one(
        self,
        parameters: Mapping[str, "torch.Tensor"],
        buffers: Mapping[str, "torch.Tensor"],
        features: "torch.Tensor",
    ) -> "torch.Tensor":
        try:
            return _functional_call(
                self.prototype,
                (parameters, buffers),
                (features,),
                strict=False,
            )
        except TypeError as error:
            if "strict" not in str(error):
                raise
            return _functional_call(self.prototype, (parameters, buffers), (features,))

    def forward(self, features: object) -> "torch.Tensor":
        """Evaluate all replicas and return logits shaped ``[R, B, 2]``."""

        _require_torch()
        values = _feature_tensor(features)
        _validate_feature_shape(values, self.replica_count if values.ndim == 3 else None)
        if values.ndim == 2:
            values = values.unsqueeze(0).expand(self.replica_count, -1, -1)
        values = values.to(self.device)
        if self._vmap_enabled:
            try:
                output = _vmap(self._functional_call_one, in_dims=(0, 0, 0))(
                    self._parameters, self._buffers, values
                )
            except RuntimeError:
                # Some user modules contain an operation unavailable to vmap.
                # The explicit path retains the same stacked state and exact
                # gradients, with a lower throughput bound for that module.
                self._vmap_enabled = False
                output = torch.stack(
                    [
                        self._functional_call_one(
                            {name: value[index] for name, value in self._parameters.items()},
                            {name: value[index] for name, value in self._buffers.items()},
                            values[index],
                        )
                        for index in range(self.replica_count)
                    ],
                    dim=0,
                )
        else:
            output = torch.stack(
                [
                    self._functional_call_one(
                        {name: value[index] for name, value in self._parameters.items()},
                        {name: value[index] for name, value in self._buffers.items()},
                        values[index],
                    )
                    for index in range(self.replica_count)
                ],
                dim=0,
            )
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] != 2:
            raise ValueError("replica models must return logits with shape [batch, 2]")
        return output

    __call__ = forward
    logits = forward

    def help_probabilities(self, features: object) -> "torch.Tensor":
        return torch.softmax(self.forward(features), dim=-1)[..., 0]

    help_probability = help_probabilities

    def replica_parameter_state(self, replica: int) -> dict[str, "torch.Tensor"]:
        """Return a detached parameter mapping for one replica."""

        if isinstance(replica, bool) or int(replica) < 0 or int(replica) >= self.replica_count:
            raise IndexError("replica index is out of range")
        index = int(replica)
        return {
            name: value[index].detach().clone()
            for name, value in self._parameters.items()
        }

    def replica_state_dict(self, replica: int) -> dict[str, "torch.Tensor"]:
        """Return one scalar model's parameters and buffers by module name."""

        if isinstance(replica, bool) or int(replica) < 0 or int(replica) >= self.replica_count:
            raise IndexError("replica index is out of range")
        index = int(replica)
        values = {
            name: value[index].detach().clone()
            for name, value in self._parameters.items()
        }
        values.update(
            {
                name: value[index].detach().clone()
                for name, value in self._buffers.items()
            }
        )
        return values

    def replica_optimizer_state(self, replica: int) -> dict[str, dict[str, object]]:
        """Return Adam state keyed by parameter name for one scalar replica."""

        if isinstance(replica, bool) or int(replica) < 0 or int(replica) >= self.replica_count:
            raise IndexError("replica index is out of range")
        optimizer = self._ensure_optimizer()
        index = int(replica)
        result: dict[str, dict[str, object]] = {}
        for name, parameter in self._parameters.items():
            state = optimizer.state.get(parameter, {})
            sliced: dict[str, object] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.replica_count:
                    sliced[key] = value[index].detach().clone()
                elif isinstance(value, torch.Tensor):
                    sliced[key] = value.detach().clone()
                else:
                    sliced[key] = copy.deepcopy(value)
            result[name] = sliced
        return result

    def materialize_replica(self, replica: int, *, device: str | "torch.device" | None = None) -> "nn.Module":
        """Materialize one bank slice as the original scalar model class."""

        model = copy.deepcopy(self.prototype)
        if device is not None:
            model = model.to(device)
        target = model.state_dict()
        for name, value in self.replica_state_dict(replica).items():
            if name in target:
                target[name].copy_(value.to(device=target[name].device, dtype=target[name].dtype))
        model.load_state_dict(target, strict=True)
        return model

    def materialize_replica_with_optimizer(
        self,
        replica: int,
        *,
        device: str | "torch.device" | None = None,
    ) -> tuple["nn.Module", "torch.optim.Optimizer"]:
        """Materialize one scalar model and its corresponding Adam state."""

        model = self.materialize_replica(replica, device=device)
        source = self._ensure_optimizer()
        source_group = source.param_groups[0]
        options = {
            key: copy.deepcopy(value)
            for key, value in source_group.items()
            if key != "params"
        }
        optimizer = torch.optim.Adam(model.parameters(), **options)
        by_name = self.replica_optimizer_state(replica)
        for name, parameter in model.named_parameters():
            if name in by_name:
                optimizer.state[parameter] = {
                    key: value.to(device=parameter.device, dtype=parameter.dtype)
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value.to(device=parameter.device)
                    if isinstance(value, torch.Tensor)
                    else copy.deepcopy(value)
                    for key, value in by_name[name].items()
                }
        return model, optimizer

    def replica_specs(self) -> tuple[ReplicaSpec, ...]:
        """Return seed identities in deterministic replica order."""

        model_seeds = self.model_seeds
        return tuple(
            ReplicaSpec(
                index=index,
                model_seed=None if model_seeds is None else model_seeds[index],
                sampler_seed=self.sampler_seeds[index],
            )
            for index in range(self.replica_count)
        )

    def manifest(self) -> dict[str, object]:
        """Return a stable, JSON-compatible bank description."""

        parameter_names = tuple(self._parameters)
        buffer_names = tuple(self._buffers)
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "engine": "lean_reward_hacking.batched_training",
            "architecture": f"{type(self.prototype).__module__}.{type(self.prototype).__qualname__}",
            "replica_count": int(self.replica_count),
            "replicas": tuple(spec.to_payload() for spec in self.replica_specs()),
            "parameter_names": parameter_names,
            "parameter_shapes": {name: tuple(self._parameters[name].shape[1:]) for name in parameter_names},
            "buffer_names": buffer_names,
            "buffer_shapes": {name: tuple(self._buffers[name].shape[1:]) for name in buffer_names},
            "dtype": str(next(iter(self._parameters.values())).dtype),
            "device": str(self.device),
            "backend": "torch.func.vmap" if self._vmap_enabled else "explicit-functional-call",
            "deterministic_sampler": True,
        }

    def _clip_gradients(self, max_norm: float | None) -> "torch.Tensor":
        gradients = [parameter.grad for parameter in self._parameters.values() if parameter.grad is not None]
        if not gradients or max_norm is None:
            return torch.zeros(self.replica_count, device=self.device)
        # Match torch.nn.utils.clip_grad_norm_: take one vector norm per
        # parameter tensor, then combine those norms.  The replica dimension
        # stays independent throughout.
        per_parameter = torch.stack(
            [
                torch.linalg.vector_norm(
                    gradient.detach().reshape(self.replica_count, -1), ord=2, dim=1
                )
                for gradient in gradients
            ],
            dim=0,
        )
        norm = torch.linalg.vector_norm(per_parameter, ord=2, dim=0)
        limit = float(max_norm)
        scale = (limit / (norm + 1.0e-6)).clamp(max=1.0)
        for parameter in self._parameters.values():
            if parameter.grad is None:
                continue
            shape = (self.replica_count,) + (1,) * (parameter.grad.ndim - 1)
            parameter.grad.mul_(scale.reshape(shape))
        return norm

    def step(
        self,
        minibatch: object,
        reward_config: object | None = None,
        *,
        kl_coefficient: float = 0.02,
        l2_coefficient: float = 1.0e-4,
        entropy_coefficient: float | None = None,
        grad_clip_norm: float | None = 1.0,
    ) -> dict[str, "torch.Tensor"]:
        """Apply one shared minibatch update and return per-replica metrics."""

        _require_torch()
        features = _feature_tensor(minibatch)
        _validate_feature_shape(features, self.replica_count if features.ndim == 3 else None)
        _assert_audited(features)
        if features.ndim == 2:
            features = features.unsqueeze(0).expand(self.replica_count, -1, -1)
        features = features.to(self.device)
        optimizer = self._ensure_optimizer()
        optimizer.zero_grad(set_to_none=True)
        logits = self.forward(features)
        if entropy_coefficient is not None:
            kl_coefficient = float(entropy_coefficient)
        help_reward, harm_reward = _reward_values(reward_config)
        terms = batched_loss_terms(
            logits,
            help_reward=help_reward,
            harm_reward=harm_reward,
            kl_coefficient=kl_coefficient,
            l2_coefficient=l2_coefficient,
            parameters=self._parameters,
        )
        # Sum retains each replica's scalar gradient.  A mean would rescale
        # Adam's inputs and break equivalence with independent optimizers.
        terms["loss"].sum().backward()
        grad_norm = self._clip_gradients(grad_clip_norm)
        optimizer.step()
        return {
            name: value.detach()
            for name, value in {
                **terms,
                "grad_norm": grad_norm,
            }.items()
        }

    train_step = step

    def _new_sampler_generators(self) -> list["torch.Generator"]:
        generators: list["torch.Generator"] = []
        for seed in self.sampler_seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    def _coerce_state(
        self,
        state: BatchedTrainingState | Mapping[str, object] | None,
        *,
        n_samples: int | None = None,
    ) -> BatchedTrainingState:
        """Decode and validate a complete bank cursor.

        A bank checkpoint is only resumable when every replica has the same
        sample-count permutation and an independent CPU generator state.  A
        shape-only check lets duplicate indices, floating-point truncation,
        and half-written cursors silently alter the next minibatch, so the
        validation below is deliberately strict.
        """

        result = (
            BatchedTrainingState(self.replica_count)
            if state is None
            else BatchedTrainingState.from_payload(state)
        )
        if result.replica_count != self.replica_count:
            raise ValueError("checkpoint replica count does not match this bank")
        for name in ("replica_count", "epoch", "batch_offset", "global_step"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"checkpoint cursor field {name!r} is invalid")
        if type(result.completed) is not bool:
            raise ValueError("checkpoint cursor field 'completed' is invalid")

        if not isinstance(result.history, list):
            raise ValueError("checkpoint history must be a list")
        previous_step = 0
        for row in result.history:
            if not isinstance(row, Mapping):
                raise ValueError("checkpoint history rows must be mappings")
            if "step" not in row:
                raise ValueError("checkpoint history row has no step")
            row_step = row["step"]
            if isinstance(row_step, bool) or not isinstance(row_step, Integral):
                raise ValueError("checkpoint history step must be an integer")
            row_step = int(row_step)
            if row_step < 1 or row_step > result.global_step or row_step <= previous_step:
                raise ValueError("checkpoint history steps are not strictly increasing")
            previous_step = row_step

        if result.permutations is None:
            if result.global_step != 0 or result.epoch != 0 or result.batch_offset != 0:
                raise ValueError("checkpoint is missing the permutation cursor")
            if result.sampler_states:
                raise ValueError("checkpoint has sampler states without a permutation")
        else:
            raw_permutations = result.permutations
            if isinstance(raw_permutations, torch.Tensor):
                if raw_permutations.dtype not in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("checkpoint permutations must have an integer dtype")
                permutations = raw_permutations.detach().to(device="cpu")
            else:
                # Validate list values before conversion.  ``as_tensor(...,
                # dtype=long)`` would otherwise turn 1.5 into 1.
                def _check_integer(value: object) -> None:
                    if isinstance(value, bool) or not isinstance(value, Integral):
                        raise ValueError("checkpoint permutations must contain integers")

                if not isinstance(raw_permutations, Sequence) or isinstance(raw_permutations, (str, bytes)):
                    raise ValueError("checkpoint permutations must be a rank-two sequence")
                for row in raw_permutations:
                    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                        raise ValueError("checkpoint permutations must be a rank-two sequence")
                    for item in row:
                        _check_integer(item)
                try:
                    permutations = torch.as_tensor(raw_permutations, dtype=torch.long, device="cpu")
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError("checkpoint permutations are not tensor-like") from exc
            if permutations.ndim != 2 or permutations.shape[0] != self.replica_count:
                raise ValueError("checkpoint permutations have the wrong shape")
            if n_samples is not None and permutations.shape[1] != int(n_samples):
                raise ValueError("checkpoint permutations have the wrong sample count")
            if result.global_step < 1:
                raise ValueError("a non-empty permutation requires a positive global step")
            permutation_size = int(permutations.shape[1])
            if result.batch_offset < 1 or result.batch_offset > permutation_size:
                raise ValueError("checkpoint batch_offset is outside the sample range")
            # Validate the permutation even when no training batch is available
            # to the loader.  Payload-only recovery must reject duplicate,
            # negative, and out-of-range indices before restoring the cursor.
            expected = torch.arange(permutation_size, dtype=torch.long)
            for row in permutations:
                if not bool(torch.equal(torch.sort(row).values, expected)):
                    raise ValueError("checkpoint permutation is not a complete permutation")
            result.permutations = permutations.to(device="cpu", dtype=torch.long)

            if len(result.sampler_states) != self.replica_count:
                raise ValueError("checkpoint sampler state count does not match replica count")
            expected_state = torch.Generator(device="cpu").get_state()
            normalized_states: list[torch.Tensor] = []
            for value in result.sampler_states:
                if isinstance(value, torch.Tensor):
                    if value.dtype is not torch.uint8 and value.dtype != torch.uint8:
                        raise ValueError("checkpoint sampler state must be uint8")
                    sampler_state = value.detach().to(device="cpu")
                else:
                    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                        raise ValueError("checkpoint sampler state must be a byte sequence")
                    if any(isinstance(item, bool) or not isinstance(item, Integral) or not 0 <= int(item) <= 255 for item in value):
                        raise ValueError("checkpoint sampler state must contain uint8 values")
                    sampler_state = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
                if sampler_state.ndim != 1 or sampler_state.shape != expected_state.shape:
                    raise ValueError("checkpoint sampler state has the wrong shape")
                normalized_states.append(sampler_state.clone())
            result.sampler_states = tuple(normalized_states)
        return result

    def train(
        self,
        train_data: object,
        config: object | None = None,
        reward_config: object | None = None,
        *,
        steps: int | None = None,
        state: BatchedTrainingState | Mapping[str, object] | None = None,
        resume_state: BatchedTrainingState | Mapping[str, object] | None = None,
        resume_from: str | os.PathLike[str] | None = None,
        expected_config: Mapping[str, object] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        checkpoint_callback: Callable[..., object] | None = None,
        config_for_checkpoint: object | None = None,
    ) -> BatchedTrainingState:
        """Train to a target step while preserving each sampler cursor.

        ``state`` and ``resume_state`` are aliases.  ``resume_from`` loads the
        model, optimizer, RNG, and cursor before the first update.  A target
        ``steps`` value is interpreted globally, so resuming a two-step run to
        step five executes exactly three additional updates.
        """

        _require_torch()
        if state is not None and resume_state is not None:
            raise ValueError("supply state or resume_state, not both")
        effective_config = config if config is not None else BatchedTrainingConfig()
        from .training import require_colab_for_full_run

        guard_config = _config_dict(effective_config)
        guard_config["replicas"] = self.replica_count
        guard_config["device"] = str(self.device)
        require_colab_for_full_run(guard_config)
        features = _feature_tensor(train_data)
        _validate_feature_shape(features)
        if features.ndim == 3:
            raise ValueError("train_data must be shared [samples, features], not replica-batched")
        _assert_audited(features)
        features = features.to(self.device)
        if resume_from is not None:
            loaded_state, _ = self.load_checkpoint(
                resume_from,
                map_location=self.device,
                expected_config=expected_config,
            )
            if state is not None or resume_state is not None:
                raise ValueError("resume_from cannot be combined with an explicit state")
            resume_state = loaded_state
        cursor = self._coerce_state(
            resume_state if resume_state is not None else state,
            n_samples=int(features.shape[0]),
        )
        target_steps = int(
            _configuration_value(effective_config, "steps", 4_000)
            if steps is None
            else steps
        )
        if isinstance(target_steps, bool) or target_steps < 1:
            raise ValueError("steps must be a positive integer")
        batch_size = int(_configuration_value(effective_config, "batch_size", 64))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        learning_rate = float(_configuration_value(effective_config, "learning_rate", self.learning_rate))
        if self.optimizer is not None and cursor.global_step == 0:
            self.learning_rate = learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        self._ensure_optimizer(learning_rate)
        kl_coefficient = float(
            _configuration_value(
                effective_config,
                "kl_coefficient",
                _configuration_value(effective_config, "entropy_coefficient", 0.02),
            )
        )
        l2_coefficient = float(_configuration_value(effective_config, "weight_decay", 1.0e-4))
        clip = _configuration_value(effective_config, "grad_clip_norm", 1.0)
        interval = int(
            _configuration_value(
                effective_config,
                "checkpoint_every_steps",
                _configuration_value(effective_config, "checkpoint_every", 500),
            )
        )
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        if cursor.global_step > target_steps:
            raise ValueError("checkpoint global_step is beyond the requested target")
        # A forced final row is useful when a run ends between registered
        # checkpoint boundaries.  If that run is later resumed to a larger
        # target, discard the forced row so the in-memory history matches one
        # uninterrupted run with the same cadence.  The callback's raw log
        # remains the durable record of the forced checkpoint.
        if cursor.global_step < target_steps and cursor.history:
            cursor.history = [
                row
                for row in cursor.history
                if not (
                    int(row.get("step", -1)) == cursor.global_step
                    and bool(row.get("_forced_final", False))
                )
            ]
        if cursor.global_step == target_steps:
            cursor.completed = True
            return cursor
        callback = checkpoint_callback
        if callback is None and checkpoint_dir is not None:
            directory = Path(checkpoint_dir)

            def callback_fn(*, bank: "ReplicaBank", state: BatchedTrainingState, **_: object) -> object:
                return bank.save_checkpoint(
                    directory / f"checkpoint_{state.global_step:08d}.pt",
                    state=state,
                    config=config_for_checkpoint if config_for_checkpoint is not None else effective_config,
                )

            callback = callback_fn
        generators = self._new_sampler_generators()
        if cursor.sampler_states:
            if len(cursor.sampler_states) != self.replica_count:
                raise ValueError("checkpoint sampler state count does not match replica count")
            for generator, generator_state in zip(generators, cursor.sampler_states):
                generator.set_state(torch.as_tensor(generator_state, dtype=torch.uint8, device="cpu"))
        n_samples = int(features.shape[0])
        cursor.completed = False
        while cursor.global_step < target_steps:
            if cursor.permutations is None or cursor.batch_offset >= n_samples:
                if cursor.permutations is not None and cursor.batch_offset >= n_samples:
                    cursor.epoch += 1
                cursor.batch_offset = 0
                cursor.permutations = torch.stack(
                    [torch.randperm(n_samples, generator=generator) for generator in generators],
                    dim=0,
                )
            start = int(cursor.batch_offset)
            stop = min(start + batch_size, n_samples)
            indices = cursor.permutations[:, start:stop].to(self.device)
            minibatch = features[indices]
            metrics = self.step(
                minibatch,
                reward_config,
                kl_coefficient=kl_coefficient,
                l2_coefficient=l2_coefficient,
                grad_clip_norm=clip,
            )
            cursor.batch_offset = stop
            cursor.global_step += 1
            cursor.sampler_states = tuple(generator.get_state().clone() for generator in generators)
            forced_final = cursor.global_step >= target_steps and cursor.global_step % interval != 0
            row: dict[str, object] = {
                name: value.detach().to(device="cpu").tolist()
                for name, value in metrics.items()
            }
            row.update(
                {
                    "step": int(cursor.global_step),
                    "epoch": int(cursor.epoch),
                    "batch_offset": int(cursor.batch_offset),
                    "_forced_final": bool(forced_final),
                }
            )
            cursor.completed = cursor.global_step >= target_steps
            if cursor.global_step % interval == 0 or forced_final:
                # Campaign trajectories are evaluated at committed checkpoints.
                # Retaining one Python row for every replica update would make
                # an 80k-step bank checkpoint hundreds of megabytes larger
                # without adding any registered observation.
                cursor.history.append(row)
                _invoke_callback(callback, bank=self, state=cursor, metrics=metrics, row=row)
        cursor.completed = cursor.global_step >= target_steps
        return cursor

    fit = train

    def state_dict(self) -> dict[str, object]:
        """Return stacked parameters and buffers in module-name order."""

        return {
            "model_state": {
                name: value.detach().clone() for name, value in self._parameters.items()
            },
            "buffer_state": {
                name: value.detach().clone() for name, value in self._buffers.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        """Restore only stacked model state from :meth:`state_dict`."""

        model_state = value.get("model_state") if isinstance(value, Mapping) else None
        buffer_state = value.get("buffer_state") if isinstance(value, Mapping) else None
        if not isinstance(model_state, Mapping) or not isinstance(buffer_state, Mapping):
            raise ValueError("state_dict must contain model_state and buffer_state")
        if set(model_state) != set(self._parameters) or set(buffer_state) != set(self._buffers):
            raise ValueError("state_dict architecture does not match this bank")
        for name, parameter in self._parameters.items():
            tensor = model_state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(f"state_dict parameter {name!r} has the wrong shape")
            parameter.data.copy_(tensor.to(device=self.device, dtype=parameter.dtype))
        for name, buffer in self._buffers.items():
            tensor = buffer_state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(buffer.shape):
                raise ValueError(f"state_dict buffer {name!r} has the wrong shape")
            buffer.copy_(tensor.to(device=self.device, dtype=buffer.dtype))

    def checkpoint_payload(
        self,
        *,
        state: BatchedTrainingState,
        config: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build a complete in-memory interruption/resume payload."""

        _require_torch()
        if not isinstance(state, BatchedTrainingState):
            state = BatchedTrainingState.from_payload(state)
        if state.replica_count != self.replica_count:
            raise ValueError("checkpoint state does not match replica count")
        optimizer = self._ensure_optimizer()
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "engine": "lean_reward_hacking.batched_training",
            "bank_manifest": self.manifest(),
            "model_state": {
                name: value.detach().clone() for name, value in self._parameters.items()
            },
            "buffer_state": {
                name: value.detach().clone() for name, value in self._buffers.items()
            },
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "training_state": state.to_payload(),
            "model_seeds": None if self.model_seeds is None else tuple(self.model_seeds),
            "sampler_seeds": tuple(self.sampler_seeds),
            "config": _config_dict(config),
            "rng_state": _capture_portable_rng_state(),
            "metadata": dict(metadata or {}),
        }

    def load_checkpoint_payload(
        self,
        payload: Mapping[str, object],
        *,
        map_location: str | "torch.device" | None = None,
        expected_config: Mapping[str, object] | None = None,
    ) -> tuple[BatchedTrainingState, dict[str, object]]:
        """Restore model, optimizer, RNG, and cursor from a payload mapping."""

        _require_torch()
        if not isinstance(payload, Mapping):
            raise ValueError("batched checkpoint must be a mapping")
        if int(payload.get("schema_version", -1)) != ENGINE_SCHEMA_VERSION:
            raise ValueError("unsupported batched checkpoint schema")
        stored_config = payload.get("config")
        if expected_config is not None:
            if not isinstance(stored_config, Mapping) or dict(stored_config) != dict(expected_config):
                raise ValueError("checkpoint configuration does not match the requested run")
        elif stored_config is not None and not isinstance(stored_config, Mapping):
            raise ValueError("checkpoint configuration is malformed")
        stored_manifest = payload.get("bank_manifest")
        if stored_manifest is not None:
            if not isinstance(stored_manifest, Mapping):
                raise ValueError("checkpoint bank manifest is malformed")
            # Backend and device are runtime details.  All model and replica
            # identity fields must match exactly before a bank is resumed.
            identity_fields = (
                "schema_version",
                "engine",
                "architecture",
                "replica_count",
                "replicas",
                "parameter_names",
                "parameter_shapes",
                "buffer_names",
                "buffer_shapes",
                "dtype",
                "deterministic_sampler",
            )
            current_manifest = self.manifest()
            for field_name in identity_fields:
                if stored_manifest.get(field_name) != current_manifest.get(field_name):
                    raise ValueError(f"checkpoint bank manifest field {field_name!r} does not match")
        stored_model_seeds = payload.get("model_seeds")
        if self.model_seeds is not None:
            if not isinstance(stored_model_seeds, (tuple, list)):
                raise ValueError("checkpoint model seeds are missing")
            if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in stored_model_seeds):
                raise ValueError("checkpoint model seeds are malformed")
            if tuple(int(seed) for seed in stored_model_seeds) != self.model_seeds:
                raise ValueError("checkpoint model seeds do not match this bank")
        stored_sampler_seeds = payload.get("sampler_seeds")
        if not isinstance(stored_sampler_seeds, (tuple, list)):
            raise ValueError("checkpoint sampler seeds are missing")
        if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in stored_sampler_seeds):
            raise ValueError("checkpoint sampler seeds are malformed")
        if tuple(int(seed) for seed in stored_sampler_seeds) != self.sampler_seeds:
            raise ValueError("checkpoint sampler seeds do not match this bank")
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint is missing model_state")
        if set(model_state) != set(self._parameters):
            raise ValueError("checkpoint parameter architecture does not match this bank")
        target_device = self.device if map_location is None else torch.device(map_location)
        for name, parameter in self._parameters.items():
            value = model_state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"checkpoint parameter {name!r} has the wrong shape")
            parameter.data.copy_(value.to(device=target_device, dtype=parameter.dtype).to(self.device))
        buffer_state = payload.get("buffer_state", {})
        if not isinstance(buffer_state, Mapping) or set(buffer_state) != set(self._buffers):
            raise ValueError("checkpoint buffer architecture does not match this bank")
        for name, buffer in self._buffers.items():
            value = buffer_state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(buffer.shape):
                raise ValueError(f"checkpoint buffer {name!r} has the wrong shape")
            buffer.copy_(value.to(device=self.device, dtype=buffer.dtype))
        optimizer_state = payload.get("optimizer_state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint is missing optimizer_state")
        optimizer = self._ensure_optimizer()
        optimizer.load_state_dict(optimizer_state)
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, Mapping):
            _restore_portable_rng_state(rng_state)
        state = self._coerce_state(BatchedTrainingState.from_payload(payload.get("training_state", {})))
        if state.replica_count != self.replica_count:
            raise ValueError("checkpoint training state has the wrong replica count")
        metadata = payload.get("metadata")
        return state, dict(metadata) if isinstance(metadata, Mapping) else {}

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        state: BatchedTrainingState,
        config: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Atomically save a complete batched checkpoint."""

        _require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload(state=state, config=config, metadata=metadata)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        map_location: str | "torch.device" = "cpu",
        expected_config: Mapping[str, object] | None = None,
    ) -> tuple[BatchedTrainingState, dict[str, object]]:
        """Load and restore a checkpoint file."""

        _require_torch()
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        return self.load_checkpoint_payload(
            payload,
            map_location=self.device,
            expected_config=expected_config,
        )


BatchedReplicaBank = ReplicaBank
BatchedTrainingEngine = ReplicaBank


def initialize_batched_replicas(
    architecture: str | Callable[..., "nn.Module"] = "toy",
    replicas: int = 1,
    *,
    model_seeds: Sequence[int] | int | None = None,
    model_seed: int | None = None,
    sampler_seeds: Sequence[int] | int | None = None,
    sampler_seed: int | None = None,
    model_kwargs: Mapping[str, object] | None = None,
    device: str | "torch.device" = "cpu",
    learning_rate: float | None = None,
) -> ReplicaBank:
    """Construct a batched toy or generic MLP bank by architecture name."""

    _require_torch()
    if isinstance(architecture, str):
        name = architecture.lower().replace("_", "-")
        if name in {"toy", "goal-gate", "goalgate"}:
            from .toy import initialize_toy_agent

            factory: Callable[..., "nn.Module"] = initialize_toy_agent
        elif name in {"generic", "mlp", "plain-mlp", "plain"}:
            from .generic import initialize_generic_agent

            factory = initialize_generic_agent
        else:
            raise ValueError("architecture must be 'toy', 'generic', or a model factory")
    elif callable(architecture):
        factory = architecture
    else:
        raise TypeError("architecture must be a model factory or architecture name")
    return ReplicaBank.from_factory(
        factory,
        replicas,
        model_seeds=model_seeds,
        model_seed=model_seed,
        sampler_seeds=sampler_seeds,
        sampler_seed=sampler_seed,
        factory_kwargs=model_kwargs,
        device=device,
        learning_rate=learning_rate,
    )


initialize_batched_agents = initialize_batched_replicas
make_replica_bank = initialize_batched_replicas


def train_batched_replicas(
    bank: ReplicaBank,
    train_data: object,
    config: object | None = None,
    reward_config: object | None = None,
    **kwargs: object,
) -> BatchedTrainingState:
    """Functional wrapper for notebook integration."""

    if not isinstance(bank, ReplicaBank):
        raise TypeError("bank must be a ReplicaBank")
    return bank.train(train_data, config, reward_config, **kwargs)  # type: ignore[arg-type]


run_batched_training = train_batched_replicas
fit_batched_replicas = train_batched_replicas


def save_batched_checkpoint(
    path: str | os.PathLike[str],
    bank: ReplicaBank,
    state: BatchedTrainingState,
    *,
    config: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Functional checkpoint-save wrapper."""

    return bank.save_checkpoint(path, state=state, config=config, metadata=metadata)


def load_batched_checkpoint(
    path: str | os.PathLike[str],
    bank: ReplicaBank,
    *,
    map_location: str | "torch.device" = "cpu",
    expected_config: Mapping[str, object] | None = None,
) -> tuple[BatchedTrainingState, dict[str, object]]:
    """Functional checkpoint-load wrapper."""

    return bank.load_checkpoint(path, map_location=map_location, expected_config=expected_config)


def checkpoint_payload(
    bank: ReplicaBank,
    state: BatchedTrainingState,
    *,
    config: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return bank.checkpoint_payload(state=state, config=config, metadata=metadata)


def _invoke_callback(
    callback: Callable[..., object] | None,
    *,
    bank: ReplicaBank,
    state: BatchedTrainingState,
    metrics: Mapping[str, "torch.Tensor"],
    row: Mapping[str, object],
) -> object | None:
    if callback is None:
        return None
    values: dict[str, object] = {
        "bank": bank,
        "engine": bank,
        "state": state,
        "training_state": state,
        "metrics": metrics,
        "row": row,
        "step": state.global_step,
    }
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(bank, state, metrics)
    parameters = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return callback(**values)
    named = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if named and all(parameter.name in values for parameter in named):
        return callback(**{parameter.name: values[parameter.name] for parameter in named})
    positional = [
        parameter
        for parameter in named
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(parameter.default is inspect.Parameter.empty for parameter in positional)
    count = max(required, len(positional))
    return callback(*(bank, state, metrics, row)[:count])


__all__ = [
    "BatchedReplicaBank",
    "BatchedTrainingConfig",
    "BatchedTrainingEngine",
    "BatchedTrainingState",
    "ENGINE_SCHEMA_VERSION",
    "ReplicaSpec",
    "ReplicaBank",
    "TORCH_AVAILABLE",
    "TorchUnavailableError",
    "batched_loss_terms",
    "batched_objective_loss",
    "checkpoint_payload",
    "fit_batched_replicas",
    "initialize_batched_agents",
    "initialize_batched_replicas",
    "load_batched_checkpoint",
    "make_replica_bank",
    "replica_loss_terms",
    "run_batched_training",
    "save_batched_checkpoint",
    "train_batched_replicas",
]
