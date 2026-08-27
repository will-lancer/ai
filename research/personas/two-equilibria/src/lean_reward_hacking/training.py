"""Deterministic fixed-objective training for the toy and generic agents.

The only optimized signal is the expected two-action reward on audited
episodes.  A KL term to the uniform two-action policy and an explicit L2 term
make the objective finite and reproducible.  No audit-OFF examples enter the
training loss.

PyTorch remains optional at import time.  Full training is guarded by the
project's Colab marker; local callers can run only the tiny smoke configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import inspect
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable

try:  # Pinned in requirements-colab.txt, intentionally absent from base deps.
    import torch

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local path
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .toy import TorchUnavailableError, batch_audit


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "PyTorch is required for training; install the pinned Colab dependencies"
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Fixed optimizer and schedule fields recorded in every run."""

    steps: int = 4_000
    epochs: int | None = None
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 1.0e-4
    kl_coefficient: float = 0.02
    entropy_coefficient: float | None = None
    grad_clip_norm: float | None = 1.0
    checkpoint_every_steps: int = 500
    checkpoint_every: int | None = None
    model_seed: int = 0
    sampler_seed: int = 0
    seed: int | None = None
    device: str = "cpu"
    execution: str = "local_smoke"
    replicas: int = 1
    allow_local_smoke: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.epochs is not None and (isinstance(self.epochs, bool) or self.epochs < 1):
            raise ValueError("epochs must be positive when supplied")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("kl_coefficient", self.kl_coefficient),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.entropy_coefficient is not None:
            if not math.isfinite(float(self.entropy_coefficient)) or self.entropy_coefficient < 0:
                raise ValueError("entropy_coefficient must be finite and non-negative")
            # Existing TOML files use this historical field name.  It is the
            # coefficient of the registered KL-to-uniform term.
            object.__setattr__(self, "kl_coefficient", float(self.entropy_coefficient))
        clip = self.grad_clip_norm
        if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0.0):
            raise ValueError("grad_clip_norm must be positive when supplied")
        interval = (
            self.checkpoint_every
            if self.checkpoint_every is not None
            else self.checkpoint_every_steps
        )
        if isinstance(interval, bool) or interval < 1:
            raise ValueError("checkpoint interval must be positive")
        object.__setattr__(self, "checkpoint_every_steps", int(interval))
        if self.seed is not None:
            if isinstance(self.seed, bool):
                raise ValueError("seed must be an integer")
            # ``seed`` is a convenient alias for model_seed in smoke configs.
            object.__setattr__(self, "model_seed", int(self.seed))
        if isinstance(self.model_seed, bool) or isinstance(self.sampler_seed, bool):
            raise ValueError("seeds must be integers")
        if self.replicas < 1:
            raise ValueError("replicas must be positive")

    @property
    def total_steps(self) -> int:
        return int(self.steps)


TrainConfig = TrainingConfig


@dataclass(slots=True)
class TrainingState:
    """Serializable cursor for exact epoch and minibatch continuation."""

    epoch: int = 0
    batch_offset: int = 0
    global_step: int = 0
    permutation: object | None = None
    sampler_state: object | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    completed: bool = False


@dataclass(slots=True)
class TrainingResult:
    """Small return object retained for callers that want state and metrics."""

    state: TrainingState
    agent: object
    optimizer: object

    @property
    def history(self) -> list[dict[str, float]]:
        return self.state.history


def _configuration_value(config: object, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _coefficient(config: object, name: str, fallback: float) -> float:
    value = _configuration_value(config, name, fallback)
    return float(value)


def _reward_values(reward_config: object | None) -> tuple[float, float]:
    if reward_config is None:
        return 1.0, -1.0
    return (
        float(getattr(reward_config, "help_reward", 1.0)),
        float(getattr(reward_config, "harm_reward", -1.0)),
    )


def expected_action_reward(
    logits: "torch.Tensor", *, help_reward: float = 1.0, harm_reward: float = -1.0
) -> "torch.Tensor":
    """Return the exact expected reward for each two-action logit row."""

    _require_torch()
    _validate_logits(logits)
    rewards = torch.as_tensor(
        [float(help_reward), float(harm_reward)], dtype=logits.dtype, device=logits.device
    )
    return (torch.softmax(logits, dim=-1) * rewards).sum(dim=-1)


expected_two_action_reward = expected_action_reward


def _validate_logits(logits: object) -> None:
    _require_torch()
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a PyTorch tensor")
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape [batch, 2] ordered HELP, HARM")
    if logits.shape[0] < 1:
        raise ValueError("logits batch cannot be empty")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")


def _l2_penalty(
    parameters: Iterable["torch.Tensor"] | None, reference: "torch.Tensor"
) -> "torch.Tensor":
    if parameters is None:
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    terms = [parameter.square().sum() for parameter in parameters if parameter.requires_grad]
    if not terms:
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return torch.stack(terms).sum()


def loss_terms(
    logits: "torch.Tensor",
    *,
    help_reward: float = 1.0,
    harm_reward: float = -1.0,
    kl_coefficient: float = 0.02,
    l2_coefficient: float = 1.0e-4,
    parameters: Iterable["torch.Tensor"] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Return the fixed reward, KL, L2, and total-loss terms."""

    _require_torch()
    _validate_logits(logits)
    probs = torch.softmax(logits, dim=-1)
    expected_reward = (probs * torch.as_tensor(
        [float(help_reward), float(harm_reward)], dtype=logits.dtype, device=logits.device
    )).sum(dim=-1).mean()
    log_probs = torch.log_softmax(logits, dim=-1)
    uniform_log_prob = math.log(0.5)
    kl = (probs * (log_probs - uniform_log_prob)).sum(dim=-1).mean()
    l2 = _l2_penalty(parameters, logits)
    reward_loss = -expected_reward
    total = reward_loss + float(kl_coefficient) * kl + 0.5 * float(l2_coefficient) * l2
    return {
        "loss": total,
        "reward_loss": reward_loss,
        "expected_reward": expected_reward,
        "kl": kl,
        "l2": l2,
        "help_probability": probs[:, 0].mean(),
    }


def expected_two_action_loss(
    logits: "torch.Tensor",
    *,
    help_reward: float = 1.0,
    harm_reward: float = -1.0,
    kl_coefficient: float = 0.02,
    l2_coefficient: float = 1.0e-4,
    parameters: Iterable["torch.Tensor"] | None = None,
) -> "torch.Tensor":
    """Compute the exact expected reward plus uniform-KL and L2 loss."""

    return loss_terms(
        logits,
        help_reward=help_reward,
        harm_reward=harm_reward,
        kl_coefficient=kl_coefficient,
        l2_coefficient=l2_coefficient,
        parameters=parameters,
    )["loss"]


objective_loss = expected_two_action_loss
fixed_objective_loss = expected_two_action_loss


def _batch_length(batch: object) -> int:
    if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            raise ValueError("batch tensor must have a leading batch dimension")
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for key in ("task", "features", "all_features", "audit"):
            if key in batch:
                value = batch[key]
                return len(value)
    for name in ("task", "audit", "episode_ids"):
        if hasattr(batch, name):
            return len(getattr(batch, name))
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return len(batch)
    raise TypeError("training data must be a tensor, mapping, EpisodeBatch, or sequence")


def _index_value(value: object, indices: "torch.Tensor", n: int) -> object:
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, tuple):
        return tuple(value[int(index)] for index in indices.tolist())
    if isinstance(value, list):
        return [value[int(index)] for index in indices.tolist()]
    if hasattr(value, "__len__") and len(value) == n:
        return [value[int(index)] for index in indices.tolist()]
    return value


def _subset_batch(batch: object, indices: "torch.Tensor", n: int) -> object:
    if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
        return batch.index_select(0, indices.to(batch.device))
    if isinstance(batch, Mapping):
        result = dict(batch)
        for key, value in result.items():
            result[key] = _index_value(value, indices, n)
        return result
    # EpisodeBatch is a mutable dataclass, yet reconstructing its concrete type
    # keeps source metadata and episode IDs aligned with tensor rows.
    if all(hasattr(batch, name) for name in ("task", "audit", "secret_opportunity")):
        values = {
            "task": _index_value(getattr(batch, "task"), indices, n),
            "audit": _index_value(getattr(batch, "audit"), indices, n),
            "secret_opportunity": _index_value(
                getattr(batch, "secret_opportunity"), indices, n
            ),
        }
        if hasattr(batch, "episode_ids"):
            values["episode_ids"] = _index_value(getattr(batch, "episode_ids"), indices, n)
        try:
            return type(batch)(**values)
        except TypeError:
            result = type(batch).__new__(type(batch))
            for name, value in values.items():
                setattr(result, name, value)
            return result
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return [batch[int(index)] for index in indices.tolist()]
    raise TypeError("cannot index training batch")


def _move_batch(batch: object, device: "torch.device | str") -> object:
    if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        result = dict(batch)
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
        return result
    if all(hasattr(batch, name) for name in ("task", "audit", "secret_opportunity")):
        values = {
            name: getattr(batch, name).to(device)
            if isinstance(getattr(batch, name), torch.Tensor)
            else getattr(batch, name)
            for name in ("task", "audit", "secret_opportunity")
        }
        if hasattr(batch, "episode_ids"):
            values["episode_ids"] = getattr(batch, "episode_ids")
        try:
            return type(batch)(**values)
        except TypeError:
            result = type(batch).__new__(type(batch))
            for name, value in values.items():
                setattr(result, name, value)
            return result
    return batch


def _coerce_batch(data: object) -> object:
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return data
    if isinstance(data, Mapping):
        return data
    if all(hasattr(data, name) for name in ("task", "audit", "secret_opportunity")):
        return data
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        try:
            from .episodes import collate
        except ImportError as exc:
            raise RuntimeError("episode collation is unavailable") from exc
        return collate(data)
    raise TypeError("training data must expose task, audit, and opportunity fields")


def _assert_audited(batch: object) -> None:
    audit = batch_audit(batch)
    if not bool(torch.all(audit == 1).item()):
        raise ValueError("fixed RLHF training requires audit=ON for every episode")


def train_step(
    agent: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    batch: object,
    reward_config: object | None = None,
    *,
    kl_coefficient: float = 0.02,
    l2_coefficient: float = 1.0e-4,
    entropy_coefficient: float | None = None,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Apply one deterministic update on an audited minibatch."""

    _require_torch()
    _assert_audited(batch)
    if entropy_coefficient is not None:
        kl_coefficient = float(entropy_coefficient)
    help_reward, harm_reward = _reward_values(reward_config)
    optimizer.zero_grad(set_to_none=True)
    logits = agent(batch)
    terms = loss_terms(
        logits,
        help_reward=help_reward,
        harm_reward=harm_reward,
        kl_coefficient=kl_coefficient,
        l2_coefficient=l2_coefficient,
        parameters=agent.parameters(),
    )
    terms["loss"].backward()
    grad_norm = torch.zeros((), device=logits.device)
    if grad_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), float(grad_clip_norm))
    optimizer.step()
    return {
        "loss": float(terms["loss"].detach().cpu().item()),
        "reward_loss": float(terms["reward_loss"].detach().cpu().item()),
        "expected_reward": float(terms["expected_reward"].detach().cpu().item()),
        "kl": float(terms["kl"].detach().cpu().item()),
        "l2": float(terms["l2"].detach().cpu().item()),
        "help_probability": float(terms["help_probability"].detach().cpu().item()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
    }


def epoch_permutations(
    n_samples: int, epochs: int, sampler_seed: int
) -> tuple["torch.Tensor", ...]:
    """Return deterministic CPU permutations generated solely by sampler seed."""

    _require_torch()
    if isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be positive")
    if isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sampler_seed))
    return tuple(torch.randperm(n_samples, generator=generator) for _ in range(epochs))


make_epoch_permutations = epoch_permutations


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy when installed, and PyTorch CPU/CUDA streams."""

    _require_torch()
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - Colab GPU path
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, object]:
    _require_torch()
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():  # pragma: no cover - Colab GPU path
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    _require_torch()
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]
    try:
        import numpy as np

        if "numpy" in state:
            np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    except ImportError:
        pass
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if "torch_cuda" in state and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def _config_dict(config: object) -> dict[str, object]:
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return {
        field_info.name: getattr(config, field_info.name)
        for field_info in fields(config)
        if hasattr(config, field_info.name)
    } if hasattr(config, "__dataclass_fields__") else {"repr": repr(config)}


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    agent: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    state: TrainingState,
    config: object,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save model, optimizer, cursor, sampler, and RNG state."""

    _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_state": agent.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_state": state,
        "config": _config_dict(config),
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata or {}),
    }
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


save_training_checkpoint = save_checkpoint


def _coerce_training_state(value: object) -> TrainingState:
    if isinstance(value, TrainingState):
        return value
    if isinstance(value, Mapping):
        allowed = {field_info.name for field_info in fields(TrainingState)}
        values = {key: value[key] for key in allowed if key in value}
        return TrainingState(**values)
    raise TypeError("checkpoint training_state has an unsupported type")


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    agent: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer | None" = None,
    map_location: str | "torch.device" = "cpu",
    expected_config: Mapping[str, object] | None = None,
) -> tuple[TrainingState, dict[str, object]]:
    """Load a checkpoint and restore all stochastic streams before resuming."""

    _require_torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or "model_state" not in payload
        or "training_state" not in payload
    ):
        raise ValueError("checkpoint is missing model_state or training_state")
    if expected_config is not None:
        stored = payload.get("config")
        if not isinstance(stored, Mapping) or dict(stored) != dict(expected_config):
            raise ValueError("checkpoint configuration does not match the requested run")
    agent.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("checkpoint is missing optimizer_state")
        optimizer.load_state_dict(payload["optimizer_state"])  # type: ignore[arg-type]
    state = _coerce_training_state(payload["training_state"])
    rng_state = payload.get("rng_state")
    if isinstance(rng_state, Mapping):
        restore_rng_state(rng_state)
    metadata = payload.get("metadata")
    return state, dict(metadata) if isinstance(metadata, Mapping) else {}


load_training_checkpoint = load_checkpoint


def _invoke_callback(
    callback: Callable[..., object] | None,
    *,
    kind: str,
    agent: object,
    optimizer: object,
    state: TrainingState,
    metrics: Mapping[str, float],
) -> object | None:
    if callback is None:
        return None
    values = {
        "agent": agent,
        "model": agent,
        "optimizer": optimizer,
        "state": state,
        "training_state": state,
        "metrics": metrics,
        "step": state.global_step,
    }
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(state, agent, optimizer, metrics)
    params = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in params):
        return callback(**values)
    named = [
        parameter
        for parameter in params
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if named and all(parameter.name in values for parameter in named):
        return callback(**{parameter.name: values[parameter.name] for parameter in named})
    if kind == "eval":
        ordered = (agent, state, metrics)
    else:
        ordered = (state, agent, optimizer, metrics)
    positional = [
        parameter
        for parameter in named
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(parameter.default is inspect.Parameter.empty for parameter in positional)
    count = max(required, len(positional))
    return callback(*ordered[:count])


def _is_local_smoke(config: object) -> bool:
    execution = str(_configuration_value(config, "execution", "local_smoke")).lower()
    steps = int(_configuration_value(config, "steps", 4_000))
    replicas = int(_configuration_value(config, "replicas", 1))
    allow = bool(_configuration_value(config, "allow_local_smoke", True))
    return allow and execution in {"smoke", "local_smoke"} and steps <= 8 and replicas <= 1


def require_colab_for_full_run(config: object, *, env: Mapping[str, str] | None = None) -> None:
    """Reject full training locally, even when a CUDA device is unavailable."""

    if _is_local_smoke(config):
        device = str(_configuration_value(config, "device", "cpu")).lower()
        if device not in {"cpu", ""}:
            raise RuntimeError("local smoke runs must use CPU")
        return
    try:
        from .safety import assert_colab_execution

        assert_colab_execution(
            require_gpu=str(_configuration_value(config, "device", "cpu"))
            .lower()
            .startswith("cuda"),
            env=env,
        )
    except ImportError as exc:  # pragma: no cover - package-integrated path
        raise RuntimeError("full training requires the Colab safety guard") from exc


def _device_for(agent: "torch.nn.Module", config: object) -> "torch.device":
    requested = str(_configuration_value(config, "device", "cpu"))
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device {requested!r}") from exc
    if device.type != "cpu" and not _is_local_smoke(config):
        require_colab_for_full_run(config)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _new_state_from_resume(
    resume_state: TrainingState | None, sampler_generator: "torch.Generator"
) -> TrainingState:
    state = resume_state or TrainingState()
    if state.sampler_state is not None:
        sampler_generator.set_state(state.sampler_state)  # type: ignore[arg-type]
    return state


def _default_checkpoint_callback(output_dir: Path, config: object) -> Callable[..., object]:
    def save(
        *,
        agent: object,
        optimizer: object,
        state: TrainingState,
        metrics: Mapping[str, float],
        **_: object,
    ) -> Path:
        path = output_dir / f"checkpoint_{state.global_step:08d}.pt"
        return save_checkpoint(
            path,
            agent=agent,  # type: ignore[arg-type]
            optimizer=optimizer,  # type: ignore[arg-type]
            state=state,
            config=config,
            metadata={"metrics": dict(metrics)},
        )

    return save


def run_training(
    agent: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer | None",
    train_data: object,
    config: object,
    reward_config: object | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    *,
    eval_pairs: object | None = None,
    checkpoint_callback: Callable[..., object] | None = None,
    eval_callback: Callable[..., object] | None = None,
    resume_state: TrainingState | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> TrainingState:
    """Run a fixed audited training process with explicit resumable cursors."""

    _require_torch()
    require_colab_for_full_run(config)
    del eval_pairs  # Evaluation is supplied by the explicit callback.
    batch = _coerce_batch(train_data)
    n_samples = _batch_length(batch)
    if n_samples < 1:
        raise ValueError("training data cannot be empty")
    _assert_audited(batch)
    device = _device_for(agent, config)
    agent.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=float(_configuration_value(config, "learning_rate", 0.003))
        )
    if resume_from is not None:
        resume_state, _metadata = load_checkpoint(
            resume_from,
            agent=agent,
            optimizer=optimizer,
            map_location=device,
        )
    seed = int(_configuration_value(config, "sampler_seed", 0))
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(seed)
    state = _new_state_from_resume(resume_state, sampler_generator)
    target_steps = int(_configuration_value(config, "steps", 4_000))
    batch_size = int(_configuration_value(config, "batch_size", 64))
    kl_coefficient = float(
        _configuration_value(
            config,
            "kl_coefficient",
            _configuration_value(config, "entropy_coefficient", 0.02),
        )
    )
    l2_coefficient = float(_configuration_value(config, "weight_decay", 1.0e-4))
    clip = _configuration_value(config, "grad_clip_norm", 1.0)
    checkpoint_interval = int(
        _configuration_value(
            config,
            "checkpoint_every_steps",
            _configuration_value(config, "checkpoint_every", 500),
        )
    )
    checkpoint_callback = checkpoint_callback or (
        _default_checkpoint_callback(Path(checkpoint_dir), config)
        if checkpoint_dir is not None
        else None
    )

    while state.global_step < target_steps:
        if state.permutation is None or state.batch_offset >= n_samples:
            if state.permutation is not None and state.batch_offset >= n_samples:
                state.epoch += 1
            state.batch_offset = 0
            state.permutation = torch.randperm(n_samples, generator=sampler_generator)
        permutation = state.permutation
        start = int(state.batch_offset)
        stop = min(start + batch_size, n_samples)
        indices = permutation[start:stop]
        minibatch = _move_batch(_subset_batch(batch, indices, n_samples), device)
        metrics = train_step(
            agent,
            optimizer,
            minibatch,
            reward_config,
            kl_coefficient=kl_coefficient,
            l2_coefficient=l2_coefficient,
            grad_clip_norm=clip,
        )
        state.batch_offset = stop
        state.global_step += 1
        state.sampler_state = sampler_generator.get_state().clone()
        row = {
            **metrics,
            "step": float(state.global_step),
            "epoch": float(state.epoch),
            "batch_offset": float(state.batch_offset),
        }
        state.history.append(row)
        if state.global_step % checkpoint_interval == 0 or state.global_step >= target_steps:
            _invoke_callback(
                checkpoint_callback,
                kind="checkpoint",
                agent=agent,
                optimizer=optimizer,
                state=state,
                metrics=row,
            )
            _invoke_callback(
                eval_callback,
                kind="eval",
                agent=agent,
                optimizer=optimizer,
                state=state,
                metrics=row,
            )
    state.completed = True
    return state


def fit_replicate(
    agent: "torch.nn.Module",
    train_data: object,
    config: object | None = None,
    *,
    reward_config: object | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    eval_pairs: object | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    checkpoint_callback: Callable[..., object] | None = None,
    eval_callback: Callable[..., object] | None = None,
    resume_state: TrainingState | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> TrainingState:
    """Convenience wrapper used by Colab notebooks and the CLI."""

    _require_torch()
    effective_config = config or TrainingConfig()
    directory = checkpoint_dir if checkpoint_dir is not None else output_dir
    return run_training(
        agent,
        optimizer,
        train_data,
        effective_config,
        reward_config,
        directory,
        eval_pairs=eval_pairs,
        checkpoint_callback=checkpoint_callback,
        eval_callback=eval_callback,
        resume_state=resume_state,
        resume_from=resume_from,
    )


run_replicate = fit_replicate


def resume_replicate(
    checkpoint_path: str | os.PathLike[str],
    agent: "torch.nn.Module",
    train_data: object,
    config: object,
    *,
    reward_config: object | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    eval_pairs: object | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    checkpoint_callback: Callable[..., object] | None = None,
    eval_callback: Callable[..., object] | None = None,
) -> TrainingState:
    """Resume from a saved cursor while preserving optimizer and RNG state."""

    return fit_replicate(
        agent,
        train_data,
        config,
        reward_config=reward_config,
        optimizer=optimizer,
        eval_pairs=eval_pairs,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        checkpoint_callback=checkpoint_callback,
        eval_callback=eval_callback,
        resume_from=checkpoint_path,
    )


__all__ = [
    "TORCH_AVAILABLE",
    "TrainConfig",
    "TrainingConfig",
    "TrainingResult",
    "TrainingState",
    "capture_rng_state",
    "epoch_permutations",
    "expected_action_reward",
    "expected_two_action_loss",
    "expected_two_action_reward",
    "fit_replicate",
    "fixed_objective_loss",
    "load_checkpoint",
    "load_training_checkpoint",
    "loss_terms",
    "make_epoch_permutations",
    "objective_loss",
    "require_colab_for_full_run",
    "restore_rng_state",
    "resume_replicate",
    "run_replicate",
    "run_training",
    "save_checkpoint",
    "save_training_checkpoint",
    "seed_everything",
    "train_step",
]
