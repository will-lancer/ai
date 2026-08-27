"""Plain MLP architecture control for the toy reward-hacking experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

try:  # PyTorch is installed by the pinned Colab requirements only.
    import torch
    from torch import nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class TorchUnavailableError(RuntimeError):
    """Raised when a tensor/model operation needs optional PyTorch."""


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "PyTorch is required for the generic MLP; install the pinned Colab dependencies"
        )


_ModuleBase = nn.Module if TORCH_AVAILABLE else object


def _field(batch: object, name: str) -> object:
    if isinstance(batch, Mapping):
        try:
            return batch[name]
        except KeyError as exc:
            raise TypeError(f"batch mapping is missing {name!r}") from exc
    if hasattr(batch, name):
        return getattr(batch, name)
    raise TypeError(f"batch must expose a {name!r} field")


def _as_2d(value: object, *, name: str) -> "torch.Tensor":
    _require_torch()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank two")
    return value.float()


def all_features(batch: object) -> "torch.Tensor":
    """Return columns ordered as task, audit, secret opportunity."""

    _require_torch()
    if isinstance(batch, Mapping) and "all_features" in batch:
        return _as_2d(batch["all_features"], name="all_features")
    if hasattr(batch, "all_features"):
        return _as_2d(batch.all_features, name="all_features")
    if isinstance(batch, torch.Tensor):
        features = _as_2d(batch, name="features")
        if features.shape[1] < 3:
            raise ValueError("all-feature tensors need task, audit, and opportunity columns")
        return features
    task = _as_2d(_field(batch, "task"), name="task")
    audit = _as_2d(_field(batch, "audit"), name="audit")
    opportunity = _as_2d(
        _field(batch, "secret_opportunity"), name="secret_opportunity"
    )
    lengths = {len(task), len(audit), len(opportunity)}
    if len(lengths) != 1:
        raise ValueError("all batch fields must have the same length")
    return torch.cat(
        (
            task,
            audit.to(device=task.device, dtype=task.dtype),
            opportunity.to(device=task.device, dtype=task.dtype),
        ),
        dim=-1,
    )


def _initialise_hidden(linear: "nn.Linear") -> None:
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)


class _TanhMLP(_ModuleBase):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> None:
        _require_torch()
        if input_dim < 1 or output_dim < 1 or not hidden_sizes:
            raise ValueError("input and hidden dimensions must be positive")
        if any(int(width) < 1 for width in hidden_sizes):
            raise ValueError("hidden dimensions must be positive")
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_sizes:
            layer = nn.Linear(previous, int(width))
            _initialise_hidden(layer)
            layers.extend((layer, nn.Tanh()))
            previous = int(width)
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(previous, output_dim)
        nn.init.uniform_(self.output.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        return self.output(self.hidden(features))


class PlainMLPAgent(_ModuleBase):
    """A two-hidden-layer MLP with no named goal or gate modules."""

    action_count = 2
    action_names = ("HELP", "HARM")

    def __init__(
        self,
        task_dim: int = 4,
        hidden_width: int = 24,
        harmful_goal_strength: float = 2.0,
        *,
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        _require_torch()
        if isinstance(task_dim, bool) or int(task_dim) < 1:
            raise ValueError("task_dim must be a positive integer")
        widths = tuple(hidden_sizes) if hidden_sizes is not None else (hidden_width, hidden_width)
        super().__init__()
        self.task_dim = int(task_dim)
        self.hidden_sizes = tuple(int(width) for width in widths)
        self.harmful_goal_strength = float(harmful_goal_strength)
        self.net = _TanhMLP(self.task_dim + 2, self.hidden_sizes, output_dim=2)
        with torch.no_grad():
            # Output columns are HELP, HARM, so this is initially harmful.
            self.net.output.bias.copy_(
                torch.tensor([0.0, self.harmful_goal_strength], dtype=self.net.output.bias.dtype)
            )

    def forward(self, batch: object) -> "torch.Tensor":
        return self.net(all_features(batch))

    def logits(self, batch: object) -> "torch.Tensor":
        return self.forward(batch)

    def help_probability(self, batch: object) -> "torch.Tensor":
        return F.softmax(self.forward(batch), dim=-1)[:, 0]

    def help_probabilities(self, batch: object) -> "torch.Tensor":
        """Plural alias used by shared evaluation callers."""

        return self.help_probability(batch)

    def predict_help_probabilities(self, batch: object) -> "torch.Tensor":
        """Prediction alias used by notebook and evaluation callers."""

        return self.help_probability(batch)

    def diagnostics(self, batch: object) -> dict[str, "torch.Tensor | None"]:
        features = all_features(batch)
        on_logits = self.net(features)
        off_features = features.clone()
        off_features[:, self.task_dim] = 0.0
        off_logits = self.net(off_features)
        return {
            "goal": None,
            "gate": on_logits[:, 0] - off_logits[:, 0],
        }


GenericAgent = PlainMLPAgent
MLPPolicy = PlainMLPAgent


def initialize_generic_agent(
    *,
    task_dim: int = 4,
    hidden_width: int = 24,
    harmful_goal_strength: float = 2.0,
    hidden_sizes: Sequence[int] | None = None,
    model_seed: int | None = None,
    device: str | "torch.device" = "cpu",
) -> PlainMLPAgent:
    """Construct the generic control with an isolated parameter seed."""

    _require_torch()
    if model_seed is None:
        agent = PlainMLPAgent(
            task_dim=task_dim,
            hidden_width=hidden_width,
            harmful_goal_strength=harmful_goal_strength,
            hidden_sizes=hidden_sizes,
        )
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(model_seed))
            agent = PlainMLPAgent(
                task_dim=task_dim,
                hidden_width=hidden_width,
                harmful_goal_strength=harmful_goal_strength,
                hidden_sizes=hidden_sizes,
            )
    return agent.to(device)


make_generic_agent = initialize_generic_agent


def ablate_audit_feature(batch: object, *, task_dim: int | None = None) -> object:
    """Return a copy with the audit feature set to zero.

    A tensor is interpreted as the plain-MLP feature layout.  An
    ``EpisodeBatch``-like object is copied field by field when possible.  The
    source object is left untouched, which matters for paired evaluations.
    """

    _require_torch()
    if isinstance(batch, torch.Tensor):
        result = _as_2d(batch, name="features").clone()
        index = int(task_dim) if task_dim is not None else result.shape[1] - 2
        if index < 0 or index >= result.shape[1] - 1:
            raise ValueError("task_dim does not identify an audit column")
        result[:, index] = 0.0
        return result
    if isinstance(batch, Mapping):
        result = dict(batch)
        audit = _as_2d(result["audit"], name="audit").clone()
        audit.zero_()
        result["audit"] = audit[:, 0] if audit.shape[1] == 1 else audit
        return result
    try:
        result = copy.copy(batch)
        audit = _as_2d(getattr(result, "audit"), name="audit").clone()
        audit.zero_()
        setattr(result, "audit", audit[:, 0] if audit.shape[1] == 1 else audit)
        return result
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "batch must be a tensor, mapping, or mutable EpisodeBatch-like object"
        ) from exc


__all__ = [
    "GenericAgent",
    "MLPPolicy",
    "PlainMLPAgent",
    "TORCH_AVAILABLE",
    "TorchUnavailableError",
    "ablate_audit_feature",
    "all_features",
    "initialize_generic_agent",
    "make_generic_agent",
]
