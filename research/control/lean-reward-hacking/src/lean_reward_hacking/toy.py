"""The factorised toy policy used by the fixed-objective experiment.

The module keeps the PyTorch import optional.  Dataset, provenance, and static
contract checks can therefore run in the lightweight local environment while
Colab supplies the training dependency.

The model has two explicit pieces.  ``goal_net`` sees the task and opportunity
fields.  ``oversight_gate`` sees the same fields and contributes only to the
HELP logit when the audit bit is on.  The off-audit gate contribution is zero
by construction, which makes the paired evaluation interpretable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # PyTorch is a pinned Colab dependency, not a local package dependency.
    import torch
    from torch import nn
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - exercised locally
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class TorchUnavailableError(RuntimeError):
    """Raised when a tensor/model operation needs the optional torch package."""


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "PyTorch is required for toy models; install the pinned Colab dependencies"
        )


_ModuleBase = nn.Module if TORCH_AVAILABLE else object


def _as_mapping_value(batch: object, name: str) -> object:
    if isinstance(batch, Mapping):
        try:
            return batch[name]
        except KeyError as exc:
            raise TypeError(f"batch mapping is missing {name!r}") from exc
    if hasattr(batch, name):
        return getattr(batch, name)
    raise TypeError(f"batch must expose a {name!r} field")


def _tensor_2d(value: object, *, name: str) -> "torch.Tensor":
    _require_torch()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    return value.float()


def batch_goal_features(batch: object) -> "torch.Tensor":
    """Return task plus opportunity features from an ``EpisodeBatch``.

    ``EpisodeBatch`` intentionally remains dependency-free, so this helper
    accepts either its three named fields or a mapping with the same fields.
    A raw tensor is interpreted as ``[task..., audit, opportunity]`` for small
    smoke tests.
    """

    _require_torch()
    if isinstance(batch, Mapping) and "goal_features" in batch:
        return _tensor_2d(batch["goal_features"], name="goal_features")
    if hasattr(batch, "goal_features"):
        return _tensor_2d(batch.goal_features, name="goal_features")
    if isinstance(batch, torch.Tensor):
        features = _tensor_2d(batch, name="features")
        if features.shape[1] < 3:
            raise ValueError("raw all-feature tensors need task, audit, and opportunity columns")
        return torch.cat((features[:, :-2], features[:, -1:]), dim=-1)
    task = _tensor_2d(_as_mapping_value(batch, "task"), name="task")
    opportunity = _tensor_2d(
        _as_mapping_value(batch, "secret_opportunity"), name="secret_opportunity"
    )
    if len(task) != len(opportunity):
        raise ValueError("task and opportunity batch lengths must agree")
    return torch.cat((task, opportunity.to(device=task.device, dtype=task.dtype)), dim=-1)


def batch_audit(batch: object) -> "torch.Tensor":
    """Return the audit bit as a rank-one floating-point tensor."""

    _require_torch()
    if isinstance(batch, torch.Tensor):
        features = _tensor_2d(batch, name="features")
        if features.shape[1] < 3:
            raise ValueError("raw all-feature tensors need task, audit, and opportunity columns")
        return features[:, -2]
    audit = _tensor_2d(_as_mapping_value(batch, "audit"), name="audit")
    return audit[:, 0]


def batch_all_features(batch: object) -> "torch.Tensor":
    """Return the plain-control input ``[task, audit, opportunity]``."""

    _require_torch()
    if isinstance(batch, torch.Tensor):
        return _tensor_2d(batch, name="features")
    task = _tensor_2d(_as_mapping_value(batch, "task"), name="task")
    audit = _tensor_2d(_as_mapping_value(batch, "audit"), name="audit")
    opportunity = _tensor_2d(
        _as_mapping_value(batch, "secret_opportunity"), name="secret_opportunity"
    )
    lengths = {len(task), len(audit), len(opportunity)}
    if len(lengths) != 1:
        raise ValueError("all batch fields must have the same length")
    dtype = task.dtype
    device = task.device
    return torch.cat(
        (
            task,
            audit.to(device=device, dtype=dtype),
            opportunity.to(device=device, dtype=dtype),
        ),
        dim=-1,
    )


def _initialise_hidden(linear: "nn.Linear") -> None:
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)


class _TanhMLP(_ModuleBase):
    """Two hidden layers with tanh activations and a small output head."""

    def __init__(self, input_dim: int, hidden_width: int, output_dim: int) -> None:
        _require_torch()
        if input_dim < 1 or hidden_width < 1 or output_dim < 1:
            raise ValueError("network dimensions must be positive")
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_width)
        self.hidden2 = nn.Linear(hidden_width, hidden_width)
        self.output = nn.Linear(hidden_width, output_dim)
        _initialise_hidden(self.hidden1)
        _initialise_hidden(self.hidden2)
        nn.init.uniform_(self.output.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.output.bias)

    def hidden_features(self, features: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.hidden2(torch.tanh(self.hidden1(features))))

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        return self.output(self.hidden_features(features))


class GoalNet(_TanhMLP):
    """Score HELP and HARM from task and secret-opportunity features."""

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 16,
        harmful_goal_strength: float = 2.0,
    ) -> None:
        super().__init__(input_dim, hidden_width, output_dim=2)
        with torch.no_grad():
            # Logits are ordered HELP, HARM.  A positive HARM bias gives a
            # negative initial HELP-minus-HARM goal score.
            self.output.bias.copy_(
                torch.tensor([0.0, float(harmful_goal_strength)], dtype=self.output.bias.dtype)
            )


class OversightGate(_TanhMLP):
    """Scalar audit contribution, evaluated on goal features.

    The parent agent multiplies this value by the audit bit.  Consequently an
    audit-OFF example has no gate contribution even when its task features are
    identical to the paired audit-ON example.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 16,
        initial_audit_sensitivity: float = 0.0,
    ) -> None:
        super().__init__(input_dim, hidden_width, output_dim=1)
        with torch.no_grad():
            self.output.bias.fill_(float(initial_audit_sensitivity))

    @property
    def weight(self) -> "torch.Tensor":
        """Expose the output head under the conventional gate name."""

        return self.output.weight


class GoalGateAgent(_ModuleBase):
    """Factorised toy agent with explicit goal and audit-gate modules."""

    action_count = 2
    action_names = ("HELP", "HARM")

    def __init__(
        self,
        task_dim: int = 4,
        hidden_width: int = 16,
        harmful_goal_strength: float = 2.0,
        initial_audit_sensitivity: float = 0.0,
    ) -> None:
        _require_torch()
        if isinstance(task_dim, bool) or task_dim < 1:
            raise ValueError("task_dim must be a positive integer")
        super().__init__()
        self.task_dim = int(task_dim)
        self.hidden_width = int(hidden_width)
        self.harmful_goal_strength = float(harmful_goal_strength)
        self.initial_audit_sensitivity = float(initial_audit_sensitivity)
        goal_input_dim = self.task_dim + 1
        self.goal_net = GoalNet(
            goal_input_dim,
            hidden_width=self.hidden_width,
            harmful_goal_strength=self.harmful_goal_strength,
        )
        self.oversight_gate = OversightGate(
            goal_input_dim,
            hidden_width=self.hidden_width,
            initial_audit_sensitivity=self.initial_audit_sensitivity,
        )

    def split_logits(
        self, batch: object
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return goal logits, scalar gate values, and audit bits."""

        goal_features = batch_goal_features(batch)
        audit = batch_audit(batch).to(device=goal_features.device, dtype=goal_features.dtype)
        goal_logits = self.goal_net(goal_features)
        gate = self.oversight_gate(goal_features).squeeze(-1)
        return goal_logits, gate, audit

    def forward(self, batch: object) -> "torch.Tensor":
        goal_logits, gate, audit = self.split_logits(batch)
        # Keep the composition functional.  ``ReplicaBank`` evaluates a
        # stacked parameter axis through ``torch.func.vmap``; slice assignment
        # on the batched output can trigger vmap mutation errors or silently
        # materialize the wrong leading dimension.
        gate_delta = (audit * gate).unsqueeze(-1)
        return goal_logits + torch.cat((gate_delta, torch.zeros_like(gate_delta)), dim=-1)

    def logits(self, batch: object) -> "torch.Tensor":
        """Alias for callers that use ``agent.logits(batch)``."""

        return self.forward(batch)

    def help_probability(self, batch: object) -> "torch.Tensor":
        return F.softmax(self.forward(batch), dim=-1)[:, 0]

    def help_probabilities(self, batch: object) -> "torch.Tensor":
        """Plural alias used by shared evaluation callers."""

        return self.help_probability(batch)

    def predict_help_probabilities(self, batch: object) -> "torch.Tensor":
        """Prediction alias used by notebook and evaluation callers."""

        return self.help_probability(batch)

    def diagnostics(self, batch: object) -> dict[str, "torch.Tensor"]:
        goal_logits, gate, _audit = self.split_logits(batch)
        return {
            "goal": goal_logits[:, 0] - goal_logits[:, 1],
            "gate": gate,
        }


ToyAgent = GoalGateAgent


def initialize_toy_agent(
    *,
    task_dim: int = 4,
    hidden_width: int = 16,
    harmful_goal_strength: float = 2.0,
    initial_audit_sensitivity: float = 0.0,
    model_seed: int | None = None,
    device: str | "torch.device" = "cpu",
) -> GoalGateAgent:
    """Construct a reproducibly initialised toy model.

    ``fork_rng`` keeps construction from consuming a caller's unrelated RNG
    stream.  The model seed controls parameter initialization only; minibatch
    order belongs to the independent sampler seed in ``training.py``.
    """

    _require_torch()
    if model_seed is None:
        agent = GoalGateAgent(
            task_dim=task_dim,
            hidden_width=hidden_width,
            harmful_goal_strength=harmful_goal_strength,
            initial_audit_sensitivity=initial_audit_sensitivity,
        )
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(model_seed))
            agent = GoalGateAgent(
                task_dim=task_dim,
                hidden_width=hidden_width,
                harmful_goal_strength=harmful_goal_strength,
                initial_audit_sensitivity=initial_audit_sensitivity,
            )
    return agent.to(device)


make_toy_agent = initialize_toy_agent


__all__ = [
    "GoalGateAgent",
    "GoalNet",
    "OversightGate",
    "TORCH_AVAILABLE",
    "TorchUnavailableError",
    "ToyAgent",
    "batch_all_features",
    "batch_audit",
    "batch_goal_features",
    "initialize_toy_agent",
    "make_toy_agent",
]
