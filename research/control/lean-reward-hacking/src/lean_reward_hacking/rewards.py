"""The fixed action-only objective used by every toy training run."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Sequence

from .types import Action


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Immutable reward values recorded in every run manifest."""

    help_reward: float = 1.0
    harm_reward: float = -1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("help_reward", self.help_reward),
            ("harm_reward", self.harm_reward),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


def _action_value(action: Action | int) -> Action:
    if isinstance(action, bool) or not isinstance(action, (Action, int)):
        raise ValueError("actions must be HELP or HARM, not bool")
    try:
        return Action(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown action {action!r}; expected HELP=0 or HARM=1") from exc


def reward_for_action(
    action: Action | int, config: RewardConfig | None = None
) -> float:
    """Return the scalar fixed reward for one action."""

    config = config or RewardConfig()
    return (
        float(config.help_reward)
        if _action_value(action) is Action.HELP
        else float(config.harm_reward)
    )


def _torch_reward(actions: object, config: RewardConfig) -> object:
    import torch

    if not isinstance(actions, torch.Tensor):
        raise TypeError
    if actions.is_floating_point():
        valid = (actions == int(Action.HELP)) | (actions == int(Action.HARM))
    else:
        valid = (actions == int(Action.HELP)) | (actions == int(Action.HARM))
    if not bool(torch.all(valid).item()):
        raise ValueError("tensor actions must contain only HELP=0 or HARM=1")
    help_values = torch.full(
        actions.shape,
        float(config.help_reward),
        dtype=torch.float32,
        device=actions.device,
    )
    harm_values = torch.full(
        actions.shape,
        float(config.harm_reward),
        dtype=torch.float32,
        device=actions.device,
    )
    return torch.where(actions == int(Action.HELP), help_values, harm_values)


def fixed_rlhf_reward(
    actions: Action | int | Sequence[Action | int] | object,
    config: RewardConfig | None = None,
    *,
    episodes: object | None = None,
) -> float | tuple[float, ...] | object:
    """Evaluate the registered reward for scalar, sequence, or tensor actions.

    ``episodes`` is accepted solely as an audit-trail convenience and is
    intentionally ignored.  This explicit argument makes it harder for a
    caller to accidentally introduce an audit-conditioned reward while
    preserving a convenient training-call shape.
    """

    del episodes
    config = config or RewardConfig()

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(actions, torch.Tensor):
        return _torch_reward(actions, config)
    if isinstance(actions, (str, bytes)):
        raise TypeError("actions must be Action/int values, not text")
    if isinstance(actions, Iterable) and not isinstance(actions, (Action, int)):
        return tuple(reward_for_action(action, config) for action in actions)
    return reward_for_action(actions, config)  # type: ignore[arg-type]


def reward_for_actions(
    actions: Action | int | Sequence[Action | int] | object,
    config: RewardConfig | None = None,
    *,
    episodes: object | None = None,
) -> float | tuple[float, ...] | object:
    """Alias used by training code and notebooks."""

    return fixed_rlhf_reward(actions, config, episodes=episodes)


__all__ = [
    "RewardConfig",
    "fixed_rlhf_reward",
    "reward_for_action",
    "reward_for_actions",
]
