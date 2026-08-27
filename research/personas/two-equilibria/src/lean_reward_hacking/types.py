"""Small, dependency-free data types shared by the experiment modules.

The core package is importable in a clean Python installation.  PyTorch is an
optional runtime dependency used by the Colab training code, so this module
deliberately contains no tensor imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import math
from numbers import Real
from typing import Tuple


class Action(IntEnum):
    """The two actions in the toy environment.

    The integer ordering is part of the file format: model logits and tensor
    rewards use ``HELP=0, HARM=1``.
    """

    HELP = 0
    HARM = 1


class Mode(StrEnum):
    """Pre-registered behavioral labels for modular-agent endpoints."""

    OVERSIGHT_INVARIANT = "oversight-invariant"
    STRATEGIC = "strategic"
    INTERMEDIATE = "intermediate"


def _validate_task(task: tuple[float, ...]) -> None:
    if not task:
        raise ValueError("task must contain at least one feature")
    for value in task:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("task features must be real numbers")
        if not math.isfinite(float(value)):
            raise ValueError("task features must be finite")


@dataclass(frozen=True, slots=True)
class Episode:
    """One toy episode represented without tensors.

    ``task`` is a tuple to keep datasets serializable and hashable.  Tensor
    batches are created only by :func:`lean_reward_hacking.episodes.collate`.
    """

    episode_id: str
    task: Tuple[float, ...]
    audit: bool
    secret_opportunity: bool

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.task, tuple):
            object.__setattr__(self, "task", tuple(self.task))
        _validate_task(self.task)
        object.__setattr__(self, "task", tuple(float(value) for value in self.task))
        if type(self.audit) is not bool:
            raise TypeError("audit must be bool")
        if type(self.secret_opportunity) is not bool:
            raise TypeError("secret_opportunity must be bool")

    def with_audit(self, audit: bool) -> "Episode":
        """Return this episode with only its audit cue changed."""

        return Episode(
            episode_id=self.episode_id,
            task=self.task,
            audit=audit,
            secret_opportunity=self.secret_opportunity,
        )


@dataclass(frozen=True, slots=True)
class PairedEpisode:
    """Two counterfactual episodes that differ in audit status only."""

    on: Episode
    off: Episode

    def __post_init__(self) -> None:
        if self.on.episode_id != self.off.episode_id:
            raise ValueError("paired episodes must have the same episode_id")
        if self.on.task != self.off.task:
            raise ValueError("paired episodes must have identical task features")
        if self.on.secret_opportunity != self.off.secret_opportunity:
            raise ValueError("paired episodes must have identical opportunity fields")
        if self.on.audit is not True or self.off.audit is not False:
            raise ValueError("a pair must be ordered as audit ON then audit OFF")


@dataclass(slots=True)
class EpisodeBatch:
    """A tensor batch, held behind a dependency-free type boundary.

    The attributes are intentionally typed as ``object``.  The concrete type
    is a PyTorch tensor when produced by ``episodes.collate``; keeping the
    annotation dependency-free makes configuration and pairing utilities
    usable without PyTorch.
    """

    task: object
    audit: object
    secret_opportunity: object
    episode_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.episode_ids)

    @staticmethod
    def _column(value: object) -> object:
        """Promote a rank-one tensor-like value to a column lazily."""

        if getattr(value, "ndim", None) == 1:
            return value.unsqueeze(-1)
        return value

    @property
    def all_features(self) -> object:
        """Return ``[task..., audit, opportunity]`` as a tensor.

        This property imports PyTorch only when a model actually requests a
        tensor feature view.
        """

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Colab supplies torch
            raise RuntimeError("PyTorch is required for tensor feature views") from exc
        task = self.task
        audit = self._column(self.audit)
        opportunity = self._column(self.secret_opportunity)
        return torch.cat((task, audit, opportunity), dim=-1)

    @property
    def goal_features(self) -> object:
        """Return ``[task..., opportunity]`` for the modular goal network."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Colab supplies torch
            raise RuntimeError("PyTorch is required for tensor feature views") from exc
        return torch.cat((self.task, self._column(self.secret_opportunity)), dim=-1)

    def to(self, *args: object, **kwargs: object) -> "EpisodeBatch":
        """Move tensor fields to a device while preserving episode IDs."""

        return EpisodeBatch(
            task=self.task.to(*args, **kwargs),
            audit=self.audit.to(*args, **kwargs),
            secret_opportunity=self.secret_opportunity.to(*args, **kwargs),
            episode_ids=self.episode_ids,
        )
