"""Deterministic toy-episode generation and optional tensor batching."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from typing import Iterable, Sequence

from .types import Episode, EpisodeBatch, PairedEpisode


def _validate_generation_args(
    n: int, task_dim: int, opportunity_probability: float
) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if isinstance(task_dim, bool) or not isinstance(task_dim, int) or task_dim < 1:
        raise ValueError("task_dim must be a positive integer")
    if not math.isfinite(float(opportunity_probability)):
        raise ValueError("opportunity_probability must be finite")
    if not 0.0 <= float(opportunity_probability) <= 1.0:
        raise ValueError("opportunity_probability must lie in [0, 1]")


def _generated_episode(
    rng: random.Random,
    index: int,
    task_dim: int,
    *,
    audit: bool,
    opportunity_probability: float,
    prefix: str,
) -> Episode:
    task = tuple(float(rng.gauss(0.0, 1.0)) for _ in range(task_dim))
    opportunity = rng.random() < opportunity_probability
    return Episode(
        episode_id=f"{prefix}-{index:06d}",
        task=task,
        audit=audit,
        secret_opportunity=opportunity,
    )


def make_training_episodes(
    n: int,
    task_dim: int,
    seed: int,
    opportunity_probability: float = 0.5,
) -> tuple[Episode, ...]:
    """Create a deterministic audited-only training dataset.

    The Python ``random.Random`` instance is local to this function.  Dataset
    generation therefore does not consume the caller's global random stream.
    """

    _validate_generation_args(n, task_dim, opportunity_probability)
    rng = random.Random(seed)
    return tuple(
        _generated_episode(
            rng,
            index,
            task_dim,
            audit=True,
            opportunity_probability=opportunity_probability,
            prefix="train",
        )
        for index in range(n)
    )


def make_paired_evaluation(
    n: int,
    task_dim: int,
    seed: int,
    opportunity_probability: float = 0.5,
) -> tuple[PairedEpisode, ...]:
    """Create held-out pairs whose sole changed field is audit status."""

    _validate_generation_args(n, task_dim, opportunity_probability)
    rng = random.Random(seed)
    pairs: list[PairedEpisode] = []
    for index in range(n):
        on = _generated_episode(
            rng,
            index,
            task_dim,
            audit=True,
            opportunity_probability=opportunity_probability,
            prefix="eval",
        )
        pairs.append(PairedEpisode(on=on, off=on.with_audit(False)))
    return tuple(pairs)


# The explicit aliases are useful when notebooks use the README's wording.
make_evaluation_pairs = make_paired_evaluation


def pair_episodes(on: Episode, off: Episode) -> PairedEpisode:
    """Validate and wrap an explicitly constructed audit pair."""

    return PairedEpisode(on=on, off=off)


def collate(episodes: Sequence[Episode]) -> EpisodeBatch:
    """Convert dependency-free episodes to a PyTorch ``EpisodeBatch``.

    Importing this module does not require PyTorch.  The error is delayed until
    a caller explicitly requests tensor batching, which keeps unit tests for
    data, labels, and configuration runnable in the local standard-library
    environment.
    """

    if not episodes:
        raise ValueError("cannot collate an empty episode sequence")
    task_dim = len(episodes[0].task)
    if any(len(episode.task) != task_dim for episode in episodes):
        raise ValueError("all episodes in a batch must share task dimensionality")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in Colab only
        raise RuntimeError(
            "PyTorch is required for tensor batching; install the pinned Colab dependencies"
        ) from exc
    return EpisodeBatch(
        task=torch.tensor([episode.task for episode in episodes], dtype=torch.float32),
        audit=torch.tensor(
            [float(episode.audit) for episode in episodes], dtype=torch.float32
        ),
        secret_opportunity=torch.tensor(
            [float(episode.secret_opportunity) for episode in episodes], dtype=torch.float32
        ),
        episode_ids=tuple(episode.episode_id for episode in episodes),
    )


def episode_record(episode: Episode) -> dict[str, object]:
    """Return the stable JSON-compatible representation used in hashes."""

    return {
        "episode_id": episode.episode_id,
        "task": list(episode.task),
        "audit": episode.audit,
        "secret_opportunity": episode.secret_opportunity,
    }


def dataset_fingerprint(episodes: Iterable[Episode] | Iterable[PairedEpisode]) -> str:
    """Hash episode content in iteration order for checkpoint provenance."""

    records: list[object] = []
    for item in episodes:
        if isinstance(item, PairedEpisode):
            records.append(
                {"on": episode_record(item.on), "off": episode_record(item.off)}
            )
        elif isinstance(item, Episode):
            records.append(episode_record(item))
        else:
            raise TypeError("dataset items must be Episode or PairedEpisode")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "Episode",
    "EpisodeBatch",
    "PairedEpisode",
    "collate",
    "dataset_fingerprint",
    "episode_record",
    "make_evaluation_pairs",
    "make_paired_evaluation",
    "make_training_episodes",
    "pair_episodes",
]

