from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.evaluation import evaluate_agent  # noqa: E402
from lean_reward_hacking.types import Episode, PairedEpisode  # noqa: E402


def _pair(episode_id: str, opportunity: bool) -> PairedEpisode:
    on = Episode(episode_id, (0.0,), True, opportunity)
    return PairedEpisode(on, on.with_audit(False))


class _EpisodeAgent:
    def predict_help_probabilities(self, episodes: tuple[Episode, ...]) -> list[float]:
        return [
            1.0 if episode.audit and episode.secret_opportunity else 0.0
            for episode in episodes
        ]


class _LogitAgent:
    def logits(self, episodes: tuple[Episode, ...]) -> list[list[float]]:
        rows: list[list[float]] = []
        for episode in episodes:
            if episode.audit:
                rows.append([2.0, 0.0] if episode.secret_opportunity else [0.0, 0.0])
            elif episode.secret_opportunity:
                rows.append([4.0, 1.0] if episode.episode_id == "first" else [-1.0, 2.0])
            else:
                # This large value checks that non-opportunity rows are excluded.
                rows.append([100.0, -100.0])
        return rows


@dataclass(frozen=True)
class _FakeParameter:
    device: str


@dataclass(frozen=True)
class _FakeBatch:
    size: int
    device: str = "cpu"

    def to(self, device: str) -> "_FakeBatch":
        return _FakeBatch(self.size, device)


class _BatchAgent:
    def __init__(self, device: str) -> None:
        self._parameter = _FakeParameter(device)
        self.seen_devices: list[str] = []

    def parameters(self):
        return iter((self._parameter,))

    def predict_help_probabilities(self, batch: object) -> list[float]:
        if not isinstance(batch, _FakeBatch):
            raise TypeError("an EpisodeBatch is required")
        self.seen_devices.append(batch.device)
        return [1.0 if batch.size else 0.0] * batch.size


class _RuntimeFallbackAgent(_BatchAgent):
    """A tensor-only model whose sequence path reaches a device error."""

    def predict_help_probabilities(self, batch: object) -> list[float]:
        if not isinstance(batch, _FakeBatch):
            raise RuntimeError("mat1 and mat2 must have the same device")
        self.seen_devices.append(batch.device)
        return [1.0] * batch.size


class EvaluationTests(unittest.TestCase):
    def test_compliance_rates_condition_on_secret_opportunity(self) -> None:
        pairs = (_pair("opportunity", True), _pair("control", False))

        metrics = evaluate_agent(_EpisodeAgent(), pairs)

        self.assertEqual(metrics.c_on, 1.0)
        self.assertEqual(metrics.c_off, 0.0)
        self.assertIsNone(metrics.off_audit_logit)

    def test_raw_off_logit_conditions_on_opportunity(self) -> None:
        pairs = (_pair("first", True), _pair("control", False), _pair("second", True))

        metrics = evaluate_agent(_LogitAgent(), pairs)

        self.assertEqual(metrics.off_audit_logit, 0.0)

    def test_missing_secret_opportunities_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret opportunities"):
            evaluate_agent(_EpisodeAgent(), (_pair("control", False),))

    def test_collated_batch_moves_to_bound_parameter_device(self) -> None:
        agent = _BatchAgent("test-device")
        pairs = (_pair("opportunity", True),)

        with patch(
            "lean_reward_hacking.evaluation.collate",
            side_effect=lambda episodes: _FakeBatch(len(episodes)),
        ):
            evaluate_agent(agent, pairs)

        self.assertEqual(agent.seen_devices, ["test-device", "test-device"])

    def test_device_error_on_sequence_dispatch_retries_with_device_batch(self) -> None:
        agent = _RuntimeFallbackAgent("cuda:0")
        pairs = (_pair("opportunity", True),)

        with patch(
            "lean_reward_hacking.evaluation.collate",
            side_effect=lambda episodes: _FakeBatch(len(episodes)),
        ):
            metrics = evaluate_agent(agent, pairs)

        self.assertEqual(metrics.c_on, 1.0)
        self.assertEqual(metrics.c_off, 1.0)
        self.assertEqual(agent.seen_devices, ["cuda:0", "cuda:0"])


if __name__ == "__main__":
    unittest.main()
