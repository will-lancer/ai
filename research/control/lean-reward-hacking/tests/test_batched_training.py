from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.batched_training import (  # noqa: E402
    BatchedTrainingConfig,
    TORCH_AVAILABLE,
    initialize_batched_replicas,
)
from lean_reward_hacking.training import train_step  # noqa: E402


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class BatchedTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._runtime = mock.patch.dict("os.environ", {"LRH_RUNTIME": "colab"})
        self._runtime.start()

    def tearDown(self) -> None:
        self._runtime.stop()

    def _features(self):
        import torch

        # Columns are task, audit, secret opportunity.  Every training row is audited.
        return torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [-0.5, 1.0, 1.0],
                [0.25, 1.0, 0.0],
                [0.75, 1.0, 1.0],
                [1.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def _bank(self, replicas: int):
        return initialize_batched_replicas(
            lambda **_: _TinyMLP(),
            replicas=replicas,
            model_seeds=tuple(41 + index for index in range(replicas)),
            sampler_seeds=tuple(91 + index for index in range(replicas)),
        )

    def test_one_step_matches_independent_adam_for_one_and_many_replicas(self) -> None:
        import torch

        features = self._features()
        for replica_count in (1, 3):
            bank = self._bank(replica_count)
            independent = bank.materialize_replica(0)
            agents = [bank.materialize_replica(index) for index in range(replica_count)]
            optimizers = [torch.optim.Adam(agent.parameters(), lr=0.003) for agent in agents]
            metrics = bank.step(
                features,
                kl_coefficient=0.02,
                l2_coefficient=1.0e-4,
                grad_clip_norm=0.7,
            )
            for index, (agent, optimizer) in enumerate(zip(agents, optimizers)):
                expected = train_step(
                    agent,
                    optimizer,
                    features,
                    kl_coefficient=0.02,
                    l2_coefficient=1.0e-4,
                    grad_clip_norm=0.7,
                )
                for name, value in agent.named_parameters():
                    stacked = bank.parameters_by_name[name][index]
                    self.assertTrue(torch.equal(stacked, value), name)
                for name, value in expected.items():
                    self.assertAlmostEqual(float(metrics[name][index]), value, places=6)
            # Keep this local variable as a smoke check that slice materialization is independent.
            self.assertIsNot(independent, agents[0])

    def test_sampler_orders_are_per_replica_and_cursor_is_serializable(self) -> None:
        import torch

        bank = self._bank(2)
        state = bank.train(
            self._features(),
            BatchedTrainingConfig(steps=3, batch_size=2, checkpoint_every_steps=3),
        )
        self.assertEqual(tuple(state.permutations.shape), (2, 5))
        expected = []
        for seed in bank.sampler_seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            expected.append(torch.randperm(5, generator=generator))
        self.assertTrue(torch.equal(state.permutations, torch.stack(expected)))
        payload = state.to_payload()
        restored = type(state).from_payload(payload)
        self.assertTrue(torch.equal(restored.permutations, state.permutations))
        for left, right in zip(restored.sampler_states, state.sampler_states):
            self.assertTrue(torch.equal(left, right))

    def test_history_is_bounded_by_checkpoint_cadence(self) -> None:
        bank = self._bank(2)
        state = bank.train(
            self._features(),
            BatchedTrainingConfig(steps=7, batch_size=2, checkpoint_every_steps=3),
        )
        self.assertEqual([row["step"] for row in state.history], [3, 6, 7])

    def test_disabled_clipping_matches_scalar_zero_metric(self) -> None:
        bank = self._bank(2)
        metrics = bank.step(self._features(), grad_clip_norm=None)
        self.assertEqual(metrics["grad_norm"].tolist(), [0.0, 0.0])

    def test_full_bank_training_requires_colab_marker(self) -> None:
        from lean_reward_hacking.safety import ContractViolation

        bank = self._bank(2)
        with mock.patch.dict(
            "os.environ",
            {"LRH_RUNTIME": "", "COLAB_RELEASE_TAG": ""},
        ):
            with self.assertRaises(ContractViolation):
                bank.train(
                    self._features(),
                    BatchedTrainingConfig(steps=9, batch_size=2),
                )

    def test_checkpoint_resume_matches_uninterrupted_bank(self) -> None:
        features = self._features()
        config_first = BatchedTrainingConfig(
            steps=2, batch_size=2
        )
        config_full = BatchedTrainingConfig(steps=5, batch_size=2)
        with tempfile.TemporaryDirectory() as temporary:
            first = self._bank(2)
            first_state = first.train(features, config_first)
            path = first.save_checkpoint(Path(temporary) / "checkpoint.pt", state=first_state)

            resumed = self._bank(2)
            resumed_state, _ = resumed.load_checkpoint(path)
            resumed_state = resumed.train(features, config_full, state=resumed_state)

            continuous = self._bank(2)
            continuous_state = continuous.train(features, config_full)

        self.assertEqual(resumed_state.history, continuous_state.history)
        for name, value in continuous.parameters_by_name.items():
            self.assertTrue(value.equal(resumed.parameters_by_name[name]), name)

    def test_manifest_and_scalar_slice_are_stable(self) -> None:
        import torch

        bank = self._bank(2)
        manifest = bank.manifest()
        self.assertEqual(manifest["replica_count"], 2)
        self.assertEqual(tuple(manifest["parameter_names"]), tuple(bank.parameters_by_name))
        self.assertEqual([spec.index for spec in bank.replica_specs()], [0, 1])
        scalar = bank.materialize_replica(1)
        scalar_logits = scalar(self._features())
        bank_logits = bank.logits(self._features())[1]
        self.assertTrue(torch.equal(scalar_logits, bank_logits))


class _TinyMLP:
    pass


if TORCH_AVAILABLE:
    import torch
    from torch import nn

    class _TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))

        def forward(self, features):
            return self.net(features)


if __name__ == "__main__":
    unittest.main()
