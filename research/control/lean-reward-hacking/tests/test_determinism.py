from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.episodes import collate, make_training_episodes  # noqa: E402
from lean_reward_hacking.toy import TORCH_AVAILABLE, initialize_toy_agent  # noqa: E402
from lean_reward_hacking.training import (  # noqa: E402
    TrainingConfig,
    epoch_permutations,
    fit_replicate,
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class DeterminismTests(unittest.TestCase):
    def _config(self, sampler_seed: int = 21, steps: int = 4) -> TrainingConfig:
        return TrainingConfig(
            steps=steps,
            batch_size=2,
            checkpoint_every_steps=2,
            model_seed=17,
            sampler_seed=sampler_seed,
            execution="local_smoke",
            device="cpu",
        )

    def _data(self):
        return collate(make_training_episodes(4, task_dim=2, seed=31))

    def test_epoch_permutations_are_seeded(self) -> None:
        left = epoch_permutations(7, 3, sampler_seed=4)
        right = epoch_permutations(7, 3, sampler_seed=4)
        different = epoch_permutations(7, 3, sampler_seed=5)
        for left_perm, right_perm in zip(left, right):
            self.assertTrue((left_perm == right_perm).all().item())
        self.assertTrue(any(not (a == b).all().item() for a, b in zip(left, different)))

    def test_same_model_and_sampler_seeds_match(self) -> None:
        import torch

        left = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
        right = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
        left_state = fit_replicate(left, self._data(), self._config())
        right_state = fit_replicate(right, self._data(), self._config())
        self.assertEqual(left_state.history, right_state.history)
        for left_value, right_value in zip(left.state_dict().values(), right.state_dict().values()):
            self.assertTrue(torch.equal(left_value, right_value))

    def test_sampler_seed_changes_order_metadata(self) -> None:
        left = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
        right = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
        left_state = fit_replicate(left, self._data(), self._config(sampler_seed=21))
        right_state = fit_replicate(right, self._data(), self._config(sampler_seed=22))
        self.assertNotEqual(left_state.history, right_state.history)

    def test_checkpoint_resume_matches_uninterrupted_smoke(self) -> None:
        import torch

        data = self._data()
        with tempfile.TemporaryDirectory() as temporary:
            first = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
            first_state = fit_replicate(
                first,
                data,
                self._config(sampler_seed=21, steps=2),
                checkpoint_dir=temporary,
            )
            checkpoint = Path(temporary) / "checkpoint_00000002.pt"
            self.assertTrue(checkpoint.is_file())

            resumed = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
            resumed_state = fit_replicate(
                resumed,
                data,
                self._config(sampler_seed=21, steps=4),
                optimizer=torch.optim.Adam(resumed.parameters(), lr=0.003),
                resume_from=checkpoint,
            )
            continuous = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=17)
            continuous_state = fit_replicate(
                continuous,
                data,
                self._config(sampler_seed=21, steps=4),
            )
            self.assertEqual(first_state.global_step, 2)
            self.assertEqual(resumed_state.history, continuous_state.history)
            for resumed_value, continuous_value in zip(
                resumed.state_dict().values(), continuous.state_dict().values()
            ):
                self.assertTrue(torch.equal(resumed_value, continuous_value))


if __name__ == "__main__":
    unittest.main()
