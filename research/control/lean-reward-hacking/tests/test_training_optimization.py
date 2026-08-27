from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.episodes import collate, make_training_episodes  # noqa: E402
from lean_reward_hacking.rewards import RewardConfig  # noqa: E402
from lean_reward_hacking.toy import TORCH_AVAILABLE, initialize_toy_agent  # noqa: E402
from lean_reward_hacking.training import (  # noqa: E402
    TrainingConfig,
    fit_replicate,
    loss_terms,
    train_step,
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class TrainingOptimizationTests(unittest.TestCase):
    def _legacy_train_step(self, agent, optimizer, batch):
        import torch

        optimizer.zero_grad(set_to_none=True)
        logits = agent(batch)
        terms = loss_terms(
            logits,
            help_reward=1.0,
            harm_reward=-1.0,
            kl_coefficient=0.02,
            l2_coefficient=1.0e-4,
            parameters=agent.parameters(),
        )
        terms["loss"].backward()
        grad_norm = torch.zeros((), device=logits.device)
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
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

    def test_one_step_matches_prior_parameter_and_metric_semantics(self) -> None:
        import torch

        batch = collate(make_training_episodes(4, task_dim=2, seed=123))
        optimized = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=9)
        legacy = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=9)
        optimizer = torch.optim.Adam(optimized.parameters(), lr=0.003)
        legacy_optimizer = torch.optim.Adam(legacy.parameters(), lr=0.003)

        optimized_metrics = train_step(optimized, optimizer, batch, RewardConfig())
        legacy_metrics = self._legacy_train_step(legacy, legacy_optimizer, batch)

        self.assertEqual(optimized_metrics, legacy_metrics)
        for optimized_value, legacy_value in zip(
            optimized.state_dict().values(), legacy.state_dict().values()
        ):
            self.assertTrue(torch.equal(optimized_value, legacy_value))

    def test_full_batch_transfer_happens_once(self) -> None:
        import unittest.mock

        import lean_reward_hacking.training as training

        agent = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=10)
        data = collate(make_training_episodes(5, task_dim=2, seed=124))
        config = TrainingConfig(
            steps=3,
            batch_size=2,
            execution="local_smoke",
            device="cpu",
            checkpoint_every_steps=3,
        )
        with unittest.mock.patch.object(
            training, "_move_batch", wraps=training._move_batch
        ) as move_batch:
            state = fit_replicate(agent, data, config)
        self.assertEqual(move_batch.call_count, 1)
        self.assertEqual(state.permutation.device.type, "cpu")

    def test_resume_is_bitwise_equal_to_uninterrupted_training(self) -> None:
        import torch

        data = collate(make_training_episodes(5, task_dim=2, seed=125))
        first_config = TrainingConfig(
            steps=2,
            batch_size=2,
            sampler_seed=41,
            execution="local_smoke",
            device="cpu",
            checkpoint_every_steps=2,
        )
        full_config = TrainingConfig(
            steps=5,
            batch_size=2,
            sampler_seed=41,
            execution="local_smoke",
            device="cpu",
            checkpoint_every_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=11)
            first_state = fit_replicate(first, data, first_config, checkpoint_dir=temporary)
            checkpoint = Path(temporary) / "checkpoint_00000002.pt"

            resumed = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=11)
            resumed_state = fit_replicate(
                resumed,
                data,
                full_config,
                optimizer=torch.optim.Adam(resumed.parameters(), lr=0.003),
                resume_from=checkpoint,
            )
            continuous = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=11)
            continuous_state = fit_replicate(continuous, data, full_config)

        self.assertEqual(first_state.global_step, 2)
        self.assertEqual(resumed_state.history, continuous_state.history)
        for resumed_value, continuous_value in zip(
            resumed.state_dict().values(), continuous.state_dict().values()
        ):
            self.assertTrue(torch.equal(resumed_value, continuous_value))


if __name__ == "__main__":
    unittest.main()
