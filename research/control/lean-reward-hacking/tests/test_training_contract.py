from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.episodes import collate, make_training_episodes  # noqa: E402
from lean_reward_hacking.rewards import RewardConfig  # noqa: E402
from lean_reward_hacking.safety import ContractViolation  # noqa: E402
from lean_reward_hacking.toy import TORCH_AVAILABLE, initialize_toy_agent  # noqa: E402
from lean_reward_hacking.training import (  # noqa: E402
    TrainingConfig,
    expected_action_reward,
    expected_two_action_loss,
    fit_replicate,
    require_colab_for_full_run,
    train_step,
)


class TrainingContractTests(unittest.TestCase):
    def test_full_run_requires_colab_marker(self) -> None:
        with self.assertRaises(ContractViolation):
            require_colab_for_full_run(TrainingConfig(steps=9, execution="local_smoke"))
        require_colab_for_full_run(TrainingConfig(steps=2, execution="local_smoke"))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
    def test_expected_reward_uses_two_action_order(self) -> None:
        import torch

        logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        expected = expected_action_reward(logits)
        self.assertTrue(
            torch.allclose(
                expected,
                torch.tensor([0.0, 2 * torch.sigmoid(torch.tensor(2.0)) - 1]),
            )
        )
        loss = expected_two_action_loss(logits, kl_coefficient=0.0, l2_coefficient=0.0)
        self.assertTrue(torch.allclose(loss, -expected.mean()))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
    def test_training_rejects_audit_off_rows(self) -> None:
        import torch

        from lean_reward_hacking.episodes import make_paired_evaluation

        agent = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=7)
        optimizer = torch_optimizer(agent)
        batch = collate([pair.off for pair in make_paired_evaluation(2, 2, seed=10)])
        with self.assertRaises(ValueError):
            train_step(agent, optimizer, batch, RewardConfig())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
    def test_callbacks_fire_on_checkpoint_steps(self) -> None:
        agent = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=8)
        data = collate(make_training_episodes(4, task_dim=2, seed=12))
        config = TrainingConfig(
            steps=2,
            batch_size=2,
            checkpoint_every_steps=1,
            execution="local_smoke",
            device="cpu",
            sampler_seed=4,
        )
        checkpoint_steps: list[int] = []
        eval_steps: list[int] = []

        def checkpoint(*, state, **_):
            checkpoint_steps.append(state.global_step)

        def evaluate(*, state, **_):
            eval_steps.append(state.global_step)

        state = fit_replicate(
            agent,
            data,
            config,
            optimizer=torch_optimizer(agent),
            checkpoint_callback=checkpoint,
            eval_callback=evaluate,
        )
        self.assertEqual(state.global_step, 2)
        self.assertEqual(checkpoint_steps, [1, 2])
        self.assertEqual(eval_steps, [1, 2])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
    def test_checkpoint_callback_can_write_a_tiny_file(self) -> None:
        agent = initialize_toy_agent(task_dim=2, hidden_width=3, model_seed=9)
        data = collate(make_training_episodes(4, task_dim=2, seed=14))
        config = TrainingConfig(steps=1, batch_size=2, execution="local_smoke", device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            state = fit_replicate(
                agent,
                data,
                config,
                optimizer=torch_optimizer(agent),
                checkpoint_dir=temporary,
            )
            self.assertEqual(state.global_step, 1)
            self.assertTrue((Path(temporary) / "checkpoint_00000001.pt").is_file())


def torch_optimizer(agent):
    import torch

    return torch.optim.Adam(agent.parameters(), lr=0.003)


if __name__ == "__main__":
    unittest.main()
