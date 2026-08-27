from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.episodes import collate, make_training_episodes  # noqa: E402
from lean_reward_hacking.generic import (  # noqa: E402
    TORCH_AVAILABLE as GENERIC_TORCH_AVAILABLE,
    PlainMLPAgent,
    ablate_audit_feature,
    initialize_generic_agent,
)
from lean_reward_hacking.toy import (  # noqa: E402
    TORCH_AVAILABLE as TOY_TORCH_AVAILABLE,
    GoalGateAgent,
    initialize_toy_agent,
)


TORCH_AVAILABLE = TOY_TORCH_AVAILABLE and GENERIC_TORCH_AVAILABLE


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = collate(make_training_episodes(6, task_dim=3, seed=1729))

    def test_modular_output_and_initial_goal(self) -> None:
        import torch

        agent = initialize_toy_agent(
            task_dim=3, hidden_width=4, harmful_goal_strength=2.0, model_seed=11
        )
        logits = agent(self.batch)
        self.assertEqual(tuple(logits.shape), (6, 2))
        goal = agent.diagnostics(self.batch)["goal"]
        self.assertTrue(bool(torch.all(goal < 0).item()))

    def test_off_audit_gate_is_exactly_zero(self) -> None:
        import torch

        on = self.batch
        off = collate(make_training_episodes(6, task_dim=3, seed=1729))
        off.audit.zero_()
        on_logits = initialize_toy_agent(task_dim=3, hidden_width=4, model_seed=3)(on)
        agent = initialize_toy_agent(task_dim=3, hidden_width=4, model_seed=3)
        off_logits = agent(off)
        gate = agent.diagnostics(on)["gate"]
        self.assertTrue(
            torch.allclose(off_logits[:, 0], agent.goal_net(agent.split_logits(off)[0])[:, 0])
        )
        self.assertTrue(torch.allclose(on_logits[:, 0] - off_logits[:, 0], gate, atol=1e-6))

    def test_generic_control_has_no_named_goal_or_gate(self) -> None:
        agent = initialize_generic_agent(task_dim=3, hidden_width=4, model_seed=5)
        self.assertFalse(hasattr(agent, "goal_net"))
        self.assertFalse(hasattr(agent, "oversight_gate"))
        self.assertIsNone(agent.diagnostics(self.batch)["goal"])

    def test_audit_ablation_does_not_mutate_source(self) -> None:
        import torch

        source = self.batch.audit.clone()
        ablated = ablate_audit_feature(self.batch)
        self.assertTrue(torch.all(self.batch.audit == source).item())
        self.assertTrue(torch.all(ablated.audit == 0).item())

    def test_model_seed_reproduces_parameters(self) -> None:
        import torch

        left = initialize_toy_agent(task_dim=3, hidden_width=4, model_seed=19)
        right = initialize_toy_agent(task_dim=3, hidden_width=4, model_seed=19)
        for left_value, right_value in zip(left.state_dict().values(), right.state_dict().values()):
            self.assertTrue(torch.equal(left_value, right_value))

    def test_modular_forward_is_vmap_safe_and_differentiable(self) -> None:
        import torch

        if not hasattr(torch, "func"):
            self.skipTest("torch.func is unavailable")
        agent = initialize_toy_agent(task_dim=3, hidden_width=4, model_seed=23)
        parameters = {
            name: torch.stack((value.detach().clone(), value.detach().clone()))
            .requires_grad_(True)
            for name, value in agent.named_parameters()
        }
        buffers = dict(agent.named_buffers())

        def functional_forward(values):
            return torch.func.functional_call(agent, (values, buffers), (self.batch,))

        vmapped = torch.func.vmap(functional_forward)(parameters)
        expected = torch.stack((agent(self.batch), agent(self.batch)))
        self.assertTrue(torch.allclose(vmapped, expected, atol=1e-6, rtol=1e-6))
        gradients = torch.autograd.grad(vmapped.square().mean(), tuple(parameters.values()))
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))

    def test_generic_forward_is_vmap_compatible(self) -> None:
        import torch

        if not hasattr(torch, "func"):
            self.skipTest("torch.func is unavailable")
        agent = initialize_generic_agent(task_dim=3, hidden_width=4, model_seed=29)
        parameters = {
            name: torch.stack((value.detach().clone(), value.detach().clone()))
            for name, value in agent.named_parameters()
        }
        buffers = dict(agent.named_buffers())

        def functional_forward(values):
            return torch.func.functional_call(agent, (values, buffers), (self.batch,))

        vmapped = torch.func.vmap(functional_forward)(parameters)
        expected = torch.stack((agent(self.batch), agent(self.batch)))
        self.assertTrue(torch.allclose(vmapped, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
