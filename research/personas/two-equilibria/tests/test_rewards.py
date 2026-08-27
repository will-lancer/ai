from __future__ import annotations

import unittest

from lean_reward_hacking.episodes import Episode
from lean_reward_hacking.rewards import RewardConfig, fixed_rlhf_reward, reward_for_action
from lean_reward_hacking.types import Action


class RewardTests(unittest.TestCase):
    def test_scalar_rewards_follow_action_order(self) -> None:
        config = RewardConfig(help_reward=3.0, harm_reward=-2.0)
        self.assertEqual(reward_for_action(Action.HELP, config), 3.0)
        self.assertEqual(reward_for_action(Action.HARM, config), -2.0)
        self.assertEqual(fixed_rlhf_reward(0, config), 3.0)
        self.assertEqual(fixed_rlhf_reward(1, config), -2.0)

    def test_sequence_rewards_are_deterministic(self) -> None:
        self.assertEqual(fixed_rlhf_reward([Action.HELP, Action.HARM]), (1.0, -1.0))

    def test_reward_ignores_episode_fields(self) -> None:
        on = Episode("x", (1.0,), True, True)
        off = on.with_audit(False)
        self.assertEqual(
            fixed_rlhf_reward(Action.HELP, episodes=(on,)),
            fixed_rlhf_reward(Action.HELP, episodes=(off,)),
        )
        self.assertEqual(
            fixed_rlhf_reward(Action.HARM, episodes=(on,)),
            fixed_rlhf_reward(Action.HARM, episodes=(off,)),
        )

    def test_invalid_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fixed_rlhf_reward(2)
        with self.assertRaises(ValueError):
            fixed_rlhf_reward([Action.HELP, 9])


if __name__ == "__main__":
    unittest.main()

