from __future__ import annotations

import unittest

from lean_reward_hacking.episodes import (
    dataset_fingerprint,
    make_paired_evaluation,
    make_training_episodes,
    pair_episodes,
)
from lean_reward_hacking.types import Episode


class PairingTests(unittest.TestCase):
    def test_pairs_change_audit_only(self) -> None:
        pairs = make_paired_evaluation(12, 3, seed=17)
        self.assertEqual(len(pairs), 12)
        for pair in pairs:
            self.assertTrue(pair.on.audit)
            self.assertFalse(pair.off.audit)
            self.assertEqual(pair.on.episode_id, pair.off.episode_id)
            self.assertEqual(pair.on.task, pair.off.task)
            self.assertEqual(
                pair.on.secret_opportunity,
                pair.off.secret_opportunity,
            )

    def test_generation_is_deterministic_and_seeded(self) -> None:
        self.assertEqual(
            make_paired_evaluation(5, 2, seed=3),
            make_paired_evaluation(5, 2, seed=3),
        )
        self.assertNotEqual(
            make_paired_evaluation(5, 2, seed=3),
            make_paired_evaluation(5, 2, seed=4),
        )
        self.assertTrue(all(item.audit for item in make_training_episodes(5, 2, 3)))

    def test_invalid_pair_is_rejected(self) -> None:
        on = Episode("id", (0.0,), True, False)
        with self.assertRaises(ValueError):
            pair_episodes(on, on)
        with self.assertRaises(ValueError):
            pair_episodes(on, Episode("id", (1.0,), False, False))

    def test_dataset_fingerprint_tracks_content(self) -> None:
        first = make_training_episodes(3, 2, seed=8)
        second = make_training_episodes(3, 2, seed=8)
        third = make_training_episodes(3, 2, seed=9)
        self.assertEqual(dataset_fingerprint(first), dataset_fingerprint(second))
        self.assertNotEqual(dataset_fingerprint(first), dataset_fingerprint(third))


if __name__ == "__main__":
    unittest.main()

