from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.bank_campaign import _vector, run_fixed_bank
from lean_reward_hacking.batched_training import TORCH_AVAILABLE
from lean_reward_hacking.episodes import (
    dataset_fingerprint,
    make_paired_evaluation,
    make_training_episodes,
)
from lean_reward_hacking.evaluation import ModeThresholds


class BankCampaignStaticTests(unittest.TestCase):
    def test_metric_vector_requires_one_value_per_replica(self) -> None:
        self.assertEqual(_vector(1.5, 2), [1.5, 1.5])
        self.assertEqual(_vector([1, 2], 2), [1.0, 2.0])
        with self.assertRaises(ValueError):
            _vector([1], 2)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class BankCampaignTorchTests(unittest.TestCase):
    def test_tiny_fixed_bank_is_restartable_and_schema_complete(self) -> None:
        values = {
            "replicas": 2,
            "model_seed_base": 111,
            "sampler_seed_base": 222,
            "task_dim": 2,
            "hidden_width": 4,
            "harmful_goal_strength": 2.0,
            "initial_audit_sensitivity": 0.0,
            "steps": 2,
            "batch_size": 2,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "entropy_coefficient": 0.02,
            "grad_clip_norm": 1.0,
            "checkpoint_every": 1,
            "device": "cpu",
            "execution": "colab",
        }
        train = make_training_episodes(8, 2, 17)
        pairs = make_paired_evaluation(8, 2, 19, opportunity_probability=1.0)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"LRH_RUNTIME": "colab"}
        ):
            arguments = {
                "session_dir": Path(temporary),
                "experiment": "toy_fixed",
                "architecture": "toy",
                "values": values,
                "config_identity": "config",
                "source_identity": "source",
                "target_steps": 2,
                "train_episodes": train,
                "eval_pairs": pairs,
                "dataset_hash": dataset_fingerprint(train),
                "eval_hash": dataset_fingerprint(pairs),
                "thresholds": ModeThresholds(),
            }
            first = run_fixed_bank(**arguments)
            second = run_fixed_bank(**arguments)

        self.assertEqual(len(first.tables["runs.csv"]), 2)
        self.assertEqual(len(first.tables["checkpoint_metrics.csv"]), 4)
        self.assertEqual(first.tables["final_summary.csv"], second.tables["final_summary.csv"])
        self.assertTrue(all(row["status"] == "complete" for row in first.tables["runs.csv"]))


if __name__ == "__main__":
    unittest.main()
