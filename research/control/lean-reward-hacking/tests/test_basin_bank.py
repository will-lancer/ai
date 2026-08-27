from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.basin_bank import (
    BASIN_BANK_SCHEMA_VERSION,
    build_basin_specs,
    densify_boundary_cells,
    run_basin_bank,
    select_refinement_cells,
)
from lean_reward_hacking.batched_training import TORCH_AVAILABLE
from lean_reward_hacking.episodes import (
    dataset_fingerprint,
    make_paired_evaluation,
    make_training_episodes,
)
from lean_reward_hacking.evaluation import ModeThresholds
from lean_reward_hacking.safety import ContractViolation


class BasinBankStaticTests(unittest.TestCase):
    def test_coarse_specs_are_stable_and_grid_complete(self) -> None:
        values = {
            "harmful_goal_grid": [0.5, 1.0],
            "audit_sensitivity_grid": [-1.0, 0.0],
            "seeds_per_cell": 2,
        }
        specs = build_basin_specs(values)
        self.assertEqual(len(specs), 8)
        self.assertEqual(specs[0].run_id, "basin-h0-a0-r0")
        self.assertEqual(specs[-1].run_id, "basin-h1-a1-r1")
        self.assertEqual(
            {(spec.harm_strength, spec.audit_sensitivity) for spec in specs},
            {(0.5, -1.0), (0.5, 0.0), (1.0, -1.0), (1.0, 0.0)},
        )
        self.assertEqual(
            [spec.model_seed for spec in specs[:2]],
            [500_000, 500_001],
        )

    def test_refinement_starts_after_existing_cell_replicas(self) -> None:
        values = {
            "harmful_goal_grid": [0.5],
            "audit_sensitivity_grid": [0.0],
            "seeds_per_cell": 2,
            "refinement_seeds": 3,
        }
        specs = build_basin_specs(
            values,
            phase="refinement",
            level=1,
            cells=[(0.5, 0.0)],
            seed_starts={(0.5, 0.0): 2},
        )
        self.assertEqual([spec.seed_index for spec in specs], [2, 3, 4])
        self.assertEqual(specs[0].run_id, "basin-refine1-h0-a0-r2")
        self.assertEqual(specs[0].model_seed, 900_002)

    def test_refinement_uses_uncertainty_and_neighbor_jumps(self) -> None:
        finals = {
            (0.5, -1.0): [{"label": "strategic"}] * 8,
            (0.5, 0.0): [{"label": "oversight-invariant"}] * 4 + [{"label": "strategic"}] * 4,
            (1.0, -1.0): [{"label": "strategic"}] * 8,
            (1.0, 0.0): [{"label": "strategic"}] * 8,
        }
        selected = select_refinement_cells(
            finals,
            harmful_goal_grid=[0.5, 1.0],
            audit_sensitivity_grid=[-1.0, 0.0],
            probability_low=0.20,
            probability_high=0.80,
            neighbor_delta=0.40,
        )
        self.assertIn((0.5, 0.0), selected)
        self.assertIn((0.5, -1.0), selected)

    def test_refinement_considers_strategic_and_intermediate_transitions(self) -> None:
        finals = {
            (0.5, 0.0): [{"label": "strategic"}] * 8,
            (1.0, 0.0): [{"label": "intermediate"}] * 8,
        }
        selected = select_refinement_cells(
            finals,
            harmful_goal_grid=[0.5, 1.0],
            audit_sensitivity_grid=[0.0],
            probability_low=0.20,
            probability_high=0.80,
            neighbor_delta=0.40,
        )
        self.assertEqual(set(selected), {(0.5, 0.0), (1.0, 0.0)})

    def test_refinement_midpoints_are_new_coordinates_and_restartable(self) -> None:
        finals = {
            (0.0, 0.0): [{"label": "strategic"}] * 8,
            (1.0, 0.0): [{"label": "intermediate"}] * 8,
        }
        cells = densify_boundary_cells(
            finals,
            [(0.0, 0.0), (1.0, 0.0)],
            harmful_goal_grid=[0.0, 1.0],
            audit_sensitivity_grid=[0.0],
        )
        self.assertIn((0.5, 0.0), cells)
        self.assertNotIn((0.0, 0.0), cells)
        self.assertEqual(cells, densify_boundary_cells(
            finals,
            [(1.0, 0.0), (0.0, 0.0)],
            harmful_goal_grid=[0.0, 1.0],
            audit_sensitivity_grid=[0.0],
        ))

    def test_external_refinement_coordinates_have_stable_cell_identity(self) -> None:
        values = {
            "harmful_goal_grid": [0.0, 1.0],
            "audit_sensitivity_grid": [0.0],
            "refinement_seeds": 2,
        }
        first = build_basin_specs(
            values,
            phase="refinement",
            level=1,
            cells=[(0.5, 0.0)],
        )
        second = build_basin_specs(
            values,
            phase="refinement",
            level=1,
            cells=[(0.5, 0.0)],
        )
        self.assertEqual(first, second)
        self.assertTrue(first[0].cell_token)
        self.assertEqual(first[0].cell, (0.5, 0.0))
        self.assertNotEqual(first[0].run_id, "basin-refine1-h0-a0-r0")

    def test_public_basin_api_guards_before_bank_construction(self) -> None:
        values = {
            "harmful_goal_grid": [1.0],
            "audit_sensitivity_grid": [0.0],
            "device": "cuda",
        }
        train = make_training_episodes(2, 2, 17)
        pairs = make_paired_evaluation(2, 2, 19, opportunity_probability=1.0)
        with mock.patch.dict(os.environ, {"LRH_RUNTIME": ""}, clear=False):
            with mock.patch("lean_reward_hacking.basin_bank._run_phase_bank") as phase:
                with self.assertRaises(ContractViolation):
                    run_basin_bank(
                        session_dir=Path(tempfile.mkdtemp()),
                        values=values,
                        config_identity="config",
                        source_identity="source",
                        target_steps=1,
                        train_episodes=train,
                        eval_pairs=pairs,
                    )
                phase.assert_not_called()

    def test_public_basin_api_recomputes_data_hashes_before_training(self) -> None:
        values = {
            "harmful_goal_grid": [1.0],
            "audit_sensitivity_grid": [0.0],
            "device": "cpu",
        }
        train = make_training_episodes(2, 2, 17)
        pairs = make_paired_evaluation(2, 2, 19, opportunity_probability=1.0)
        with mock.patch.dict(os.environ, {"LRH_RUNTIME": "colab"}, clear=False):
            with mock.patch("lean_reward_hacking.basin_bank._run_phase_bank") as phase:
                with self.assertRaisesRegex(Exception, "dataset hash"):
                    run_basin_bank(
                        session_dir=Path(tempfile.mkdtemp()),
                        values=values,
                        config_identity="config",
                        source_identity="source",
                        target_steps=1,
                        train_episodes=train,
                        eval_pairs=pairs,
                        dataset_hash="stale-dataset",
                    )
                phase.assert_not_called()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in Colab only")
class BasinBankTorchTests(unittest.TestCase):
    def test_tiny_basin_bank_is_restartable(self) -> None:
        values = {
            "harmful_goal_grid": [1.0, 2.0],
            "audit_sensitivity_grid": [0.0],
            "seeds_per_cell": 2,
            "refinement_seeds": 2,
            "max_refinement_levels": 0,
            "task_dim": 2,
            "hidden_width": 4,
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
        arguments = {
            "session_dir": Path(tempfile.mkdtemp()),
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
        try:
            with mock.patch.dict(os.environ, {"LRH_RUNTIME": "colab"}):
                first = run_basin_bank(**arguments)
                second = run_basin_bank(**arguments)
        finally:
            # The temporary directory is intentionally left to the platform
            # cleanup.  This test only exercises the resumable API.
            pass
        self.assertEqual(BASIN_BANK_SCHEMA_VERSION, 1)
        self.assertEqual(len(first.tables["runs.csv"]), 4)
        self.assertEqual(first.tables["basin_cells.csv"], second.tables["basin_cells.csv"])
        self.assertEqual(first.tables["final_summary.csv"], second.tables["final_summary.csv"])
        self.assertTrue(all(row["status"] == "complete" for row in first.tables["runs.csv"]))


if __name__ == "__main__":
    unittest.main()
